"""Separate A/B identification and C detection-plus-identification evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import math
from typing import Any

from ..evaluation.matcher import CharacterMatcher
from ..evaluation.scoring import mcnemar, wilson_ci
from .parsing import ParsedMultiResponse, box_iou, parse_response
from .prompts import PROMPT_VERSION
from .stratification import load_frozen_stratification, validate_stratification, stratification_hash


def get(obj: Any, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def identity_category(person: Any) -> str:
    if get(person, "review_status", "pending") != "approved":
        return "pending"
    status = get(person, "identity_status", "needs_review")
    if status == "verified" and get(person, "character"):
        return "verified"
    return "unknown" if status == "unknown" else "pending"


def maximum_iou_matching(gt_boxes: list[list[float]], predictions: list[list[float]],
                         threshold: float = .5) -> list[tuple[int, int, float]]:
    """Hungarian assignment: maximum cardinality, then maximum total IoU.

    Matching uses geometry only. A correct name cannot rescue the wrong box.
    """
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must be in (0,1]")
    if not gt_boxes or not predictions:
        return []
    n = max(len(gt_boxes), len(predictions))
    ious = [[box_iou(a, b) for b in predictions] for a in gt_boxes]
    # One extra edge beats any possible difference in all IoU tie-breakers.
    costs = [[-(n + 1 + ious[i][j]) if i < len(gt_boxes) and j < len(predictions)
              and ious[i][j] >= threshold else 0. for j in range(n)] for i in range(n)]
    u, v, p, way = [0.] * (n + 1), [0.] * (n + 1), [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minimum, used = [math.inf] * (n + 1), [False] * (n + 1)
        while True:
            used[j0], i0, delta, j1 = True, p[j0], math.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j], way[j] = current, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0], j0 = p[j1], j1
            if j0 == 0:
                break
    result = []
    for j in range(1, n + 1):
        i = p[j] - 1
        if 0 <= i < len(gt_boxes) and j - 1 < len(predictions) and ious[i][j - 1] >= threshold:
            result.append((i, j - 1, ious[i][j - 1]))
    return sorted(result)


def overlaps_ignore(box: list[float], regions: list[Any], threshold: float) -> bool:
    area = (box[2] - box[0]) * (box[3] - box[1])
    for region in regions:
        other = region if isinstance(region, list) else get(region, "bbox")
        if not other:
            continue
        intersection = max(0., min(box[2], other[2]) - max(box[0], other[0])) * max(0., min(box[3], other[3]) - max(box[1], other[1]))
        if area and intersection / area >= threshold:
            return True
    return False


def classify(name: str | None, person: Any, slug: str, matcher: CharacterMatcher) -> tuple[str, bool]:
    if name is None:
        return "uncertain", False
    match = matcher.match(name, slug, get(person, "character"), get(person, "character_aliases", []))
    if match.is_correct:
        return "correct", False
    if match.kind in {"actor", "actor_of_other"}:
        return "actor_name", False
    return "wrong_name", match.needs_review


def score_sample(sample: Any, experiment: str, parsed: ParsedMultiResponse,
                 matcher: CharacterMatcher, *, api_state: str = "ok", iou_threshold: float = .5,
                 ignore_overlap_threshold: float = .5) -> dict[str, Any]:
    experiment = experiment.upper()
    persons = get(sample, "persons", [])
    counts = Counter(identity_category(person) for person in persons)
    row = {
        "sample_id": get(sample, "sample_id"), "work_slug": get(sample, "work_slug"),
        "experiment": experiment, "person_count": len(persons), "verified_count": counts["verified"],
        "unknown_count": counts["unknown"], "pending_count": counts["pending"],
        "api_state": api_state, "parse_ok": parsed.parse_ok if api_state == "ok" else False,
        "format_error": api_state == "ok" and not parsed.refused and bool(parsed.issues or not parsed.parse_ok),
        "refused": api_state == "ok" and parsed.refused, "issues": parsed.issues,
        "warnings": parsed.warnings, "image_correct": None, "person_details": [],
        "identity_correct": 0, "matched_verified": 0, "unknown_null": 0,
        "unknown_guess": 0, "unknown_missing": 0, "extra_ids": len(parsed.extra_ids),
        "localization_eligible": bool(get(sample, "foreground_complete", False)),
        "localization_tp": 0, "localization_fp": 0, "localization_fn": 0,
        "e2e_tp": 0, "e2e_fp": 0, "e2e_fn": 0, "ignored_predictions": 0,
        "unassessed_predictions": 0, "identity_errors": 0,
    }
    usable = api_state == "ok" and parsed.parse_ok and not parsed.refused
    base_error = api_state if api_state != "ok" else "refused" if parsed.refused else "parse_error"
    detail = []

    def person_result(person, predicted, present=True, prediction_index=None, iou=None):
        category = identity_category(person)
        needs_review = False
        if category == "pending":
            outcome = "pending_unscored"
        elif not usable:
            outcome = base_error
        elif not present:
            outcome = "missing_id" if experiment in {"A", "B"} else "missed_detection"
        elif category == "unknown":
            outcome = "unknown_null" if predicted is None else "unknown_guess"
        else:
            outcome, needs_review = classify(predicted, person, row["work_slug"], matcher)
        record = {
            "person_id": get(person, "person_id"), "identity_category": category,
            "annotation_id": get(person, "annotation_id"),
            "gt_character": get(person, "character") if category == "verified" else None,
            "predicted": predicted, "outcome": outcome, "needs_review": needs_review,
            "prediction_index": prediction_index, "iou": iou,
        }
        detail.append(record)
        if category == "verified":
            row["identity_correct"] += outcome == "correct"
            if present and usable and experiment == "C":
                row["matched_verified"] += 1
                row["identity_errors"] += outcome != "correct"
        if category == "unknown":
            row["unknown_null"] += outcome == "unknown_null"
            row["unknown_guess"] += outcome == "unknown_guess"
            row["unknown_missing"] += outcome not in {"unknown_null", "unknown_guess"}
        return outcome

    if experiment in {"A", "B"}:
        predictions = {p.person_id: p.character for p in parsed.persons} if usable else {}
        for person in persons:
            pid = get(person, "person_id")
            person_result(person, predictions.get(pid), pid in predictions)
        for pid in parsed.extra_ids:
            detail.append({"person_id": pid, "identity_category": "extra", "gt_character": None,
                           "predicted": predictions.get(pid), "outcome": "extra_id", "needs_review": False})
        if not counts["pending"] and persons:
            row["image_correct"] = usable and not parsed.issues and all(
                p["outcome"] in {"correct", "unknown_null"} for p in detail)
    elif experiment == "C":
        predictions = parsed.persons if usable else []
        matches = maximum_iou_matching([get(p, "bbox") for p in persons], [p.bbox for p in predictions], iou_threshold)
        by_gt = {g: (pred, iou) for g, pred, iou in matches}
        used = {pred for _g, pred, _iou in matches}
        ignored = {i for i, prediction in enumerate(predictions) if i not in used and overlaps_ignore(
            prediction.bbox, get(sample, "ignore_regions", []), ignore_overlap_threshold)}
        unmatched = len(predictions) - len(used) - len(ignored)
        invalid_predictions = parsed.raw_person_count if api_state == "ok" and not parsed.parse_ok else 0
        row.update(localization_tp=len(matches), localization_fn=len(persons) - len(matches),
                   localization_fp=unmatched + invalid_predictions,
                   ignored_predictions=len(ignored))
        if not row["localization_eligible"]:
            row["unassessed_predictions"] = row["localization_fp"]
        matched_unscored = 0
        for i, person in enumerate(persons):
            pred_index, iou = by_gt.get(i, (None, None))
            prediction = predictions[pred_index].character if pred_index is not None else None
            person_result(person, prediction, pred_index is not None, pred_index, iou)
            if pred_index is not None and identity_category(person) != "verified":
                matched_unscored += 1
        row["e2e_tp"] = row["identity_correct"]
        row["e2e_fn"] = counts["verified"] - row["identity_correct"]
        row["e2e_fp"] = len(predictions) - len(ignored) - matched_unscored + invalid_predictions - row["identity_correct"]
        for index, prediction in enumerate(predictions):
            if index not in used:
                detail.append({"person_id": None, "identity_category": "extra", "gt_character": None,
                               "predicted": prediction.character, "prediction_index": index,
                               "outcome": "ignored_prediction" if index in ignored else "false_positive" if row["localization_eligible"] else "unassessed_prediction",
                               "bbox": prediction.bbox, "needs_review": False})
        if row["localization_eligible"] and not counts["pending"]:
            row["image_correct"] = (usable and row["localization_fn"] == row["localization_fp"] == 0
                                     and all(p["outcome"] in {"correct", "unknown_null", "ignored_prediction"} for p in detail))
    else:
        raise ValueError("experiment must be A, B, or C")
    row["person_details"] = detail
    return row


def fraction(k: int, n: int) -> dict[str, Any]:
    return {"value": k / n if n else None, "numerator": k, "denominator": n,
            "ci95": list(wilson_ci(k, n)) if n else None}


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    return {"tp": tp, "fp": fp, "fn": fn, "precision": fraction(tp, tp + fp),
            "recall": fraction(tp, tp + fn), "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "f1_ci95": None, "f1_ci_note": "F1 is not a binomial proportion; Wilson intervals apply to P/R only."}


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def estimated_cost(usage: dict[str, Any], pricing: dict[str, Any]) -> tuple[float | None, str | None, str | None]:
    """Configured token estimates are explicitly distinct from a full bill."""
    def finite_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    supplied = usage.get("estimated_cost", usage.get("cost_usd", usage.get("cost")))
    if finite_number(supplied):
        return supplied, usage.get("cost_currency", "USD" if "cost_usd" in usage else None), "provider_supplied"
    values = [usage.get("prompt_tokens"), usage.get("completion_tokens"),
              pricing.get("input_cny_per_million"), pricing.get("output_cny_per_million")]
    if all(finite_number(value) for value in values):
        inp, out, input_price, output_price = values
        return (inp * input_price + out * output_price) / 1_000_000, "CNY", "configured_token_price_estimate_not_full_bill"
    return None, None, None


def aggregate(rows: list[dict[str, Any]], experiment: str) -> dict[str, Any]:
    planned, valid = len(rows), [r for r in rows if r["api_state"] == "ok"]
    verified = sum(r["verified_count"] for r in rows)
    correct = sum(r["identity_correct"] for r in rows)
    images = [r for r in rows if r["image_correct"] is not None]
    latency = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float)) and math.isfinite(r["latency_ms"]) and r["latency_ms"] >= 0]
    usage, costs, usage_records, cost_records = Counter(), Counter(), 0, 0
    for row in rows:
        tokens = row.get("usage") or {}
        if tokens:
            usage_records += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if isinstance(tokens.get(key), (int, float)) and not isinstance(tokens[key], bool) and tokens[key] >= 0:
                usage[key] += tokens[key]
        cost = row.get("estimated_cost")
        if isinstance(cost, (int, float)) and not isinstance(cost, bool) and math.isfinite(cost) and cost >= 0:
            costs[row.get("cost_currency") or "unspecified"] += cost
            cost_records += 1
    outcomes = Counter(p["outcome"] for r in rows for p in r["person_details"])
    result = {
        "task": "specified_person_identification" if experiment in {"A", "B"} else "discovery_localization_identification",
        "experiment": experiment, "planned_images": planned, "valid_responses": len(valid),
        "formatted_responses": sum(r["parse_ok"] for r in valid), "person_targets": sum(r["person_count"] for r in rows),
        "verified_targets": verified, "unknown_targets": sum(r["unknown_count"] for r in rows),
        "pending_identity_targets": sum(r["pending_count"] for r in rows),
        "api_success_rate": fraction(len(valid), planned),
        "api_failure_rate": fraction(sum(r["api_state"] == "api_failure" for r in rows), planned),
        "not_run": sum(r["api_state"] == "not_run" for r in rows),
        "format_error_rate": fraction(sum(r["format_error"] for r in rows), len(valid)),
        "refusal_rate": fraction(sum(r["refused"] for r in rows), len(valid)),
        "uncertain_rate": fraction(outcomes["uncertain"], sum(r["verified_count"] for r in valid)),
        "image_exact_accuracy": fraction(sum(bool(r["image_correct"]) for r in images), len(images)),
        "unknown_null": sum(r["unknown_null"] for r in rows), "unknown_wrong_guess": sum(r["unknown_guess"] for r in rows),
        "unknown_missing_or_error": sum(r["unknown_missing"] for r in rows),
        "person_outcomes": dict(outcomes), "error_types": dict(Counter(r.get("error_type") or "unknown" for r in rows if r["api_state"] == "api_failure")),
        "latency_ms_p50": percentile(latency, .5), "latency_ms_p95": percentile(latency, .95),
        "latency_records": len(latency), "usage": dict(usage), "usage_records": usage_records,
        "estimated_cost_by_currency": dict(costs) if cost_records else None, "cost_records": cost_records,
        "cost_methods": dict(Counter(r.get("cost_method") for r in rows if r.get("cost_method"))),
    }
    if experiment in {"A", "B"}:
        result.update(person_accuracy=fraction(correct, sum(r["verified_count"] for r in valid)),
                      person_accuracy_e2e=fraction(correct, verified))
        confusion = Counter((p["gt_character"], p["predicted"] or f"<{p['outcome']}>")
                            for r in rows for p in r["person_details"] if p["identity_category"] == "verified")
        result["confusion"] = [{"gt_character": gt, "predicted": pred, "count": count} for (gt, pred), count in sorted(confusion.items())]
    else:
        complete = [r for r in rows if r["localization_eligible"]]
        result.update(
            localization=prf(*(sum(r[f"localization_{suffix}"] for r in complete) for suffix in ("tp", "fp", "fn"))),
            end_to_end=prf(*(sum(r[f"e2e_{suffix}"] for r in complete) for suffix in ("tp", "fp", "fn"))),
            matched_identity_accuracy=fraction(correct, sum(r["matched_verified"] for r in rows)),
            localization_eligible_images=len(complete), incomplete_foreground_images=planned - len(complete),
            diagnostic_localization_counts={suffix: sum(r[f"localization_{suffix}"] for r in rows) for suffix in ("tp", "fp", "fn")},
            unassessed_predictions=sum(r["unassessed_predictions"] for r in rows),
            ignored_predictions=sum(r["ignored_predictions"] for r in rows),
            identity_errors=sum(r["identity_errors"] for r in complete),
            diagnostic_identity_errors=sum(r["identity_errors"] for r in rows),
        )
    return result


def aggregate_person_stratum(entries: list[tuple[dict[str, Any], dict[str, Any]]],
                             experiment: str) -> dict[str, Any]:
    """GT-person strata: no image exactness or invented pose for false positives."""
    verified = [(r, p) for r, p in entries if p["identity_category"] == "verified"]
    correct = sum(p["outcome"] == "correct" for _r, p in verified)
    result = {
        "experiment": experiment, "unit": "gt_person",
        "sample_count": len({r["sample_id"] for r, _p in entries}),
        "person_targets": len(entries), "verified_targets": len(verified),
        "unknown_targets": sum(p["identity_category"] == "unknown" for _r, p in entries),
        "pending_identity_targets": sum(p["identity_category"] == "pending" for _r, p in entries),
        "http_success_persons": sum(r["api_state"] == "ok" for r, _p in entries),
        "api_failure_persons": sum(r["api_state"] == "api_failure" for r, _p in entries),
        "not_run_persons": sum(r["api_state"] == "not_run" for r, _p in entries),
        "format_error_persons": sum(r["format_error"] for r, _p in entries),
        "refused_persons": sum(r["refused"] for r, _p in entries),
        "person_outcomes": dict(Counter(p["outcome"] for _r, p in entries)),
    }
    if experiment in {"A", "B"}:
        result.update(person_accuracy=fraction(correct, sum(r["api_state"] == "ok" for r, _p in verified)),
                      person_accuracy_e2e=fraction(correct, len(verified)))
    else:
        eligible = [(r, p) for r, p in entries if r["localization_eligible"]]
        eligible_verified = [(r, p) for r, p in eligible if p["identity_category"] == "verified"]
        matched = lambda r, p: (r["api_state"] == "ok" and r["parse_ok"] and not r["refused"]
                                 and p.get("prediction_index") is not None)
        eligible_correct = sum(p["outcome"] == "correct" for _r, p in eligible_verified)
        result.update(
            localization_eligible_persons=len(eligible),
            localization_excluded_persons=len(entries) - len(eligible),
            localization_recall=fraction(sum(matched(r, p) for r, p in eligible), len(eligible)),
            matched_identity_accuracy=fraction(eligible_correct, sum(matched(r, p) for r, p in eligible_verified)),
            end_to_end_recall=fraction(eligible_correct, len(eligible_verified)),
        )
    return result


@dataclass
class MultiScoreResult:
    run_id: str
    dataset_hash: str
    records: list[dict[str, Any]]
    metrics: dict[str, dict[str, Any]]
    by_work: dict[str, Any]
    by_person_count: dict[str, Any]
    by_single_multi: dict[str, Any]
    ab_effects: dict[str, Any]
    is_mock: bool
    quality: dict[str, Any]
    policy: dict[str, Any]
    by_face_orientation: dict[str, Any] = field(default_factory=dict)
    by_visibility: dict[str, Any] = field(default_factory=dict)
    stratification: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {"benchmark_protocol": "multi-v2", "prompt_version": PROMPT_VERSION, **self.__dict__}


def score_run(store, run_id: str, samples: list[Any], matcher: CharacterMatcher,
              settings: dict[str, Any], *, stratification: dict[str, Any] | None = None) -> MultiScoreResult:
    run = store.get_run(run_id)
    if not run:
        raise ValueError(f"Unknown multi run: {run_id}")
    meta = run.get("fairness_meta") or {}
    if meta.get("benchmark_protocol") != "multi-v2":
        raise ValueError("Single-target/v1 responses cannot be scored as multi-v2")
    from .schema import dataset_hash
    digest = dataset_hash(samples)
    if run["dataset_hash"] != digest:
        raise ValueError("Multi-v2 scoring snapshot hash does not match the run")
    tags_payload = load_frozen_stratification(run, samples)
    if stratification is not None:
        if tags_payload is not None:
            raise ValueError("Frozen run stratification cannot be overridden")
        tags_payload = validate_stratification(stratification, samples)
    tags = {(r["sample_id"], r["person_id"]): r for r in (tags_payload or {}).get("persons", [])}
    tags_meta = {"status": "frozen" if meta.get("stratification") else "explicit_posthoc" if tags_payload is not None else "not_provided",
                 "person_labels": len(tags), "model_inputs_changed": False}
    if tags_payload is not None:
        tags_meta["dataset_hash"] = digest
        tags_meta["content_hash"] = stratification_hash(tags_payload)
        tags_meta["frozen_file_sha256"] = (meta.get("stratification") or {}).get("sha256")
    options = settings.get("multi", {}).get("scoring", settings.get("multi_scoring", {}))
    iou = float(options.get("iou_threshold", .5))
    ignore = float(options.get("ignore_overlap_threshold", .5))
    if not 0 < iou <= 1 or not 0 < ignore <= 1:
        raise ValueError("IoU/ignore thresholds must be in (0,1]")
    samples_by_id = {get(s, "sample_id"): s for s in samples}
    if len(samples_by_id) != len(samples):
        raise ValueError("Duplicate multi-v2 sample IDs")
    source_records = store.records(run_id)
    records_by_key = {}
    for record in source_records:
        key = record["model_id"], record["experiment"], record["sample_id"]
        if key in records_by_key or record["sample_id"] not in samples_by_id or record["model_id"] not in run["models"] or record["experiment"] not in run["experiments"]:
            raise ValueError("Response outside multi-v2 run scope or duplicate response")
        if record["work_slug"] != get(samples_by_id[record["sample_id"]], "work_slug"):
            raise ValueError("Response work does not match multi-v2 snapshot")
        records_by_key[key] = record
    rows = []
    for model in run["models"]:
        for exp in run["experiments"]:
            for sample in samples:
                source = records_by_key.get((model, exp, get(sample, "sample_id")))
                state = "not_run" if source is None else "ok" if source["ok"] else "api_failure"
                raw = (source or {}).get("raw") or {}
                response = raw.get("response") or {}
                parsed = parse_response(response.get("raw_text", ""), exp,
                                        [get(p, "person_id") for p in get(sample, "persons", [])],
                                        settings.get("scoring", {}).get("refusal_keywords"))
                row = score_sample(sample, exp, parsed, matcher, api_state=state,
                                   iou_threshold=iou, ignore_overlap_threshold=ignore)
                for person in row["person_details"]:
                    if person["identity_category"] != "extra" and tags:
                        tag = tags[(row["sample_id"], person["person_id"])]
                        person.update(face_orientation=tag["face_orientation"], visibility=tag["visibility"])
                usage = (source or {}).get("usage") or response.get("usage") or {}
                if isinstance(usage, str):
                    usage = json.loads(usage)
                price = ((meta.get("pricing") or {}).get("models") or {}).get(model) or {}
                amount, currency, method = estimated_cost(usage, price)
                row.update(run_id=run_id, model_id=model, latency_ms=(source or {}).get("latency_ms"),
                           usage=usage, error_type=(source or {}).get("error_type"),
                           estimated_cost=amount, cost_currency=currency, cost_method=method,
                           cost_source_url=price.get("source_url") if method and method.startswith("configured_") else None)
                rows.append(row)
    cells = defaultdict(list)
    for row in rows:
        cells[f"{row['model_id']}|{row['experiment']}"].append(row)
    metrics = {key: aggregate(items, key.rsplit("|", 1)[1]) for key, items in cells.items()}

    def grouped(group):
        result = {}
        for key, items in cells.items():
            groups = defaultdict(list)
            for row in items:
                groups[group(row)].append(row)
            result[key] = {name: aggregate(rs, key.rsplit("|", 1)[1]) for name, rs in groups.items()}
        return result

    def person_groups(axis):
        if not tags:
            return {}
        result = {}
        for key, items in cells.items():
            groups = defaultdict(list)
            for row in items:
                for person in row["person_details"]:
                    if person["identity_category"] != "extra":
                        groups[person[axis]].append((row, person))
            result[key] = {name: aggregate_person_stratum(entries, key.rsplit("|", 1)[1])
                           for name, entries in groups.items()}
        return result
    effects = {}
    for model in run["models"]:
        a, b = metrics.get(f"{model}|A"), metrics.get(f"{model}|B")
        if not a or not b:
            continue
        av, bv = a["person_accuracy"]["value"], b["person_accuracy"]["value"]
        a_rows = {(r["sample_id"], p["person_id"]): p["outcome"] == "correct"
                  for r in cells[f"{model}|A"] for p in r["person_details"] if p["identity_category"] == "verified"}
        b_rows = {(r["sample_id"], p["person_id"]): p["outcome"] == "correct"
                  for r in cells[f"{model}|B"] for p in r["person_details"] if p["identity_category"] == "verified"}
        keys = sorted(a_rows.keys() & b_rows.keys())
        effects[model] = {"person_accuracy_delta_B_minus_A": bv - av if av is not None and bv is not None else None,
                          "paired_verified_targets": len(keys), "mcnemar_e2e": mcnemar([a_rows[k] for k in keys], [b_rows[k] for k in keys])}
    quality_counts = Counter(identity_category(p) for s in samples for p in get(s, "persons", []))
    return MultiScoreResult(
        run_id, digest, rows, metrics, grouped(lambda r: r["work_slug"]),
        grouped(lambda r: str(r["person_count"]) if r["person_count"] <= 3 else "4+"),
        grouped(lambda r: "single" if r["person_count"] == 1 else "multi" if r["person_count"] > 1 else "zero"),
        effects, bool(meta.get("is_mock", False)),
        {"samples": len(samples), "persons": sum(quality_counts.values()), "identity_counts": dict(quality_counts),
         "foreground_complete_samples": sum(bool(get(s, "foreground_complete", False)) for s in samples)},
        {"iou_threshold": iou, "ignore_overlap_threshold": ignore,
         "identity_truth": "verified identity AND approved person review; pending excluded from name truth",
         "unknown_identity": "null/guess/missing tracked separately; never an ordinary character-name truth",
         "c_primary_scope": "foreground_complete images only; incomplete-image localization reported as diagnostics",
         "c_matching": "maximum cardinality then maximum IoU; geometry only; one-to-one",
         "ignore": "foreground matched first; unmatched predictions covered by ignore region over prediction area are suppressed",
         "format": "duplicate IDs/positions, invalid bbox/schema/candidate lists invalidate response; missing/extra A/B IDs score explicitly and prevent image exactness",
         "intervals": "Wilson 95% for binomial ratios; person/image correlation not adjusted; no binomial CI for F1",
         "latency": "recorded HTTP latency; excludes scheduler pacing/backoff where provider does so",
         "cost": "explicit returned cost or configured input/output CNY-per-million prices times returned usage; token estimate is not a complete bill; missing usage/prices stay null",
         "pricing_sources": {model: (info or {}).get("source_url") for model, info in ((meta.get("pricing") or {}).get("models") or {}).items()},
         "person_strata": "GT-person tags bound by dataset hash, image SHA, annotation_id and person_id; no pose/visibility assigned to extra predictions, no per-pose precision or image exactness; C recall includes failed/unexecuted eligible targets; sample counts may overlap across groups"},
        person_groups("face_orientation"), person_groups("visibility"), tags_meta,
    )
