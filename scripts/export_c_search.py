"""Export an offline C-only, seven-model search comparison from saved evidence.

This command never calls model APIs or updates the experiment database. It can
be rerun while a test is running; unexecuted tasks retain their planned targets.
"""
from __future__ import annotations

import argparse
import base64
import binascii
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.export_qwenmax_review import CSS, POSE, VISIBILITY, bbox_issue, esc, overlay, table
from scripts.export_qwen_search_pilot import search_evidence as legacy_search_evidence
from vcrb.annotation.rosters import load_roster
from vcrb.config import Work
from vcrb.evaluation.matcher import CharacterMatcher
from vcrb.multi.identity_review import CORRECTIONS_PATH, apply_corrections, score_name_only
from vcrb.multi.parsing import parse_response
from vcrb.multi.schema import MultiSample, dataset_hash, load_dataset
from vcrb.multi.scoring import identity_category, score_sample
from vcrb.multi.stratification import load_frozen_stratification, load_stratification

BASELINE = "multi_v2_20261113_075729_2e4f15"
MODEL_IDS = ("deepseek-flash", "kimi-k3.1-pro", "glm-5.3-flash", "glm-4.6v",
             "qwen3.8-max", "qwen3.7-plus", "qwen3.8-flash")
FAILURES = {"account_arrearage": "厂商返回Arrearage账户拒绝", "content_filter": "厂商内容过滤拦截",
            "other_api_failure": "其他 / 未确认原因的接口失败"}
MODEL_LABELS = {"deepseek-flash": "DeepSeek Flash", "qwen3.8-max": "Qwen Max", "kimi-k3.1-pro": "Kimi K3.1 Pro",
                "glm-5.3-flash": "GLM 5.3 Flash", "glm-4.6v": "GLM 4.6V",
                "qwen3.7-plus": "Qwen Plus", "qwen3.8-flash": "Qwen Flash"}
REVIEW_MODELS = ("deepseek-flash", "qwen3.8-max")
OVERRIDES_MANIFEST = Path("results/multi/c_search/model_run_overrides.json")
PROTOCOL_MARKER = "x-anthropic-billing-header: cc_entrypoint=cli;"


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def parse_model_runs(values):
    result = {}
    for value in values or []:
        if not isinstance(value, str) or "=" not in value:
            raise ValueError("--model-run requires MODEL=RUNID")
        model, rid = value.split("=", 1)
        if model not in MODEL_IDS or not rid.strip() or model in result:
            raise ValueError("Unknown model, empty run ID, or duplicate --model-run")
        result[model] = rid.strip()
    return result


def resolve_model_runs(root, primary_id, explicit=None):
    path = Path(root) / OVERRIDES_MANIFEST
    manifest = {}
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Model-run manifest must be an object")
    defaults = manifest.get("model_runs", {}) if manifest.get("primary_run_id") == primary_id else {}
    if not isinstance(defaults, dict) or explicit is not None and not isinstance(explicit, dict):
        raise ValueError("Model-run selection must be a model-to-run object")
    effective = {**defaults, **(explicit or {})}
    for model, rid in effective.items():
        if model not in MODEL_IDS or not isinstance(rid, str) or not rid.strip():
            raise ValueError("Unknown model or invalid run ID in model-run selection")
    reason = manifest.get("selection_reason") if manifest.get("primary_run_id") == primary_id else None
    return {model: rid for model, rid in effective.items() if rid != primary_id}, str(reason or "按模型整体选择接口修订后的运行；不按单张正确率挑选或回填结果。")


def business_system(text):
    if not isinstance(text, str):
        raise ValueError("Expected saved business system prompt text")
    return "\n".join(line for line in text.splitlines() if line.strip() != PROTOCOL_MARKER).strip()


def thinking_signature(notes):
    params = notes.get("request_params") or {}
    extra = params.get("extra_body", notes.get("extra_body")) or {}
    keys = ("enable_thinking", "thinking", "reasoning_effort", "thinking_budget", "reasoning_budget", "reasoning")
    return {key: deepcopy(extra[key]) for key in keys if key in extra}


def validate_source_metadata(model, source, metadata, primary_metadata):
    if metadata.get("is_mock") is not False or metadata.get("benchmark_protocol") != "multi-v2":
        raise ValueError("Selected model source must be a real multi-v2 run")
    if model not in json.loads(source["models"]) or "C" not in json.loads(source["experiments"]):
        raise ValueError("Selected run does not contain the same model and C experiment")
    primary_scope = (primary_metadata.get("budget_scope") or {}).get("root_run_id")
    if not primary_scope or (metadata.get("budget_scope") or {}).get("root_run_id") != primary_scope:
        raise ValueError("Model sources must share the inherited cumulative budget scope")
    if metadata.get("prompt_version") != primary_metadata.get("prompt_version"):
        raise ValueError("Model source changed the frozen prompt version")
    old_generation = primary_metadata.get("generation_params") or {}
    new_generation = metadata.get("generation_params") or {}
    if any(new_generation.get(key) != old_generation.get(key) for key in ("temperature", "max_tokens")):
        raise ValueError("Model source changed generation temperature/output budget")
    original = (primary_metadata.get("per_model_notes") or {}).get(model)
    replacement = (metadata.get("per_model_notes") or {}).get(model)
    if not isinstance(original, dict) or not isinstance(replacement, dict):
        raise ValueError("Model source is missing frozen model configuration")
    if thinking_signature(original) != thinking_signature(replacement):
        raise ValueError("Model source changed thinking settings")
    mode = (replacement.get("request_params") or {}).get("api_mode")
    allowed = {"dashscope_anthropic", "dashscope_multimodal_search"} if model.startswith("qwen") else {"moonshot_responses"} if model.startswith("kimi") else {"deepseek_native_search"} if model.startswith("deepseek") else {"zhipu_native_search", "zhipu_function_search"}
    if mode not in allowed:
        raise ValueError("Model source changed original-vendor transport")
    for key in ("passes_temperature", "token_limit_parameter"):
        if replacement.get(key) != original.get(key):
            raise ValueError("Model source changed generation parameter semantics")
    def scoring_snapshot(meta):
        settings = meta.get("settings_snapshot") or {}
        return {"scoring": settings.get("scoring", {}),
                "multi": (settings.get("multi") or {}).get("scoring", settings.get("multi_scoring", {}))}
    old_scoring, new_scoring = scoring_snapshot(primary_metadata), scoring_snapshot(metadata)
    if new_scoring != old_scoring:
        raise ValueError("Model source changed scoring settings")


def validate_business_request(model, request, baseline_request, expected_system, expected_thinking=None):
    if request.get("model") != model or request.get("user") != baseline_request.get("user"):
        raise ValueError("Model source changed actual model or C business/title prompt")
    if business_system(request.get("system")) != business_system(expected_system):
        raise ValueError("Model source changed the business system prompt")
    if any(request.get(key) != baseline_request.get(key) for key in ("temperature", "max_tokens")):
        raise ValueError("Model source changed actual generation inputs")
    if expected_thinking is not None and thinking_signature({"extra_body": request.get("extra_body") or {}}) != expected_thinking:
        raise ValueError("Model source changed actual thinking inputs")


def ledger_totals(rows):
    return {"known_token_estimate_cny": sum(row["known_cny"] for row in rows),
            "uncertain_cny": sum(row["uncertain_cny"] for row in rows),
            "inflight_reserve_cny": sum(row["committed_cny"] for row in rows if row["status"] == "in_flight"),
            "committed_cny": sum(row["committed_cny"] for row in rows), "http_attempt_rows": len(rows)}


def annotation_review_context(root, current):
    """Validate the audit replay internally; present only the current truth."""
    from vcrb.multi.review_revision import REVIEW_REVISION_PATH, replay_review_revision
    path = Path(root) / REVIEW_REVISION_PATH
    if not path.exists():
        return current, {"applied": False}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    reviewed = {sample.sample_id: sample.to_dict() for sample in current}
    base = replay_review_revision(reviewed, manifest, reverse=True)
    base_samples = [MultiSample.from_dict(sample) for sample in base.values()]
    if dataset_hash(base_samples) != manifest["before_dataset_hash"] or dataset_hash(current) != manifest["after_dataset_hash"]:
        raise ValueError("Review revision dataset fingerprints differ")
    if replay_review_revision(base, manifest) != reviewed:
        raise ValueError("Review revision cannot reproduce current confirmed annotations")
    return base_samples, {"applied": True, "manifest_path": str(REVIEW_REVISION_PATH),
                          "revision_id": manifest.get("revision_id"),
                          "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "source_dataset_hash": manifest["before_dataset_hash"], "current_dataset_hash": manifest["after_dataset_hash"],
                          "reviewed_sample_ids": [row["sample_id"] for row in manifest["revisions"]],
                          "reviewed_sample_count": len(manifest["revisions"]),
                          "policy": "Offline rescoring of unchanged saved responses; current annotations only in report; no new model calls."}


def select_model_records(per_run_raw, source_ids):
    """Select a source once per model, with no per-image fallback or ranking."""
    return {(model, sid): value for model, rid in source_ids.items()
            for (saved_model, sid), value in per_run_raw[rid].items() if saved_model == model}


def validate_frozen_source(source, metadata, current, tags, primary_stratification):
    rid = source["run_id"]
    frozen = load_dataset(Path(source["dataset_path"]))
    current_hash = dataset_hash(current)
    if len(frozen) != len(current) or dataset_hash(frozen) != current_hash or source["dataset_hash"] != current_hash:
        raise ValueError(f"Model source does not match full current confirmed truth: {rid}")
    expected_hashes = {sample.sample_id: sample.image_sha256 for sample in current}
    if metadata.get("image_hashes") != expected_hashes:
        raise ValueError(f"Model source image manifest differs: {rid}")
    for sample in frozen:
        image_path = Path(source["dataset_path"]).parent / sample.image_path
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != sample.image_sha256:
            raise ValueError(f"Model source frozen image bytes differ: {rid}/{sample.sample_id}")
    strata = load_frozen_stratification({**source, "fairness_meta": metadata}, frozen)
    if strata is None or strata["persons"] != tags:
        raise ValueError(f"Model source stratification differs from confirmed labels: {rid}")
    if (metadata.get("stratification") or {}).get("sha256") != (primary_stratification or {}).get("sha256"):
        raise ValueError(f"Model sources must share identical frozen stratification: {rid}")
    return strata


def report_request(request):
    """Keep actual prompts/settings, replacing embedded image bytes with hashes.

    The original SQLite/JSONL evidence is never modified. Images are embedded
    exactly once per sample in the HTML, rather than inside every wire request.
    """
    def descriptor(payload, mime):
        try:
            decoded = base64.b64decode(payload, validate=True)
        except (ValueError, binascii.Error):
            return {"report_image_payload_omitted": True, "mime_type": mime,
                    "base64_valid": False, "encoded_text_sha256": hashlib.sha256(payload.encode()).hexdigest()}
        return {"report_image_payload_omitted": True, "mime_type": mime,
                "encoded_image_sha256": hashlib.sha256(decoded).hexdigest(), "byte_length": len(decoded)}

    def walk(value, key="", parent=None):
        if isinstance(value, str):
            if value.lower().startswith("data:image/"):
                header, separator, payload = value.partition(",")
                return descriptor(payload if separator else value, header[5:].split(";")[0])
            parent = parent or {}
            mime = parent.get("media_type", parent.get("mime_type", ""))
            if key in ("data", "base64", "image_base64") and isinstance(mime, str) and mime.startswith("image/"):
                return descriptor(value, mime)
            return value
        if isinstance(value, dict):
            return {name: walk(item, name, value) for name, item in value.items()}
        if isinstance(value, list):
            return [walk(item, key, parent) for item in value]
        return value

    return walk(request)


def safe_source(source):
    if not isinstance(source, dict) or not isinstance(source.get("url"), str):
        return None
    try:
        parsed = urlsplit(source["url"])
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
            return None
    except ValueError:
        return None
    return {"url": source["url"], "title": str(source.get("title") or parsed.hostname), "domain": parsed.hostname}


def evidence(raw):
    """Count sources, not intent counters or links invented in the final answer."""
    if raw is None:
        return {"requested": False, "attempted": None, "retrieved": False,
                "sources": [], "queries": [], "billing": {"units": {}, "known_units": False}}
    response, request = raw.get("response") or {}, raw.get("request") or {}
    metadata = response.get("search_metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    normalized = any(key in metadata for key in ("requested", "retrieved", "sources"))
    if normalized:
        values = metadata.get("sources") if isinstance(metadata.get("sources"), list) else []
        sources = [safe_source(source) for source in values]
        sources = list({source["url"]: source for source in sources if source}.values())
        attempted = metadata.get("attempted") if type(metadata.get("attempted")) is bool else None
        billing = metadata.get("billing") if isinstance(metadata.get("billing"), dict) else {}
        units = {str(key): value for key, value in (billing.get("units") or {}).items()
                 if type(value) in (int, float) and math.isfinite(value) and value >= 0} if isinstance(billing.get("units"), dict) else {}
        query_values = metadata.get("queries") if isinstance(metadata.get("queries"), list) else []
        retrieved = metadata.get("retrieved") is True and bool(sources)
        return {"requested": metadata.get("requested") is True or request.get("search_requested") is True,
                "attempted": True if retrieved else attempted, "retrieved": retrieved,
                "sources": sources, "queries": list(dict.fromkeys(q for q in query_values if isinstance(q, str))),
                "billing": {"units": units, "known_units": billing.get("known_units") is True}}
    legacy = legacy_search_evidence(raw)
    return {"requested": request.get("search_requested") is True or (request.get("extra_body") or {}).get("enable_search") is True,
            "attempted": True if legacy["reported_calls"] or legacy["verified"] else None,
            "retrieved": legacy["verified"], "sources": legacy["sources"], "queries": legacy["queries"],
            "billing": {"units": {}, "known_units": False},
            "reported_search_calls_not_bill": legacy["reported_calls"]}


def failure_category(response):
    """Human diagnosis from explicit errors; never rewrite stored classifications."""
    if response.get("ok"):
        return None
    message = str(response.get("error_message") or "")
    if "Arrearage" in message:
        return "account_arrearage"
    if "不安全或敏感内容" in message or "content_filter" in message.lower() or "datainspectionfailed" in message.lower():
        return "content_filter"
    return "other_api_failure"


def score_record(sample, raw, matcher, tags, settings=None):
    response = (raw or {}).get("response") or {}
    state = "not_run" if raw is None else "ok" if response.get("ok") else "api_failure"
    raw_text = response.get("raw_text") or ""
    settings = settings or {}
    scoring = settings.get("multi", {}).get("scoring", settings.get("multi_scoring", {}))
    parsed = parse_response(raw_text, "C", refusal_keywords=settings.get("scoring", {}).get("refusal_keywords"))
    strict = score_sample(sample, "C", parsed, matcher, api_state=state,
                          iou_threshold=float(scoring.get("iou_threshold", .5)),
                          ignore_overlap_threshold=float(scoring.get("ignore_overlap_threshold", .5)))
    names = score_name_only(sample, "C", raw_text if state == "ok" else "", matcher)
    return {"state": state, "completed": raw is not None, "names": names, "strict": strict,
            "raw_text": raw_text, "search": evidence(raw), "tags": tags,
            "request": report_request((raw or {}).get("request")),
            "error_type": response.get("error_type"), "error_message": response.get("error_message"),
            "failure_category": failure_category(response) if raw is not None else None,
            "output_issue": response.get("output_issue"), "served_model": response.get("served_model"),
            "latency_ms": response.get("latency_ms"), "usage": response.get("usage") or {}}


def aggregate(rows, person_filter=None):
    selected = []
    for row in rows:
        people = [person for person in row["names"]["person_details"]
                  if person_filter is None or person_filter(row["tags"].get(person["person_id"], {}))]
        if people:
            selected.append((row, people))
    quota = person_filter is None
    target_of = lambda row, people: row["names"]["target_count"] if quota else len(people)
    correct_of = lambda row, people: row["names"]["correct"] if quota else sum(person.get("full_correct", person["outcome"] in ("correct", "optional_correct")) for person in people)
    targets = sum(target_of(row, people) for row, people in selected)
    correct = sum(correct_of(row, people) for row, people in selected)
    strict_targets = sum(row["strict"].get("verified_count", len(people)) if quota else len(people) for row, people in selected)
    strict_correct = sum(row["strict"]["identity_correct"] if quota else sum(person["outcome"] == "correct" for person in row["strict"]["person_details"]
                         if person.get("person_id") in {p["person_id"] for p in people}) for row, people in selected)
    complete = sum(row["completed"] for row, _people in selected)
    api_ok = sum(row["state"] == "ok" for row, _people in selected)
    legal = sum(row["state"] == "ok" and row["strict"]["parse_ok"] and not row["strict"]["refused"]
                for row, _people in selected)
    completed_targets = sum(target_of(row, people) for row, people in selected if row["completed"])
    api_ok_targets = sum(target_of(row, people) for row, people in selected if row["state"] == "ok")
    result = {"planned_images": len(selected), "planned_targets": targets,
              "completed_images": complete, "completed_targets": completed_targets,
              "api_ok_images": api_ok, "api_failure_images": sum(row["state"] == "api_failure" for row, _ in selected),
              "api_ok_targets": api_ok_targets, "name_accuracy_api_ok": correct / api_ok_targets if api_ok_targets else None,
              "api_ok_name_miss_targets": api_ok_targets - correct,
              "api_failure_categories": {kind: sum(row.get("failure_category") == kind for row, _ in selected) for kind in FAILURES},
              "pending_images": len(selected) - complete,
              "name_correct": correct, "name_accuracy_planned": correct / targets if targets else None,
              "name_accuracy_completed": correct / completed_targets if completed_targets else None,
              "name_denominator_policy": "required_name_quota" if quota else "full_verified_person_coverage",
              "strict_correct": strict_correct, "strict_planned_targets": strict_targets,
              "strict_accuracy_planned": strict_correct / strict_targets if strict_targets else None,
              "legal_format_images": legal, "format_rate_completed": legal / complete if complete else None,
              "format_rate_planned": legal / len(selected) if selected else None,
              "format_error_images": sum(row["strict"]["format_error"] for row, _ in selected),
              "search_requested_images": sum(row["search"]["requested"] for row, _ in selected),
              "search_attempted_images": sum(row["search"]["attempted"] is True for row, _ in selected),
              "search_attempt_unknown_images": sum(row["completed"] and row["search"]["attempted"] is None for row, _ in selected),
              "search_retrieved_images": sum(row["search"]["retrieved"] for row, _ in selected),
              "search_source_coverage_planned": sum(row["search"]["retrieved"] for row, _ in selected) / len(selected) if selected else None}
    if person_filter is None:
        result["named_predictions"] = sum(row["names"]["named_predictions"] for row, _ in selected)
        result["extra_names"] = sum(row["names"]["extra_names"] for row, _ in selected)
        result["full_name_targets"] = sum(row["names"].get("full_target_count", row["names"]["target_count"]) for row, _ in selected)
        result["full_name_correct"] = sum(row["names"].get("full_correct", row["names"]["correct"]) for row, _ in selected)
        result["name_precision"] = result["full_name_correct"] / result["named_predictions"] if result["named_predictions"] else None
    return result


def name_error_reasons(result):
    if not result["completed"]:
        return []
    if result["state"] != "ok":
        return ["api_failure"]
    names = result["names"]
    return (["missing_or_wrong_name"] if names["correct"] < names["target_count"] else []) + (["extra_name"] if names["extra_names"] > 0 else [])


def coordinate_only_issue(result):
    if not result["completed"] or result["state"] != "ok" or name_error_reasons(result):
        return None
    strict = result["strict"]
    if not strict["parse_ok"] or strict["format_error"]:
        return "主姓名要求已满足；仅格式 / 坐标不合规"
    if strict["identity_correct"] < strict.get("verified_count", result["names"].get("full_target_count", result["names"]["target_count"])):
        return "主姓名要求已满足；实际人物位置与姓名覆盖未通过"
    return None


def select_error_review(summary):
    ranked = [model for model in REVIEW_MODELS if model in summary["models"]]
    counts = {model: 0 for model in ranked}
    selected, auxiliary = [], []
    for case in summary["cases"]:
        errors = {model: name_error_reasons(case["results"][model]["new"]) for model in ranked}
        problems = [model for model, reasons in errors.items() if reasons]
        if problems:
            selected.append({"sample_id": case["sample_id"], "models": problems, "reasons": {model: errors[model] for model in problems}})
            for model in problems:
                counts[model] += 1
        elif all(case["results"][model]["new"]["completed"] and case["results"][model]["new"]["state"] == "ok" for model in ranked):
            coordinates = {model: issue for model in ranked if (issue := coordinate_only_issue(case["results"][model]["new"]))}
            if coordinates:
                auxiliary.append({"sample_id": case["sample_id"], "work_name": case["sample"]["work_name"], "issues": coordinates})
    return {"ranking_metric": f'user-selected DeepSeek/Qwen Max; name_accuracy_planned ({summary["planned_targets_per_model"]} required names per model)', "top_models": ranked,
            "model_error_images": counts, "selected_images": selected, "selected_image_count": len(selected),
            "error_model_panels": sum(counts.values()), "coordinate_only_images_not_embedded": auxiliary,
            "coordinate_only_image_count": len(auxiliary), "embedded_image_count": len(selected),
            "policy": "Only user-selected DeepSeek and Qwen Max name misses/wrong names, extra names or API failures enter image cards. Pending tasks and coordinate-only issues are not name errors. Full statistics and JSON retain all samples/models."}


def build_summary(context):
    samples, models, matcher = context["samples"], context["models"], context["matcher"]
    sources = context.get("model_sources") or {model: {"run_id": context["run"]["run_id"], "status": context["run"]["status"], "selection": "primary_run"} for model in models}
    cases = []
    for sid, sample in samples.items():
        tags = {p["person_id"]: {axis: p[axis] for axis in ("face_orientation", "visibility")}
                for p in context["tags"] if p["sample_id"] == sid}
        results = {}
        for model in models:
            results[model] = {kind: score_record(sample, context[kind].get((model, sid)), matcher, tags,
                                                (context.get(kind + "_settings_by_model") or {}).get(model, context.get(kind + "_settings"))) for kind in ("old", "new")}
            results[model]["new"]["source_run_id"] = sources[model]["run_id"]
        cases.append({"sample_id": sid, "sample": sample, "results": results,
                      "input_sha256": context["images"][sid]["sha256"]})

    def comparison(case_filter=lambda _case: True, person_filter=None):
        selected = [case for case in cases if case_filter(case)]
        return {model: {kind: aggregate([case["results"][model][kind] for case in selected], person_filter)
                        for kind in ("old", "new")} for model in models}

    work_names = {sample["work_slug"]: sample["work_name"] for sample in samples.values()}
    groups = {"works": {slug: comparison(lambda case, slug=slug: case["sample"]["work_slug"] == slug) for slug in work_names},
              "single_multi": {"single": comparison(lambda case: len(case["sample"]["persons"]) == 1),
                               "multi": comparison(lambda case: len(case["sample"]["persons"]) > 1)},
              "orientation": {value: comparison(person_filter=lambda tags, value=value: tags.get("face_orientation") == value) for value in POSE},
              "visibility": {value: comparison(person_filter=lambda tags, value=value: tags.get("visibility") == value) for value in VISIBILITY}}
    completed = sum(case["results"][model]["new"]["completed"] for case in cases for model in models)
    planned = len(samples) * len(models)
    source_states = {source["run_id"]: source["status"] for source in sources.values()}
    complete = completed == planned
    active = any(state in ("running", "pending") for state in source_states.values())
    status = "running" if active else "done" if complete else "limited"
    diagnostics = {}
    for model, diagnostic in (context.get("diagnostics") or {}).items():
        records = []
        for sid, raw in diagnostic["records"].items():
            tags = {p["person_id"]: {axis: p[axis] for axis in ("face_orientation", "visibility")}
                    for p in context["tags"] if p["sample_id"] == sid}
            records.append({"sample_id": sid, "source_run_id": diagnostic["run_id"],
                            "result": score_record(samples[sid], raw, matcher, tags, context.get("primary_settings", context.get("new_settings")))})
        diagnostics[model] = {key: deepcopy(value) for key, value in diagnostic.items() if key != "records"}
        diagnostics[model].update(recorded_images=len(records), original_planned_images=len(samples),
                                  inclusion="Historical whole-model cohort only; excluded from the selected 1120 tasks; costs retained in the shared budget.",
                                  metrics_recorded_subset=aggregate([record["result"] for record in records]), records=records)
    result = {"run_id": context["run"]["run_id"], "baseline_run_id": context["baseline"]["run_id"],
            "generated_at": datetime.now(timezone.utc).isoformat(), "status": status,
            "primary_run_status": context["run"]["status"], "source_run_states": source_states,
            "model_sources": sources, "has_model_overrides": any(source["run_id"] != context["run"]["run_id"] for source in sources.values()),
            "excluded_model_diagnostics": diagnostics,
            "run_note": context["run"].get("note", ""),
            "complete": complete, "finalized": complete and not active,
            "models": list(models), "planned_images_per_model": len(samples),
            "planned_targets_per_model": sum(case["results"][models[0]]["new"]["names"]["target_count"] for case in cases),
            "strict_targets_per_model": sum(case["results"][models[0]]["new"]["strict"]["verified_count"] for case in cases),
            "planned_tasks": planned, "completed_tasks": completed, "pending_tasks": planned - completed,
            "dataset_hash": context["dataset_hash"], "baseline_dataset_hash": context["baseline"]["dataset_hash"],
            "scoring_policy": "Current confirmed GT; main names use required quotas and optional identities, independently of coordinates; strict matching uses all annotated verified people and IoU>=0.5. Pose/visibility use full individual name coverage. Pending tasks remain in planned denominators.",
            "all_works": comparison(), "without_gongfunvzu": comparison(lambda case: case["sample"]["work_slug"] != "gongfunvzu"),
            "groups": groups, "work_names": work_names, "costs": context.get("costs", {}),
            "configuration": context.get("metadata", {}), "annotation_review": context.get("annotation_review", {"applied": False}),
            "operational_control": context.get("operational_control", {}), "cases": cases}
    result["review_selection"] = select_error_review(result)
    return result


def scoring_truth(samples):
    """Ignore explanatory prose edits, but reject unlogged scoring changes."""
    sample_keys = ("work_slug", "work_name", "work_category", "image_sha256", "foreground_complete", "review_status")
    person_keys = ("person_id", "annotation_id", "bbox", "character", "character_aliases", "identity_status", "review_status")
    return {sid: {**{key: sample.get(key) for key in sample_keys},
                  "persons": [{key: person.get(key) for key in person_keys} for person in sample["persons"]],
                  "ignore_boxes": [region if isinstance(region, list) else region.get("bbox") for region in sample.get("ignore_regions", [])]}
            for sid, sample in samples.items()}


def read_context(root, run_id, baseline_id=BASELINE, model_runs=None):
    root = Path(root).resolve()
    overrides, selection_reason = resolve_model_runs(root, run_id, model_runs)
    current = load_dataset(root / "data/v2/dataset.json")
    samples = {sample.sample_id: sample.to_dict() for sample in current}
    source_truth, review_info = annotation_review_context(root, current)
    tags = load_stratification(root / "data/v2/stratification.json", current)["persons"]
    if len(samples) != 160:
        raise ValueError("Expected the approved 160-image dataset")
    with sqlite3.connect((root / "results/multi/runs.db").as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        runs = {}
        for label, rid in (("run", run_id), ("baseline", baseline_id)):
            value = conn.execute("SELECT * FROM runs WHERE run_id=?", (rid,)).fetchone()
            if value is None:
                raise ValueError(f"Run does not exist: {rid}")
            runs[label] = dict(value)
        metadata = json.loads(runs["run"]["fairness_meta"])
        old_metadata = json.loads(runs["baseline"]["fairness_meta"])
        models = json.loads(runs["run"]["models"])
        if len(models) != len(MODEL_IDS) or set(models) != set(MODEL_IDS) or json.loads(runs["run"]["experiments"]) != ["C"] or metadata.get("is_mock"):
            raise ValueError("Expected a real, seven-model, C-only run")
        if not set(models) <= set(json.loads(runs["baseline"]["models"])) or old_metadata.get("is_mock"):
            raise ValueError("Baseline is missing required real models")
        source_ids = {model: overrides.get(model, run_id) for model in MODEL_IDS}
        source_runs, source_meta = {run_id: runs["run"]}, {run_id: metadata}
        for rid in set(source_ids.values()) - {run_id}:
            value = conn.execute("SELECT * FROM runs WHERE run_id=?", (rid,)).fetchone()
            if value is None:
                raise ValueError(f"Selected model run does not exist: {rid}")
            source_runs[rid] = dict(value)
            source_meta[rid] = json.loads(value["fairness_meta"])
        for model, rid in source_ids.items():
            validate_source_metadata(model, source_runs[rid], source_meta[rid], metadata)
        per_run_raw = {rid: {(row["model_id"], row["sample_id"]): json.loads(row["raw_json"]) for row in conn.execute(
            "SELECT model_id,sample_id,raw_json FROM records WHERE run_id=? AND experiment='C'", (rid,))}
                       for rid in source_runs}
        raw = {"old": {(row["model_id"], row["sample_id"]): json.loads(row["raw_json"]) for row in conn.execute(
            "SELECT model_id,sample_id,raw_json FROM records WHERE run_id=? AND experiment='C'", (baseline_id,))},
               "new": select_model_records(per_run_raw, source_ids)}
        has_ledger = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='multi_budget_calls'").fetchone()
        scope = metadata.get("budget_scope") or {}
        scope_root = scope.get("root_run_id", run_id)
        scope_ids = [row["run_id"] for row in conn.execute("SELECT run_id,fairness_meta FROM runs")
                     if row["run_id"] == scope_root or (json.loads(row["fairness_meta"]).get("budget_scope") or {}).get("root_run_id") == scope_root]
        scope_ledger = [dict(row) for row in conn.execute("SELECT * FROM multi_budget_calls WHERE run_id IN (" + ",".join("?" for _ in scope_ids) + ")", scope_ids)] if has_ledger and scope_ids else []
    frozen_strata = {}
    source_primary_samples = load_dataset(Path(runs["run"]["dataset_path"]))
    source_primary_strata = load_frozen_stratification({**runs["run"], "fairness_meta": metadata}, source_primary_samples)
    if source_primary_strata is None:
        raise ValueError("Source runs require their original frozen stratification")
    for rid, source in source_runs.items():
        frozen_strata[rid] = validate_frozen_source(source, source_meta[rid], source_truth, source_primary_strata["persons"], metadata.get("stratification"))
    old_frozen = load_dataset(Path(runs["baseline"]["dataset_path"]))
    if dataset_hash(old_frozen) != runs["baseline"]["dataset_hash"]:
        raise ValueError("Baseline frozen dataset hash mismatch")
    corrections_path = root / CORRECTIONS_PATH
    corrections = json.loads(corrections_path.read_text())["corrections"] if corrections_path.exists() else []
    corrected = apply_corrections({s.sample_id: s.to_dict() for s in old_frozen}, corrections)
    source_samples = {sample.sample_id: sample.to_dict() for sample in source_truth}
    if scoring_truth(corrected) != scoring_truth(source_samples):
        raise ValueError("Baseline truth plus explicit corrections differs from current truth")
    if review_info["applied"]:
        from vcrb.multi.review_revision import validate_frozen_source as validate_revision_source
        validate_revision_source(corrected, source_samples)
    rosters = {slug: load_roster(Work(slug, name, "", "", None), root / "config/rosters")
               for slug, name in {sample["work_slug"]: sample["work_name"] for sample in samples.values()}.items()}
    gallery = {row["sample_id"]: row for row in json.loads((root / "results/multi/input_gallery/manifest.json").read_text())["rows"]}
    images = {}
    expected_keys = {(model, sid) for model in models for sid in samples}
    if set(raw["old"]) & expected_keys != expected_keys or set(raw["new"]) - expected_keys:
        raise ValueError("Baseline incomplete or new run contains foreign C tasks")
    expected_system = next((value["request"]["system"] for value in per_run_raw[run_id].values()), None)
    expected_system_hash = metadata.get("active_system_prompt_sha256")
    for sid, sample in samples.items():
        path = root / "results/multi/input_gallery" / gallery[sid]["experiments"]["C"]["image"]
        image_bytes = path.read_bytes()
        sha = hashlib.sha256(image_bytes).hexdigest()
        first = raw["old"][(models[0], sid)]["request"]["image_meta"]
        images[sid] = {"bytes": image_bytes, "sha256": sha, "size": first["encoded_size"]}
        for kind in ("old", "new"):
            for model in models:
                record = raw[kind].get((model, sid))
                if not record:
                    continue
                info = record["request"]["image_meta"]
                if info["encoded_sha256"] != sha or info["source_sha256"] != sample["image_sha256"] or info.get("annotation_mode") != "unmarked":
                    raise ValueError(f"C input image mismatch: {kind}/{model}/{sid}")
                if kind == "new":
                    if expected_system is not None:
                        validate_business_request(model, record["request"], raw["old"][(model, sid)]["request"], expected_system,
                                                  thinking_signature(metadata["per_model_notes"][model]))
                    elif not expected_system_hash or hashlib.sha256(business_system(record["request"].get("system")).encode()).hexdigest() != expected_system_hash:
                        raise ValueError("Cannot verify selected source business system prompt")
                    else:
                        validate_business_request(model, record["request"], raw["old"][(model, sid)]["request"], record["request"]["system"],
                                                  thinking_signature(metadata["per_model_notes"][model]))
    for model, rid in source_ids.items():
        if rid != run_id and not any(saved_model == model for saved_model, _sid in per_run_raw[rid]) and source_meta[rid].get("active_system_prompt_sha256") != expected_system_hash:
            raise ValueError("Empty model source cannot prove the same business prompt")
    try:
        gate = json.loads((root / "data/v2/real_call_gate.json").read_text())
    except (OSError, ValueError):
        gate = {}
    holds = gate.get("provider_holds") if isinstance(gate.get("provider_holds"), dict) else {}
    control = {"read_at": datetime.now(timezone.utc).isoformat(),
               "provider_holds": {provider: {key: hold.get(key) for key in ("state", "kind", "reason", "recorded_at", "source_run_id")}
                                  for provider, hold in holds.items() if isinstance(hold, dict) and hold.get("state") == "held"},
               "balance_confirmation_providers": list((gate.get("provider_balance_confirmations") or {}).keys())}
    combined_meta = {key: deepcopy(metadata.get(key)) for key in ("per_model_notes", "search_policy", "generation_params", "budget_scope", "benchmark_protocol", "pricing", "cost_reservation_policy")}
    model_sources = {}
    for model, rid in source_ids.items():
        own_meta = source_meta[rid]
        for key in ("per_model_notes", "search_policy"):
            combined_meta[key] = combined_meta.get(key) or {}
            combined_meta[key][model] = deepcopy((own_meta.get(key) or {}).get(model))
        if own_meta.get("pricing"):
            combined_meta["pricing"] = combined_meta.get("pricing") or {}
            combined_meta["pricing"].setdefault("models", {})[model] = deepcopy((own_meta["pricing"].get("models") or {}).get(model))
        model_sources[model] = {"run_id": rid, "status": source_runs[rid]["status"],
                                "dataset_hash": source_runs[rid]["dataset_hash"], "stratification_sha256": (own_meta.get("stratification") or {}).get("sha256"),
                                "request_params": deepcopy((own_meta["per_model_notes"][model].get("request_params") or {})),
                                "served_models": sorted({value["response"].get("served_model") for (saved_model, _sid), value in per_run_raw[rid].items()
                                                         if saved_model == model and isinstance(value["response"].get("served_model"), str)}),
                                "selection": "model_run_override" if rid != run_id else "primary_run",
                                "selection_reason": selection_reason if rid != run_id else "沿用原正式run该模型的全部C组记录。"}
    selected_ledger = [row for row in scope_ledger if source_ids.get(row["model_id"]) == row["run_id"]]
    diagnostics = {}
    for model in overrides:
        excluded = {sid: value for (saved_model, sid), value in per_run_raw[run_id].items() if saved_model == model}
        diagnostics[model] = {"run_id": run_id, "replaced_by_run_id": source_ids[model], "records": excluded,
                              "request_params": deepcopy((metadata["per_model_notes"][model].get("request_params") or {})),
                              "costs": ledger_totals([row for row in scope_ledger if row["run_id"] == run_id and row["model_id"] == model])}
    cumulative = {**ledger_totals(scope_ledger), "root_run_id": scope_root, "run_ids": scope_ids,
                  "local_protection_cap_cny": max((own.get("budget_scope") or {}).get("max_cost_cny", 0) for own in source_meta.values())}
    return {**runs, **raw, "samples": samples, "models": list(MODEL_IDS), "matcher": CharacterMatcher(rosters),
            "tags": tags, "images": images, "dataset_hash": dataset_hash(current),
            "annotation_review": review_info,
            "model_sources": model_sources, "diagnostics": diagnostics,
            "operational_control": control,
            "old_settings": old_metadata.get("settings_snapshot", {}), "new_settings": metadata.get("settings_snapshot", {}),
            "new_settings_by_model": {model: source_meta[rid].get("settings_snapshot", {}) for model, rid in source_ids.items()},
            "primary_settings": metadata.get("settings_snapshot", {}),
            "metadata": combined_meta,
            "costs": {**ledger_totals(selected_ledger), "scope_note": "Selected whole-model source runs only; superseded model records remain in cumulative_scope and diagnostics without double counting.",
                      "cumulative_scope": cumulative}}


def rate(numerator, denominator, unit="人"):
    return f'<strong>{numerator / denominator:.2%}</strong><small>{numerator} / {denominator} {unit}</small>' if denominator else "—"


def comparison_table(comparison):
    rows = []
    for model, values in sorted(comparison.items(), key=lambda item: item[1]['new']['name_accuracy_planned']
                                if item[1]['new']['name_accuracy_planned'] is not None else -1, reverse=True):
        old, new = values["old"], values["new"]
        delta = (new["name_accuracy_planned"] - old["name_accuracy_planned"]) * 100 if new["planned_targets"] else None
        rows.append([esc(MODEL_LABELS.get(model, model)), rate(old["name_correct"], old["planned_targets"]),
                     rate(new["name_correct"], new["planned_targets"]),
                     "待全部完成" if new["pending_images"] else "—" if delta is None else f'{delta:+.2f} 个百分点'])
    return table(["模型", "未联网姓名命中率", "联网姓名命中率", "变化"], rows)


def brief_results_table(summary):
    rows = []
    for model, values in sorted(summary['all_works'].items(),
                                key=lambda item: item[1]['new']['name_accuracy_planned']
                                if item[1]['new']['name_accuracy_planned'] is not None else -1, reverse=True):
        new = values['new']
        rows.append([esc(MODEL_LABELS.get(model, model)),
                     rate(new['name_correct'], new['planned_targets']),
                     rate(new['api_ok_images'], new['planned_images'], '张')])
    return table(['模型', '姓名命中率', 'API调用成功率'], rows)


def methods_table(summary):
    labels = {"deepseek_native_search": "DeepSeek原厂原生搜索", "moonshot_responses": "Kimi原厂原生Responses搜索",
              "dashscope_anthropic": "Qwen原厂Anthropic兼容原生搜索工具",
              "dashscope_multimodal_search": "Qwen原厂DashScope多模态流式搜索",
              "zhipu_function_search": "智谱自有搜索API＋模型function工具循环"}
    configuration = summary.get("configuration") or {}
    notes = configuration.get("per_model_notes") or {}
    policies = configuration.get("search_policy") or {}
    rows = []
    for model in summary["models"]:
        params = (notes.get(model) or {}).get("request_params") or {}
        policy = policies.get(model) or {}
        mode = params.get("api_mode") or policy.get("api_mode")
        extra = params.get("extra_body") or {}
        thinking = "关闭" if extra.get("enable_thinking") is False or (extra.get("thinking") or {}).get("type") == "disabled" else "沿用最低力度（不可关闭）" if model in ("kimi-k3.1-pro", "glm-5.3-flash") else "见冻结配置"
        limit = params.get("max_client_search_calls") if mode == "zhipu_function_search" else params.get("search_max_uses") if mode == "dashscope_anthropic" else None
        constraint = f"客户端每图最多{limit}次搜索" if mode == "zhipu_function_search" and limit else f"声明{limit}次，仅为建议，服务端可能超出" if limit else "无已验证的服务端搜索次数硬上限"
        choice = policy.get("tool_choice") or {}
        forced = (extra.get("search_options") or {}).get("forced_search")
        strategy = "AUTO：由模型决定是否检索" if mode == "dashscope_anthropic" and (choice.get("type") == "auto" or forced is False) else "强制指定web_search工具" if mode == "dashscope_anthropic" and (choice.get("type") == "tool" or forced is True) else "max检索策略（多模态流式）" if mode == "dashscope_multimodal_search" else "见实际来源配置与检索证据"
        source = (summary.get("model_sources") or {}).get(model) or {}
        rows.append([esc(model), esc(source.get("run_id", summary["run_id"])), esc(labels.get(mode, mode or "见冻结配置")), esc(strategy), thinking, esc(constraint)])
    return table(["模型", "全部160张的来源run", "本轮检索接入", "检索策略", "思考设置", "检索次数约束"], rows)


def failures_table(summary):
    rows = []
    for model, values in summary["all_works"].items():
        new = values["new"]
        categories = new["api_failure_categories"]
        rows.append([esc(model), str(new["api_ok_images"]), str(new["api_failure_images"]),
                     str(categories["account_arrearage"]), str(categories["content_filter"]), str(categories["other_api_failure"]),
                     rate(new["name_correct"], new["api_ok_targets"]), str(new["pending_images"])])
    return table(["模型", "API成功图片", "API失败图片", "明确Arrearage账户拒绝", "明确内容拦截", "其他 / 原因未确认", "仅API成功子集的姓名命中", "尚未执行"], rows)


def model_panel(model, result, *, compact=False):
    old, new = result["old"], result["new"]
    status = "尚未执行" if not new["completed"] else "接口失败" if new["state"] != "ok" else "已返回"
    parts = [f'<section class="model-panel" data-model="{esc(model)}"><h3>{esc(MODEL_LABELS.get(model, model))} · {status}</h3>']
    if not compact:
        parts.append('<p class="model-source">本模型测试来源：' + esc(new.get('source_run_id')) + '</p>')
    if compact:
        reasons = name_error_reasons(new)
        labels = {"api_failure": "接口失败，无法评估姓名", "missing_or_wrong_name": "角色姓名有漏答或未命中", "extra_name": "多报未命中目标的姓名"}
        parts.append('<p class="error-summary">' + '；'.join(labels[reason] for reason in reasons) + '。姓名命中 ' + str(new["names"]["correct"]) + ' / ' + str(new["names"]["target_count"]) + '；多报姓名 ' + str(new["names"]["extra_names"]) + ' 条。</p>')
    if new["state"] == "api_failure":
        parts.append('<div class="notice"><strong>' + esc(FAILURES[new["failure_category"]]) + '</strong>：接口没有成功返回可评估答案，不能据此判定人物认错。下方保留原始错误类型和原文；没有账单证据时，不将该次费用视为零。</div>')
    old_people = {p["person_id"]: p for p in old["names"]["person_details"]}
    strict_people = {p["person_id"]: p for p in new["strict"]["person_details"] if p.get("person_id")}
    rows = []
    for person in new["names"]["person_details"]:
        pid = person["person_id"]
        tag = new["tags"].get(pid, {})
        old_hit = old_people[pid]["outcome"] == "correct"
        hit = person["outcome"] == "correct"
        strict = strict_people[pid]
        if new["state"] == "api_failure":
            verdict = '<span class="bad">接口失败，未判定姓名</span>'
        elif person.get("optional"):
            verdict = '<span class="good">可选名已命中：' + esc(person["predicted"]) + '</span>' if person.get("full_correct") else '<span class="metric-caption">可选旁人，可不回答</span>'
        elif person.get("scored") is False:
            verdict = '<span class="good">配额外正确名</span>' if person.get("full_correct") else '<span class="metric-caption">不计必答名额</span>'
        else:
            verdict = '<span class="good">命中：' + esc(person["predicted"]) + '</span>' if hit else '<span class="bad">' + ("尚未执行" if not new["completed"] else "未命中") + '</span>'
        if compact:
            rows.append([str(pid), esc(person["gt_character"]), verdict])
        else:
            rows.append([str(pid), esc(person["gt_character"]), "命中" if old_hit else "未命中", verdict,
                     "位置和姓名均正确" if strict["outcome"] == "correct" else "接口失败，未判定位置与姓名" if new["state"] == "api_failure" else esc(strict["outcome"]),
                     esc(POSE.get(tag.get("face_orientation"), "—")) + " / " + esc(VISIBILITY.get(tag.get("visibility"), "—"))])
    parts.append(table(["真值编号", "人工确认角色", "本次姓名判定"] if compact else ["人工真值编号", "当前确认角色", "旧组仅姓名", "新组仅姓名", "新组严格判定", "朝向 / 可见性"], rows))
    predictions = new["names"]["predictions"]
    names = "、".join("null" if p["character"] is None else str(p["character"]) for p in predictions)
    parts.append('<p class="name-list">模型实际姓名列表：<strong>' + esc(names or "（无姓名条目）") + '</strong></p>')
    extra = [prediction["character"] for prediction in predictions if prediction["outcome"] == "extra_name"]
    if extra:
        parts.append('<p class="bad">多报且未命中目标的姓名：' + esc('、'.join(str(name) for name in extra)) + '</p>')
    if compact:
        parts.append('<details class="technical-details"><summary>搜索与格式记录</summary>')
    search = new["search"]
    attempted = "已触发" if search["attempted"] is True else "未触发" if search["attempted"] is False else "无法核实是否触发"
    parts.append('<p>搜索请求：' + ("已配置" if search["requested"] else "未发出 / 未配置") + '；' + attempted + '；有效来源：' + str(len(search["sources"])) + '条。' + ('已核实检索得到内容。' if search["retrieved"] else '未核实检索得到内容。') + '</p>')
    if search["queries"]:
        parts.append('<details class="queries"><summary>搜索查询</summary><pre>' + esc("\n".join(search["queries"])) + '</pre></details>')
    if search["sources"]:
        parts.append('<details class="sources"><summary>搜索来源（' + str(len(search["sources"])) + '条）</summary><ul>' + ''.join('<li><a href="' + esc(source["url"]) + '" target="_blank" rel="noopener noreferrer">' + esc(source["title"]) + '</a> · ' + esc(source["domain"]) + '</li>' for source in search["sources"]) + '</ul></details>')
    if new["completed"]:
        issues = new["strict"]["issues"]
        if issues or new["error_type"] or new["output_issue"]:
            parts.append('<div class="notice">格式 / 接口记录：' + esc(dump({"issues": issues, "error_type": new["error_type"], "error_message": new["error_message"], "output_issue": new["output_issue"]})) + '</div>')
        coordinate_rows = [[str(p["prediction_index"] + 1), esc(p["character"]), esc(dump(p["original_bbox"])), esc(bbox_issue(p["original_bbox"]))] for p in predictions]
        parts.append('<details><summary>原始坐标与格式核对</summary>' + table(["预测序号", "实际姓名", "原始bbox", "坐标说明"], coordinate_rows) + '</details>')
    if compact:
        parts.append('</details>')
    parts.append('<details class="raw-responses"><summary>查看模型原始回答</summary><h4>联网原始最终返回</h4><pre>' + esc(new["raw_text"] or ("尚未执行" if not new["completed"] else "（空正文）")) + '</pre><h4>未联网原始最终返回</h4><pre>' + esc(old["raw_text"]) + '</pre><h4>联网实际请求条件</h4><pre>' + esc(dump(new["request"])) + '</pre><h4>服务端检索计量（非账单）</h4><pre>' + esc(dump(search["billing"])) + '</pre></details></section>')
    return ''.join(parts)


STYLE = """
body{background:#f3f7f8;font-size:14px;line-height:1.55}main{max-width:1460px;margin:26px auto;padding:0 24px}h1{font-size:28px;margin:0 0 5px}h2{font-size:20px}p{margin:8px 0}.brief-header{display:flex;justify-content:space-between;align-items:center;gap:20px;margin-bottom:18px}.brief-header p{color:#607585;font-size:13px;margin:0}.review-status{background:#e7f5ef;color:#087f5b;padding:7px 12px;border-radius:20px;font-size:12px;white-space:nowrap}.panel,.case{padding:18px;margin-bottom:15px}.overview h2{margin-bottom:12px}.overview table{table-layout:fixed}.overview th:first-child{width:36%}.overview td{padding:9px 12px;vertical-align:middle}.overview td strong{font-size:17px}.overview td small{display:inline-block;margin-left:12px;font-size:12px}.overview .metric-caption{margin-bottom:0}.breakdown{padding:4px 18px}.summary-toggle{font-size:14px}.group-table{margin:10px 0}.group-table summary{font-size:13px}.metric-caption{color:#607585;font-size:12px}.status-note{background:#fff4df;padding:10px;border-radius:8px}.error-heading{margin:23px 0 10px}.error-heading h2{display:inline-block;margin:0 15px 0 0}.error-heading>div{display:inline}.issue-chip{display:inline-block;padding:4px 9px;border-radius:15px;background:#fff1e8;color:#8d3a10;font-size:12px;margin:2px}.error-heading p{margin:6px 0}.toolbar{position:static;display:grid;grid-template-columns:1.1fr .8fr 1fr auto;align-items:end;gap:10px;padding:12px;margin:14px 0}.toolbar label{font-size:12px}.toolbar select,.toolbar input{width:100%;max-width:100%;font-size:13px}input[type=search]{min-width:0}#visible-count{grid-column:1/-1;font-size:12px;color:#526b7b}.case{margin-top:15px}.case-head{display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:12px}.case h2{font-size:18px;margin:0}.case-body{display:grid;grid-template-columns:minmax(0,1.08fr) minmax(0,1fr);gap:20px;align-items:start}.case-visual{min-width:0;position:sticky;top:18px;align-self:start}.case-analysis{min-width:0}.image-wrap{position:relative;cursor:zoom-in}.image-wrap>img{display:block;width:100%;height:auto}.review-layer{position:absolute;inset:0;pointer-events:none}.prediction-layer .gt-box{display:none}.image-wrap.hide-review .review-layer{display:none}.model-panel[hidden],.review-layer[hidden],.case[hidden]{display:none}.image-controls{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:9px 0}.image-controls button,.image-controls select{font-size:12px;padding:6px 8px}.image-controls select{max-width:210px}.image-controls label{display:flex;align-items:center;gap:5px;font-size:12px}.image-controls input[type=checkbox]{margin:0}.case-visual .metric-caption{margin:5px 0;line-height:1.4}.prediction-caption{color:#8d5216}.model-panels{display:grid;grid-template-columns:minmax(0,1fr);gap:12px}.model-panel{min-width:0;border:1px solid #d8e4ec;border-radius:9px;padding:12px}.model-panel h3{font-size:16px;margin:0}.model-panel table{font-size:12px}.model-panel th,.model-panel td{padding:6px 8px}.model-panel th{white-space:normal}.model-panel .good{color:#087f5b;font-weight:700}.model-panel .bad{color:#ac281e;font-weight:700}.model-panel p{font-size:13px}.error-summary{font-size:12px!important;background:#fff1e8;padding:8px;border-radius:6px;color:#883b0c}.name-list{background:#f0f6fa;padding:8px;border-radius:6px}.model-panel details{margin-top:5px}.model-panel summary{font-size:12px;padding:5px 0}.model-panel pre{max-height:260px;font-size:12px;padding:10px}.model-source{font-size:11px;color:#607788;overflow-wrap:anywhere}.notice{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;padding:9px}.source-list,li{overflow-wrap:anywhere}#zoom-content{max-height:80vh;overflow:auto}#zoom-content .image-wrap{cursor:default}footer{margin:22px 0 35px;font-size:12px}
@media(max-width:820px){main{padding:0 12px;margin-top:16px}.brief-header{display:block}.review-status{display:inline-block;margin-top:10px}h1{font-size:24px}.case-body{grid-template-columns:minmax(0,1fr)}.case-visual{position:static}.panel,.case{padding:13px}.toolbar{grid-template-columns:1fr 1fr}.toolbar button{font-size:12px}.overview th:first-child{width:34%}.overview th,.overview td{padding:8px}.overview td strong{font-size:15px}.overview td small{display:block;margin-left:0;font-size:10px}.case-head{align-items:start;flex-wrap:wrap}.case h2{font-size:16px}.error-heading h2{font-size:18px}}
@media print{body{background:white;font-size:11px}main{max-width:none;margin:0;padding:0}.toolbar,.image-controls{display:none}.case{break-inside:avoid;padding:12px}.case-visual{position:static}.case-body{grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:15px}.brief-header{display:flex}.model-panel{padding:8px}.model-panel p,.metric-caption{font-size:10px}.model-panel table{font-size:10px}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
"""
JS = """
const model=document.getElementById('model'),work=document.getElementById('work'),query=document.getElementById('search'),cards=[...document.querySelectorAll('.case')];
function update(){let count=0,panelsShown=0;for(const card of cards){const panels=[...card.querySelectorAll('.model-panel')];for(const panel of panels)panel.hidden=model.value!=='both'&&panel.dataset.model!==model.value;const active=panels.filter(p=>!p.hidden).map(p=>p.dataset.model);const show=active.length>0&&(!work.value||card.dataset.work===work.value)&&(!query.value.trim()||card.dataset.search.includes(query.value.trim().toLowerCase()));card.hidden=!show;count+=Number(show);if(show)panelsShown+=active.length;const select=card.querySelector('.prediction-model');for(const option of select.options)option.disabled=!!option.value&&!active.includes(option.value);if(select.value&&!active.includes(select.value))select.value='';for(const layer of card.querySelectorAll('.prediction-layer'))layer.hidden=!select.value||layer.dataset.model!==select.value;card.querySelector('.prediction-caption').textContent=select.value?`橙框：${select.options[select.selectedIndex].text}（仅绘制合法坐标）`:'默认只显示人工标注；可选择一款模型叠加预测框。';}document.getElementById('visible-count').textContent=`显示 ${count} / ${cards.length} 张错误图 · ${panelsShown} 个错误模型面板`;}
for(const input of [model,work,query])input.addEventListener(input===query?'input':'change',update);for(const input of document.querySelectorAll('.prediction-model'))input.addEventListener('change',update);document.getElementById('reset').addEventListener('click',()=>{model.value='both';work.value='';query.value='';update();});
for(const checkbox of document.querySelectorAll('.show-review'))checkbox.addEventListener('change',()=>checkbox.closest('.case').querySelector('.image-wrap').classList.toggle('hide-review',!checkbox.checked));
const dialog=document.getElementById('zoom-dialog');for(const button of document.querySelectorAll('.zoom')){button.addEventListener('click',()=>{const card=button.closest('.case');document.getElementById('zoom-title').textContent=card.querySelector('h2').textContent;document.getElementById('zoom-content').replaceChildren(card.querySelector('.image-wrap').cloneNode(true));dialog.showModal();});button.closest('.case').querySelector('.image-wrap').addEventListener('click',()=>button.click());}document.getElementById('close-zoom').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});update();
"""


def render_report(summary, images):
    complete = summary["complete"]
    review = summary.get("review_selection") or select_error_review(summary)
    info = summary.get('annotation_review') or {}
    current_hash = info.get('current_dataset_hash')
    if info.get('applied') and current_hash and current_hash != summary['dataset_hash']:
        raise ValueError('Report annotations do not match the confirmed review version')
    reviewed = info.get('applied') and current_hash == summary['dataset_hash']
    revision = str(info.get('revision_id') or '当前确认版本').rsplit('_', 1)[-1]
    version = ('最终复核标注 · ' + revision if reviewed else '当前标注版本') + ' · ' + summary['dataset_hash'][:12]
    parts = ['<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>联网角色识别测试 · 错误案例</title><style>' + CSS + STYLE + '</style></head><body><main><header class="brief-header"><div><h1>联网角色识别测试</h1><p>' + str(summary['planned_images_per_model']) + '张图片 · ' + str(summary['planned_targets_per_model']) + '名必答角色 · 七模型</p></div><span class="review-status" data-dataset-hash="' + esc(summary['dataset_hash']) + '" title="标注指纹：' + esc(summary['dataset_hash']) + '">' + esc(version) + '</span></header>']
    if not summary.get('finalized'):
        status = '全部计划任务均有记录，仍在收尾或重试；结果尚未定稿' if complete else '进行中 / 尚未全部执行，以下不是最终结果' if summary['status'] in ('running', 'pending') else '已停止 / 尚未全部执行，以下不是全量结果'
        parts.append('<p class="status-note">' + status + ' · 已保存' + str(summary['completed_tasks']) + '/' + str(summary['planned_tasks']) + '项</p>')
    if summary.get('stage_label') and not summary.get('finalized'):
        parts.append('<p class="metric-caption">' + esc(summary['stage_label']) + '</p>')
    parts.append('<section class="panel overview"><h2>模型结果总览</h2>' + brief_results_table(summary) + '<p class="metric-caption">姓名命中率忽略坐标与格式，接口失败仍计入全部必答角色分母；调用成功率单独统计。</p></section>')
    parts.append('<section class="panel breakdown"><details><summary class="summary-toggle">联网前后对比与分类统计</summary><h3>全部四部</h3>' + comparison_table(summary['all_works']))
    labels = {"works": summary["work_names"], "single_multi": {"single": "单人图片", "multi": "多人图片"}, "orientation": POSE, "visibility": VISIBILITY}
    for axis, groups in summary["groups"].items():
        for value, comparison in groups.items():
            heading = labels[axis].get(value, value)
            parts.append('<details class="group-table"><summary>' + esc(heading) + '</summary>' + comparison_table(comparison) + '</details>')
    parts.append('<p class="metric-caption">作品、单人/多人按整图必答名额统计；朝向与可见性按全部具名人物统计，包含可选旁人。</p></details></section>')
    chips = ''.join('<span class="issue-chip">' + esc(MODEL_LABELS.get(model, model)) + '：' + str(review['model_error_images'][model]) + '张</span>' for model in review['top_models'])
    parts.append('<section class="error-heading"><h2>错误案例 · ' + str(review['selected_image_count']) + '张</h2><div>' + chips + '</div><p class="metric-caption">仅展示DeepSeek与Qwen Max按最终复核标注判定的姓名/API错误，不包含其余正确图；同一图片可有两个模型出错。</p></section>')
    model_options = '<option value="both" selected>两模型有错面板 · 同时对照</option>' + ''.join('<option value="' + esc(model) + '">' + esc(MODEL_LABELS.get(model, model)) + ' · ' + str(review["model_error_images"][model]) + '张错误图</option>' for model in review["top_models"])
    work_options = ''.join('<option value="' + esc(slug) + '">' + esc(name) + '</option>' for slug, name in summary["work_names"].items())
    parts.append('<div class="toolbar"><label>错误模型对照<select id="model">' + model_options + '</select></label><label>作品<select id="work"><option value="">全部作品</option>' + work_options + '</select></label><label>编号或角色<input id="search" type="search" placeholder="例如 0029 或 钰珑"></label><button id="reset" type="button">重置筛选</button><strong id="visible-count"></strong></div>')
    selected = {row["sample_id"]: row for row in review["selected_images"]}
    for case in summary["cases"]:
        sid, sample = case["sample_id"], case["sample"]
        if sid not in selected:
            continue
        image = images[sid]
        problems = selected[sid]["models"]
        search_terms = " ".join([sid, sample["work_name"]] + [p["character"] for p in sample["persons"]] + [str(p["character"] or "") for model in problems for p in case["results"][model]["new"]["names"]["predictions"]]).lower()
        chips = ''.join('<span class="issue-chip">' + esc(MODEL_LABELS.get(model, model)) + '</span>' for model in problems)
        prediction_options = '<option value="">仅人工真值，不叠预测框</option>' + ''.join('<option value="' + esc(model) + '">' + esc(MODEL_LABELS.get(model, model)) + '</option>' for model in problems)
        parts.append('<article class="case case-card" id="' + esc(sid) + '" data-work="' + esc(sample["work_slug"]) + '" data-search="' + esc(search_terms) + '"><div class="case-head"><h2>' + esc(sid) + ' · ' + esc(sample["work_name"]) + '</h2><div>' + chips + '</div></div><div class="case-body"><div class="case-visual"><div class="image-wrap"><img loading="lazy" width="' + str(image["size"][0]) + '" height="' + str(image["size"][1]) + '" alt="' + esc(sid) + ' · 实际无框输入" src="data:image/jpeg;base64,' + base64.b64encode(image["bytes"]).decode("ascii") + '"><div class="review-layer gt-layer">' + overlay(sample, {"parse_ok": False}, [], image["size"]) + '</div>')
        for model in problems:
            result = case["results"][model]
            new = result["new"]
            predictions = [{"bbox": p["original_bbox"], "character": p["character"]} for p in new["names"]["predictions"]]
            parts.append('<div class="review-layer prediction-layer" data-model="' + esc(model) + '" hidden>' + overlay(sample, new["strict"], predictions, image["size"]) + '</div>')
        parts.append('</div><div class="image-controls"><button class="zoom" type="button">放大图片</button><label><input class="show-review" type="checkbox" checked>人工标注</label><label>预测框<select class="prediction-model">' + prediction_options + '</select></label></div><p class="metric-caption prediction-caption">默认只显示人工标注。</p><p class="metric-caption image-note">绿框为最终复核标注，原始模型输入无框；点击图片可放大。</p></div><div class="case-analysis"><div class="model-panels">')
        for model in problems:
            result = case["results"][model]
            new = result["new"]
            state = "failed" if new["state"] != "ok" else "wrong"
            panel = model_panel(model, result, compact=True).replace('data-model="' + esc(model) + '"', 'data-model="' + esc(model) + '" data-result="' + state + '" data-failure="' + esc(new["failure_category"] or "") + '"', 1)
            parts.append(panel)
        parts.append('</div></div></div></article>')
    parts.append('<footer><p>仅姓名命中不保证姓名与屏幕人物的位置对应；多余姓名另记为未命中目标的返回条目。来源链接仅在点击时联网打开，报告自身没有外部资源依赖。</p></footer></main><dialog id="zoom-dialog"><div class="dialog-head"><strong id="zoom-title"></strong><button id="close-zoom" type="button">关闭（Esc）</button></div><div id="zoom-content"></div></dialog><script>' + JS + '</script></body></html>')
    return ''.join(parts)


def atomic_text(path, text):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix="." + path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def export_report(run_id, output, root=ROOT, baseline_id=BASELINE, stage_label=None, pause_note=None, model_runs=None):
    context = read_context(root, run_id, baseline_id, model_runs)
    summary = build_summary(context)
    if stage_label:
        summary["stage_label"] = stage_label
    if pause_note:
        summary["pause_note"] = pause_note
    content = render_report(summary, context["images"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_text(output / "index.html", content)
    summary["html_sha256"] = hashlib.sha256(content.encode()).hexdigest()
    atomic_text(output / "summary.json", json.dumps(summary, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    if summary["excluded_model_diagnostics"]:
        atomic_text(output / "excluded_model_diagnostics.json", json.dumps(summary["excluded_model_diagnostics"], ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/2026-11-15-c-search")
    parser.add_argument("--baseline-run-id", default=BASELINE)
    parser.add_argument("--stage-label", help="Optional visible label, e.g. a stopped validation cohort")
    parser.add_argument("--pause-note", help="Optional operator context shown only while this run is stopped")
    parser.add_argument("--model-run", action="append", default=[], metavar="MODEL=RUNID", help="Whole-model source override; repeat for multiple models. Overrides matching persisted manifest entries.")
    args = parser.parse_args()
    summary = export_report(args.run_id, args.output, baseline_id=args.baseline_run_id, stage_label=args.stage_label, pause_note=args.pause_note, model_runs=parse_model_runs(args.model_run))
    print(dump({key: summary[key] for key in ("run_id", "status", "complete", "completed_tasks", "planned_tasks")}))
    print("report:", args.output / "index.html")


if __name__ == "__main__":
    main()
