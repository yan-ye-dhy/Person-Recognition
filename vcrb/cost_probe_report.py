"""Offline, sanitized billing reconciliation for a small real API probe.

Amounts come only from explicitly imported official billing evidence.  Token
prices, account balances and currency conversion are deliberately not inputs.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, localcontext
import json
from pathlib import Path
import re
import tempfile
from zoneinfo import ZoneInfo

SHANGHAI = ZoneInfo("Asia/Shanghai")
KINDS = {"inference", "search", "combined"}
STATUSES = {"pending", "running", "completed", "complete", "done", "failed", "limited", "stopped", "awaiting_billing"}
CSV_FIELDS = ("bill_id", "model_id", "currency", "amount", "kind", "cash_amount", "credit_amount", "source", "period_start", "period_end")
LABELS = {"deepseek-flash": "DeepSeek Flash", "kimi-k3.1-pro": "Kimi K3.1 Pro", "glm-5.3-flash": "GLM 5.3 Flash", "glm-4.6v": "GLM 4.6V", "qwen3.8-max": "Qwen Max", "qwen3.7-plus": "Qwen Plus", "qwen3.8-flash": "Qwen Flash"}
EVIDENCE_STATUS = {
    "not_provided": "尚未取得可归属本批的完整费用明细",
    "matched_complete": "官方金额及用量已与本批核对",
    "request_log_only": "请求用量已核对；文件不含费用金额，待补费用明细",
    "date_not_covered": "现有账单尚未覆盖本次测试日期",
    "daily_aggregate_not_attributed": "按日汇总用量尚不能单独归属本次5张；需本批时间段明细及搜索费用",
    "inference_only": "已核推理费用；搜索费用尚待核齐",
}
HISTORY_ITEMS = {**LABELS, "search-pro": "智谱 SearchPro 搜索",
                 "server-tool-image-search": "DeepSeek 图片搜索",
                 "server-tool-search": "DeepSeek 文本搜索"}
HISTORY_NOTES = {
    "posted": "官方已出日账金额",
    "resource_pack": "资源包抵扣，未计资源包购置成本",
    "search_unallocated": "搜索合计，未分摊到两个模型",
    "inference_only": "仅见推理费用，搜索费用未核齐",
    "day_mixed": "同日包含凌晨旧批次和今晚小批",
    "no_amount": "调用记录已核对，文件没有费用金额",
}


def _load(value):
    if isinstance(value, (str, Path)):
        value = json.loads(Path(value).read_text(encoding="utf-8-sig"), parse_float=Decimal)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _identifier(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", value) or value.startswith(("sk-", "AKIA", "LTAI")):
        raise ValueError(f"Invalid public {field}")
    return value


def _count(value, field, *, unknown=False):
    if value is None and unknown:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Invalid {field} count")
    return value


def _time(value, field):
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"Invalid {field} timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid {field} timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Timezone required for {field}")
    return parsed.astimezone(SHANGHAI).isoformat(timespec="seconds")


def _money(value, field):
    # Never stringify binary floats into apparently exact bill amounts.
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError(f"{field} must be an exact decimal string")
    text = str(value)
    if not re.fullmatch(r"\d+(?:\.\d+)?", text) or len(text) > 80:
        raise ValueError(f"Invalid nonnegative decimal {field}")
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal {field}") from exc
    if not number.is_finite():
        raise ValueError(f"Invalid decimal {field}")
    return format(number, "f")


def validate_manifest(manifest):
    """Return only report-safe public metadata; ignore private extra fields."""
    manifest = _load(manifest)
    ids = manifest.get("planned_sample_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(sid, str) for sid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Expected unique planned sample IDs")
    samples = [_identifier(sid, "sample ID") for sid in ids]
    digest = manifest.get("dataset_hash")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Invalid dataset hash")
    models, seen = [], set()
    source_models = manifest.get("models")
    if not isinstance(source_models, list):
        raise ValueError("Models must be a list")
    for original in source_models:
        if not isinstance(original, dict):
            raise ValueError("Invalid model record")
        mid = _identifier(original.get("model_id"), "model ID")
        if mid in seen:
            raise ValueError("Duplicate model ID")
        seen.add(mid)
        row = {"model_id": mid, "vendor": _identifier(original.get("vendor"), "vendor"),
               "planned_count": _count(original.get("planned_count", len(samples)), "planned"),
               "submitted_count": _count(original.get("submitted_count", 0), "submitted"),
               "success_count": _count(original.get("success_count", 0), "success"),
               "failure_count": _count(original.get("failure_count", 0), "failure"),
               "api_attempts": _count(original.get("api_attempts"), "API attempts", unknown=True),
               "search_attempts": _count(original.get("search_attempts"), "search attempts", unknown=True),
               "native_tool_usage_known_attempts": _count(original.get("native_tool_usage_known_attempts"), "native tool known attempts", unknown=True),
               "status": original.get("status", "pending"),
               "started_at": _time(original.get("started_at"), "started_at"),
               "finished_at": _time(original.get("finished_at"), "finished_at")}
        if not isinstance(row["status"], str) or row["status"] not in STATUSES:
            raise ValueError("Invalid model status")
        if row["planned_count"] != len(samples) or row["submitted_count"] > row["planned_count"] or row["success_count"] + row["failure_count"] > row["submitted_count"]:
            raise ValueError("Inconsistent submitted/success/failure counts")
        if row["started_at"] and row["finished_at"] and row["finished_at"] < row["started_at"]:
            raise ValueError("Finish precedes start")
        units = original.get("native_tool_units", {})
        if not isinstance(units, dict):
            raise ValueError("Native tool units must be an explicit provider-reported object")
        row["native_tool_units"] = {_identifier(key, "tool unit"): _count(value, "native tool unit") for key, value in units.items()}
        if original.get("run_id"):
            row["run_id"] = _identifier(original["run_id"], "run ID")
        evidence_status = original.get("billing_evidence_status", "not_provided")
        if not isinstance(evidence_status, str) or evidence_status not in EVIDENCE_STATUS:
            raise ValueError("Invalid billing evidence status")
        row["billing_evidence_status"] = evidence_status
        latest_date = original.get("billing_latest_date")
        if latest_date is not None:
            if not isinstance(latest_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", latest_date):
                raise ValueError("Invalid billing latest date")
            datetime.fromisoformat(latest_date)
            row["billing_latest_date"] = latest_date
        models.append(row)
    if not models:
        raise ValueError("No models in manifest")
    history, history_seen = [], set()
    historical_rows = manifest.get("historical_daily_bills", [])
    if not isinstance(historical_rows, list):
        raise ValueError("Historical bills must be a list")
    for original in historical_rows:
        if not isinstance(original, dict):
            raise ValueError("Invalid historical bill row")
        day, mid, vendor = original.get("date"), original.get("model_id"), original.get("vendor")
        if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            raise ValueError("Invalid historical bill date")
        datetime.fromisoformat(day)
        if not isinstance(mid, str) or mid not in HISTORY_ITEMS or not isinstance(vendor, str) or vendor not in {"DeepSeek", "Kimi", "GLM", "Qwen"}:
            raise ValueError("Invalid historical bill model or vendor")
        kind, currency = original.get("kind"), original.get("currency")
        if not isinstance(kind, str) or kind not in KINDS or not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency):
            raise ValueError("Invalid historical bill kind or currency")
        key = (day, vendor, mid, kind, currency)
        if key in history_seen:
            raise ValueError("Duplicate historical bill aggregate")
        history_seen.add(key)
        note = original.get("note_code", "posted")
        if not isinstance(note, str) or note not in HISTORY_NOTES:
            raise ValueError("Invalid historical bill note")
        amount = original.get("amount")
        history.append({"date": day, "model_id": mid, "vendor": vendor, "kind": kind,
                        "currency": currency, "amount": _money(amount, "historical amount") if amount is not None else None,
                        "request_count": _count(original.get("request_count"), "historical requests", unknown=True),
                        "note_code": note})
    historical_batches, batch_seen = [], set()
    batch_rows = manifest.get("historical_batches", [])
    if not isinstance(batch_rows, list):
        raise ValueError("Historical batches must be a list")
    for original in batch_rows:
        if not isinstance(original, dict):
            raise ValueError("Invalid historical batch row")
        day, mid, currency = original.get("date"), original.get("model_id"), original.get("currency")
        if not isinstance(day, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
            raise ValueError("Invalid historical batch date")
        datetime.fromisoformat(day)
        if not isinstance(mid, str) or mid not in LABELS or currency != "CNY":
            raise ValueError("Invalid historical batch model or currency")
        key = (day, mid, currency)
        if key in batch_seen:
            raise ValueError("Duplicate historical batch")
        batch_seen.add(key)
        tasks = _count(original.get("image_tasks"), "historical image tasks")
        if tasks == 0 or original.get("inference_and_search_confirmed") is not True:
            raise ValueError("Historical average requires image tasks and complete fees")
        historical_batches.append({"date": day, "model_id": mid, "currency": currency,
            "image_tasks": tasks, "amount": _money(original.get("amount"), "historical batch amount"),
            "inference_and_search_confirmed": True})
    references, reference_seen = [], set()
    reference_rows = manifest.get("daily_reference_tasks", [])
    if not isinstance(reference_rows, list):
        raise ValueError('Daily reference tasks must be a list')
    for original in reference_rows:
        if not isinstance(original, dict):
            raise ValueError("Invalid daily reference task record")
        day, mid = original.get('date'), original.get('model_id')
        if not isinstance(day, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', day):
            raise ValueError('Invalid daily reference date')
        datetime.fromisoformat(day)
        if not isinstance(mid, str) or mid not in LABELS or (day, mid) in reference_seen:
            raise ValueError('Invalid or duplicate daily reference model')
        reference_seen.add((day, mid))
        tasks = _count(original.get('image_tasks'), 'daily reference image tasks')
        scope = original.get('fee_scope')
        if not tasks or scope not in ('inference_and_search', 'inference_only'):
            raise ValueError('Daily reference needs positive image tasks and an explicit fee scope')
        source_rows = [row for row in history if row['date'] == day and row['model_id'] == mid
                       and row['currency'] == 'CNY' and row['amount'] is not None
                       and (scope == 'inference_and_search' or row['kind'] == 'inference')]
        kinds = {row['kind'] for row in source_rows}
        if not source_rows or ('combined' in kinds and len(kinds) > 1):
            raise ValueError('Daily reference needs unambiguous official cost rows')
        if scope == 'inference_and_search' and kinds not in ({'combined'}, {'inference', 'search'}):
            raise ValueError('Search-inclusive reference requires official search costs')
        flags = {key: original.get(key, False) for key in ('contains_debugging', 'cross_midnight')}
        if any(type(value) is not bool for value in flags.values()):
            raise ValueError('Daily reference flags must be boolean')
        references.append({'date': day, 'model_id': mid, 'image_tasks': tasks, 'fee_scope': scope,
                           'amount': _sum(source_rows, 'amount'), 'currency': 'CNY', **flags})
    return {"probe_id": _identifier(manifest.get("probe_id"), "probe ID"), "dataset_hash": digest,
            "planned_sample_ids": samples, "models": models, "historical_daily_bills": history,
            "historical_batches": historical_batches, "daily_reference_tasks": references}


def validate_billing(manifest, billing=None):
    """Validate official entries without guessing ownership or prorating costs.

    ``complete_models`` is the importer's explicit assertion that all inference
    and tool fees for those batches have appeared.  Bill IDs identify unique
    detail rows, not an invoice number shared by several rows.
    """
    manifest = validate_manifest(manifest)
    billing = {"entries": [], "complete_models": []} if billing is None else _load(billing)
    models = {row["model_id"]: row for row in manifest["models"]}
    complete = billing.get("complete_models", [])
    if not isinstance(complete, list) or any(not isinstance(mid, str) or mid not in models for mid in complete) or len(set(complete)) != len(complete):
        raise ValueError("Invalid or duplicate complete_models")
    raw_entries = billing.get("entries", [])
    if not isinstance(raw_entries, list):
        raise ValueError("Billing entries must be a list")
    entries, seen = [], set()
    for original in raw_entries:
        if not isinstance(original, dict):
            raise ValueError("Invalid billing entry")
        bid = original.get("bill_id")
        if not isinstance(bid, str) or not bid.strip() or len(bid) > 500:
            raise ValueError("A unique official bill detail ID is required")
        bid = bid.strip()
        if bid in seen:
            raise ValueError("Duplicate bill_id; do not import the same charge twice")
        seen.add(bid)
        mid = original.get("model_id")
        if not isinstance(mid, str) or mid not in models:
            raise ValueError("Bill cannot be assigned to an unknown or ambiguous model")
        currency = original.get("currency")
        if not isinstance(currency, str) or not re.fullmatch(r"[A-Z]{3}", currency):
            raise ValueError("Use the official uppercase currency code")
        kind = original.get("kind")
        if not isinstance(kind, str) or kind not in KINDS:
            raise ValueError("Invalid bill kind")
        if not isinstance(original.get("source"), str) or not original["source"].strip():
            raise ValueError("Official billing evidence source is required")
        start, end = _time(original.get("period_start"), "period_start"), _time(original.get("period_end"), "period_end")
        if not start or not end or end < start:
            raise ValueError("A valid bill period is required")
        entry = {"bill_id": bid, "model_id": mid, "currency": currency, "amount": _money(original.get("amount"), "amount"),
                 "kind": kind, "source": original["source"], "period_start": start, "period_end": end}
        for key in ("cash_amount", "credit_amount"):
            if original.get(key) not in (None, ""):
                entry[key] = _money(original[key], key)
        entries.append(entry)
    for mid, row in models.items():
        owned = [entry for entry in entries if entry["model_id"] == mid]
        kinds = {entry["kind"] for entry in owned}
        if "combined" in kinds and len(kinds) > 1:
            raise ValueError("Do not add combined totals to inference/search components")
        if mid in complete and (not owned or row["status"] in {"pending", "running"} or row["success_count"] + row["failure_count"] != row["submitted_count"]):
            raise ValueError("Complete billing requires evidence for a finished batch")
    return {"entries": entries, "complete_models": list(complete)}


def import_billing_csv(manifest, csv_path, *, complete_models=()):
    """Import the explicit mapping template, never infer models from shared bills."""
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = set(CSV_FIELDS) - {"cash_amount", "credit_amount"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("CSV does not match the explicit official-bill mapping template")
        entries = [{key: value for key, value in row.items() if key in CSV_FIELDS and value not in (None, "")} for row in reader]
    return validate_billing(manifest, {"entries": entries, "complete_models": list(complete_models)})


def write_billing_template(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        csv.writer(stream).writerow(CSV_FIELDS)


def _sum(entries, key):
    if not entries or any(key not in entry for entry in entries):
        return None
    with localcontext() as context:
        context.prec = 100
        return format(sum((Decimal(entry[key]) for entry in entries), Decimal(0)), "f")


def _average(amount, submitted):
    if amount is None or not submitted:
        return None
    with localcontext() as context:
        context.prec = 100
        return format(Decimal(amount) / submitted, "f")


def summarize_probe(manifest, billing=None):
    public = validate_manifest(manifest)
    official = validate_billing(public, billing)
    models = []
    for model in public["models"]:
        mid = model["model_id"]
        entries = [entry for entry in official["entries"] if entry["model_id"] == mid]
        complete = mid in official["complete_models"]
        charges = []
        for currency in sorted({entry["currency"] for entry in entries}):
            part = [entry for entry in entries if entry["currency"] == currency]
            posted = _sum(part, "amount")
            total = posted if complete else None
            charges.append({"currency": currency, "official_amount": total,
                            "posted_amount": posted,
                            "posted_average_per_submitted_image": _average(posted, model["submitted_count"]),
                            "posted_by_kind": {kind: _sum([entry for entry in part if entry['kind'] == kind], 'amount')
                                               for kind in sorted({entry['kind'] for entry in part})},
                            "cash_amount": _sum(part, "cash_amount") if complete else None,
                            "credit_amount": _sum(part, "credit_amount") if complete else None,
                            "average_per_submitted_image": _average(total, model["submitted_count"]),
                            "entry_count": len(part), "kinds": sorted({entry["kind"] for entry in part})})
        models.append({**model, "billing_complete": complete, "billing_status": "official_batch_confirmed" if complete else "pending_official_billing",
                       "imported_entry_count": len(entries), "charges": charges})
    batches = [{**row, "average_per_image_task": _average(row['amount'], row['image_tasks'])}
               for row in public['historical_batches']]
    return {**{key: public[key] for key in ("probe_id", "dataset_hash", "planned_sample_ids", "historical_daily_bills", "daily_reference_tasks")},
            "models": models, "historical_batches": batches,
            "generated_at": datetime.now(timezone.utc).astimezone(SHANGHAI).isoformat(timespec="seconds"),
            "money_policy": "Official imported batch amount / submitted images; no token estimates, balance deltas, FX or ambiguous prorating. Unknown is null, never zero.",
            "has_private_billing_evidence": bool(official["entries"])}


def render_report(manifest, output_path, billing=None, recognition=None):
    """Write a compact, standalone briefing; retain full facts in the summary."""
    from .cost_probe_presentation import build_presentation, render_presentation

    summary = summarize_probe(manifest, billing)
    presentation = build_presentation(summary, _load(recognition) if recognition is not None else None)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, prefix="." + output.name, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(render_presentation(summary, presentation))
    temporary.replace(output)
    return {**summary, "presentation": presentation}
