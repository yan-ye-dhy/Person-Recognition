"""Offline, sanitized billing reconciliation for a small real API probe.

Amounts come only from explicitly imported official billing evidence.  Token
prices, account balances and currency conversion are deliberately not inputs.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
from html import escape
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
SOURCE_LOOKUP_HTML = '''<section><h2>平台入口与待补资料</h2><ul>
<li><a href="https://platform.deepseek.com/usage" target="_blank" rel="noopener noreferrer">DeepSeek用量与费用</a>：下载用量明细和费用明细，核对请求、Token和搜索费用。</li>
<li><a href="https://platform.kimi.com/console" target="_blank" rel="noopener noreferrer">Kimi控制台</a>：进入费用明细，查询本批推理和搜索费用。请求日志只有Token用量，不能代替费用金额。<a href="https://www.kimi.com/help/kimi-api/api-balance-and-usage" target="_blank" rel="noopener noreferrer">官方说明</a>称当日账单次日07:00更新。</li>
<li><a href="https://open.bigmodel.cn/finance/expensebill/list?active=detail" target="_blank" rel="noopener noreferrer">智谱费用账单</a>：确认账单覆盖测试日期，可从<a href="https://open.bigmodel.cn/finance/exportrecord" target="_blank" rel="noopener noreferrer">导出记录</a>下载更新后的费用明细。</li>
<li><a href="https://bailian.console.aliyun.com/cn-beijing/costing-balance/overview" target="_blank" rel="noopener noreferrer">百炼费用概览</a>：按分钟筛选上表对应模型的北京时间窗口，并查看明细。还需核对单独出账的联网搜索费用。若此处导出仍按天汇总，可在<a href="https://billing-cost.console.aliyun.com/" target="_blank" rel="noopener noreferrer">阿里云费用与成本</a>查看更细的账单。</li>
</ul><p>平台下载的原始文件直接交给核对者即可，无需手工改列名；也可以提供费用明细截图。</p><p class="note">若费用只有混合其他调用的全天合计，或跨模型的工具合计，各模型本批费用继续待核。只有推理和搜索费用都核齐后，才计算本批实际均价。</p></section>'''


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
    return {"probe_id": _identifier(manifest.get("probe_id"), "probe ID"), "dataset_hash": digest,
            "planned_sample_ids": samples, "models": models, "historical_daily_bills": history,
            "historical_batches": historical_batches}


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
            total = _sum(part, "amount") if complete else None
            charges.append({"currency": currency, "official_amount": total,
                            "cash_amount": _sum(part, "cash_amount") if complete else None,
                            "credit_amount": _sum(part, "credit_amount") if complete else None,
                            "average_per_submitted_image": _average(total, model["submitted_count"]),
                            "entry_count": len(part), "kinds": sorted({entry["kind"] for entry in part})})
        models.append({**model, "billing_complete": complete, "billing_status": "official_batch_confirmed" if complete else "pending_official_billing",
                       "imported_entry_count": len(entries), "charges": charges})
    batches = [{**row, "average_per_image_task": _average(row['amount'], row['image_tasks'])}
               for row in public['historical_batches']]
    return {**{key: public[key] for key in ("probe_id", "dataset_hash", "planned_sample_ids", "historical_daily_bills")},
            "models": models, "historical_batches": batches,
            "generated_at": datetime.now(timezone.utc).astimezone(SHANGHAI).isoformat(timespec="seconds"),
            "money_policy": "Official imported batch amount / submitted images; no token estimates, balance deltas, FX or ambiguous prorating. Unknown is null, never zero.",
            "has_private_billing_evidence": bool(official["entries"])}


def render_report(manifest, output_path, billing=None):
    """Write one standalone HTML and return its safe summary (no model calls)."""
    summary = summarize_probe(manifest, billing)
    esc = lambda value: escape(str(value), quote=True)
    history_html = ""
    if summary["historical_daily_bills"]:
        history_parts = ['<section id="historical-billing"><h2>历史项目费用：官方已出日账</h2><p>以下按北京时间自然日展示导出文件中的已出账金额，供核对昨晚项目费用。跨过零点的记录记在次日；11月16日日账可能同时包含凌晨旧批次和今晚小批。这些金额不用于上方5张实测均价，搜索费用缺行时也不视为0元。</p><div class="scroll"><table><tr><th>账单日期</th><th>平台</th><th>模型 / 项目</th><th>计费项</th><th>官方日账金额</th><th>说明</th></tr>']
        kind_labels = {"inference": "推理", "search": "搜索", "combined": "合计"}
        for row in summary["historical_daily_bills"]:
            exact = row['amount']
            if exact is not None and '.' in exact:
                exact = exact.rstrip('0').rstrip('.')
            amount = '<span class="pending">缺金额明细</span>' if exact is None else esc(row['currency'] + ' ' + exact)
            note = HISTORY_NOTES[row['note_code']]
            if row['request_count'] is not None:
                note += "；调用记录" + str(row['request_count']) + "次"
            history_parts.append('<tr><td>' + esc(row['date']) + '</td><td>' + esc(row['vendor']) + '</td><td>' + esc(HISTORY_ITEMS[row['model_id']]) + '</td><td>' + kind_labels[row['kind']] + '</td><td>' + amount + '</td><td>' + esc(note) + '</td></tr>')
        history_parts.append('</table></div><p class="muted">“官方日账金额”保留各平台消费或应付口径，不统一称作现金实扣；资源包抵扣和未出账金额另行说明。</p></section>')
        if summary['historical_batches']:
            history_parts.append('<section><h2>已核完整历史批次的单图均价</h2><p>只对推理、搜索费用与图片任务数量均已核齐的批次计算；多轮推理和重试计入费用，每个图片任务在分母中只计一次。</p><table><tr><th>北京时间日期</th><th>模型</th><th>官方费用合计</th><th>图片任务数</th><th>平均每图片任务</th></tr>')
            for row in summary['historical_batches']:
                with localcontext() as context:
                    context.prec = 100
                    average = format(Decimal(row['average_per_image_task']).quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP), 'f')
                history_parts.append('<tr><td>' + esc(row['date']) + '</td><td>' + esc(LABELS[row['model_id']]) + '</td><td>' + esc(row['currency'] + ' ' + row['amount']) + '</td><td>' + str(row['image_tasks']) + '</td><td>' + esc(row['currency'] + ' ' + average) + '</td></tr>')
            history_parts.append('</table><p class="muted">历史批次可能包含接口调试和失败重试，均价只描述该批实际消耗，不是未来固定单价；同一原图在不同批次再次测试属于不同图片任务。</p></section>')
        history_html = ''.join(history_parts)
    def money(charges, field):
        if not charges:
            return '<span class="pending">待官方账单核对</span>'
        def display(value):
            if value is None:
                return '<span class="pending">待核</span>'
            if field != "average_per_submitted_image":
                return esc(value)
            with localcontext() as context:
                context.prec = 100
                number = Decimal(value)
                rounded = number.quantize(Decimal("0.00000001"), rounding=ROUND_HALF_UP)
            if number and not rounded:
                return '&lt;0.00000001'
            return esc(format(rounded, "f").rstrip("0").rstrip(".") if rounded else "0")
        return "<br>".join(esc(row["currency"]) + " " + display(row[field]) for row in charges)
    parts = ['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实小批调用与官方账单核对</title><style>
*{box-sizing:border-box}body{margin:0;background:#f2f5f8;color:#20344a;font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif}main{max-width:1550px;margin:auto;padding:24px}section,header{background:white;border:1px solid #d4dfe9;border-radius:12px;padding:20px;margin-bottom:20px}h1{margin:0;font-size:27px}h2{font-size:21px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:12px;border:1px solid #d9e3ec;text-align:left;vertical-align:top}th{background:#edf4f9;white-space:nowrap}.pending{color:#8a5100}.muted{color:#65758a;font-size:14px}code{overflow-wrap:anywhere}a{color:#075d87}.note{background:#fff4df;border-left:4px solid #d3932c;padding:12px}td.cost{overflow-wrap:anywhere;max-width:230px}@media(max-width:650px){main{padding:10px}header,section{padding:14px}h1{font-size:23px}}
</style><main><header><h1>真实小批调用与官方账单核对</h1>''']
    planned = sum(row["planned_count"] for row in summary["models"])
    submitted = sum(row["submitted_count"] for row in summary["models"])
    parts.append(f'<p>计划{len(summary["models"])}个模型 × {len(summary["planned_sample_ids"])}张图片＝{planned}项；已提交{submitted}项。时间均为北京时间（Asia/Shanghai）。</p><p class="note">金额只采用已导入并确认完整的官方账单，不用Token单价估算或账户余额相减。未取得账单显示“待核”，不当成0元；接口成功也不代表费用已结算。</p>')
    parts.append(f'<p class="muted">批次：<code>{esc(summary["probe_id"])}</code> · 报告生成：{esc(summary["generated_at"])}</p></header><section><h2>调用结果与批次实际费用</h2><div class="scroll"><table><tr><th>模型</th><th>提交 / 计划</th><th>成功 / 失败 / 未终结</th><th>开始 / 结束（北京时间）</th><th>推理HTTP / 独立搜索HTTP</th><th>厂商回报的原生工具用量</th><th>官方批次金额</th><th>现金支付</th><th>抵扣金额</th><th>批次金额 ÷ 已提交图片</th><th>账单状态</th></tr>')
    for row in summary["models"]:
        unknown = row["submitted_count"] - row["success_count"] - row["failure_count"]
        attempts = lambda value: "未知" if value is None else str(value)
        state = f'已核完整官方账单 · {row["imported_entry_count"]}条' if row["billing_complete"] else f'待核 · 已导入{row["imported_entry_count"]}条'
        evidence_status = "matched_complete" if row["billing_complete"] else row["billing_evidence_status"]
        evidence_note = EVIDENCE_STATUS[evidence_status]
        if evidence_status == "date_not_covered" and row.get("billing_latest_date"):
            evidence_note += "（目前仅到" + row["billing_latest_date"] + "）"
        state_html = esc(state) + '<br><span class="muted">' + esc(evidence_note) + '</span>'
        units = '<br>'.join(esc(key) + '：' + str(value) for key, value in row["native_tool_units"].items())
        known = row["native_tool_usage_known_attempts"]
        coverage = '已知用量来自' + str(known) + '次推理回包' if known else '未取得可核的原生工具用量'
        native = (units + '<br>' if units else '') + '<span class="muted">' + coverage + '</span>'
        parts.append('<tr><td><b>' + esc(LABELS.get(row["model_id"], row["model_id"])) + '</b><br><span class="muted">' + esc(row["vendor"]) + ' · ' + esc(row["status"]) + '</span></td><td>' + f'{row["submitted_count"]} / {row["planned_count"]}' + '</td><td>' + f'{row["success_count"]} / {row["failure_count"]} / {unknown}' + '</td><td>' + esc(row["started_at"] or "尚未开始") + '<br>' + esc(row["finished_at"] or "尚未结束") + '</td><td>' + esc(attempts(row["api_attempts"])) + ' / ' + esc(attempts(row["search_attempts"])) + '</td><td>' + native + '</td>' + ''.join('<td class="cost">' + money(row["charges"], field) + '</td>' for field in ("official_amount", "cash_amount", "credit_amount", "average_per_submitted_image")) + '<td>' + state_html + '</td></tr>')
    parts.append('''</table></div><p class="muted">“独立搜索HTTP”仅统计本地另行发出的搜索接口请求；DeepSeek/Qwen等在推理请求内部执行的原生搜索不计入该列，所以0次独立搜索HTTP不代表未联网。原生工具用量仅展示厂商明确回报的计量单位，未回报为未知；它不是网址数量，也不是实际账单金额。</p><p class="muted">均价分母为已提交图片，包括失败；额外API重试另列。这是本次小批实际平均费用，不是每一张图片各自的账单。不同币种分别展示，未换汇。现金、抵扣和账单金额保留导入时的官方口径，不把充值当消费，也不将三列重复相加。</p></section><section><h2>如何取得和导入官方费用</h2><ol><li>在各厂商账单页面按本批模型、专用Key/业务空间及调用时间筛选，核对调用数量和所有计费项。联网搜索可能单独计费，须一并核入。</li><li>保留原官方导出文件，核对每项费用的模型、币种、金额和计费时段。共同账单不能确定归属时先留待核。</li><li>确认本模型推理和工具费用都已出齐后，再确认完整费用。汇总金额与其分项金额只计一次，同一条账单也只计一次。</li><li>免费额度、优惠券、订阅Credits、缓存折扣和现金支付分别核对；抵扣或现金未知仍显示待核，账单0元必须有明确证据。</li></ol><p>百炼官方说明推理账单通常延迟2–10分钟，繁忙时可能更久；账户余额存在预占。账单查询API还有独立更新周期，当月账单可调整，因此未确认的费用不能提前填为实际扣款。</p><p><a href="https://help.aliyun.com/zh/model-studio/bill-query-and-cost-management" target="_blank" rel="noopener noreferrer">百炼账单查询与费用字段</a> · <a href="https://help.aliyun.com/zh/model-studio/model-usage-statistics" target="_blank" rel="noopener noreferrer">百炼模型及工具用量</a> · <a href="https://help.aliyun.com/zh/model-studio/web-search" target="_blank" rel="noopener noreferrer">百炼联网搜索计费</a> · <a href="https://help.aliyun.com/zh/user-center/developer-reference/api-bssopenapi-2017-12-14-describeinstancebill" target="_blank" rel="noopener noreferrer">百炼账单API更新周期</a></p><p class="muted">报告为单文件离线HTML，无外部脚本、图片或自动请求。账户余额、账号、API Key、原始账单ID和证据路径不写入此页面。</p></section></main></html>''')
    parts[-1] = parts[-1].replace('不同币种分别展示，未换汇。', '均价展示四舍五入至最多8位小数，精确计算值保留在结构化结果中；官方账单原金额不改动。不同币种分别展示，未换汇。').replace('</main></html>', SOURCE_LOOKUP_HTML + '</main></html>')
    if history_html:
        parts[-1] = parts[-1].replace('<section><h2>如何取得和导入官方费用</h2>', history_html + '<section><h2>如何取得和导入官方费用</h2>')
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, prefix="." + output.name, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write("".join(parts))
    temporary.replace(output)
    return summary
