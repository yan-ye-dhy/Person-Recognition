"""本地 HTML 错误案例浏览器（自包含单文件，图片 base64 内嵌）。

展示：图片（匿名化正式集原图）+ 作品名称 + 正确角色 + 模型预测 + 原始响应 + 错误类型。
支持按模型/作品/错误类型客户端筛选。
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

from ..evaluation.scoring import OUTCOME_LABELS, ScoreResult

_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>影视角色识别 · 错误案例浏览器（{run_id}）</title>
<style>
  body {{ font-family: "Noto Sans CJK SC", "Microsoft YaHei", sans-serif; margin: 24px; background: #f7f7f8; }}
  h1 {{ font-size: 20px; }}
  .filters {{ margin: 12px 0 20px; display: flex; gap: 12px; flex-wrap: wrap; }}
  select {{ padding: 6px 10px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 16px; }}
  .card {{ background: #fff; border: 1px solid #e2e2e6; border-radius: 10px; padding: 12px; }}
  .card img {{ width: 100%; border-radius: 6px; }}
  .meta {{ font-size: 13px; margin-top: 8px; line-height: 1.6; }}
  .tag {{ display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px; color: #fff; margin-right: 6px; }}
  .correct {{ background: #4c9a62; }} .wrong_name {{ background: #d05c5c; }}
  .actor_name {{ background: #a06ab8; }} .uncertain {{ background: #c9a13d; }}
  .refused {{ background: #8d8d3a; }} .parse_error {{ background: #5f8fa8; }}
  .api_failure {{ background: #7a7a7a; }} .not_run {{ background: #b0b0b0; }}
  pre {{ white-space: pre-wrap; word-break: break-all; background: #f2f2f4; padding: 8px;
        border-radius: 6px; font-size: 12px; max-height: 180px; overflow: auto; }}
  .ok-note {{ color: #4c9a62; font-size: 12px; }}
</style>
</head>
<body>
<h1>影视角色识别 · 错误案例浏览器 <small style="color:#888">run {run_id}</small></h1>
<div class="filters">
  <label>模型 <select id="f-model" onchange="apply()"><option value="">全部</option>{model_options}</select></label>
  <label>作品 <select id="f-work" onchange="apply()"><option value="">全部</option>{work_options}</select></label>
  <label>结果 <select id="f-outcome" onchange="apply()"><option value="">全部（非正确）</option>{outcome_options}</select></label>
  <span id="count" style="align-self:center;color:#666"></span>
</div>
<div class="grid" id="grid"></div>
<script>
const CARDS = {cards_json};
function apply() {{
  const m = document.getElementById('f-model').value;
  const w = document.getElementById('f-work').value;
  const o = document.getElementById('f-outcome').value;
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  let n = 0;
  for (const c of CARDS) {{
    if (m && c.model !== m) continue;
    if (w && c.work !== w) continue;
    if (o && c.outcome !== o) continue;
    if (!o && c.outcome === 'correct') continue;
    n += 1;
    grid.insertAdjacentHTML('beforeend', c.html);
  }}
  document.getElementById('count').textContent = n + ' 条';
}}
apply();
</script>
</body>
</html>
"""


def render_html_report(
    result: ScoreResult,
    store: Any,
    dataset_dir: Path,
    out_path: Path,
    sample_images: dict[str, str] | None = None,  # sample_id → image_path(rel)
    max_cards: int = 400,
) -> Path:
    records_by_key = {
        (r["model_id"], r["experiment"], r["sample_id"]): r for r in store.records(result.run_id)
    }
    if sample_images is None:
        dataset_json = dataset_dir / "dataset.json"
        sample_images = {}
        if dataset_json.exists():
            payload = json.loads(dataset_json.read_text(encoding="utf-8"))
            sample_images = {s["sample_id"]: s["image_path"] for s in payload.get("samples", [])}

    cards: list[dict[str, str]] = []
    shown = 0
    for rec in sorted(
        result.records, key=lambda r: (r.outcome != "wrong_name", r.model_id, r.sample_id)
    ):
        if rec.outcome in {"correct", "not_run"}:
            continue
        if shown >= max_cards:
            break
        shown += 1
        img_html = "<div style='color:#999'>图片缺失</div>"
        rel = sample_images.get(rec.sample_id)
        if rel:
            img_path = dataset_dir / rel
            if img_path.is_file():
                b64 = base64.b64encode(img_path.read_bytes()).decode()
                img_html = f'<img src="data:image/jpeg;base64,{b64}" alt="{rec.sample_id}">'
        raw = records_by_key.get((rec.model_id, rec.experiment, rec.sample_id), {})
        raw_text = (raw.get("raw", {}).get("response", {}) or {}).get("raw_text", "")
        card_html = (
            f'<div class="card">{img_html}<div class="meta">'
            f'<span class="tag {rec.outcome}">{OUTCOME_LABELS.get(rec.outcome, rec.outcome)}</span>'
            f'<b>{html.escape(rec.model_id)}</b> · {rec.experiment}组 · {rec.sample_id}<br>'
            f'正确角色：<b>{html.escape(rec.gt_character)}</b> ｜ '
            f'模型预测：<b>{html.escape(rec.predicted or "（空）")}</b><br>'
            f'<span class="ok-note">{html.escape(rec.detail[:160])}</span>'
            f'<pre>{html.escape((raw_text or "")[:600])}</pre>'
            f"</div></div>"
        )
        cards.append(
            {
                "model": rec.model_id,
                "work": rec.work_slug,
                "outcome": rec.outcome,
                "html": card_html,
            }
        )

    def options(values: list[str]) -> str:
        return "".join(f'<option value="{v}">{html.escape(v)}</option>' for v in values)

    page = _TEMPLATE.format(
        run_id=result.run_id,
        model_options=options(sorted({c["model"] for c in cards})),
        work_options=options(sorted({c["work"] for c in cards})),
        outcome_options=options(
            [o for o in ["wrong_name", "actor_name", "uncertain", "refused", "parse_error", "api_failure"] if any(c["outcome"] == o for c in cards)]
        ),
        cards_json=json.dumps(cards, ensure_ascii=False),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page, encoding="utf-8")
    return out_path
