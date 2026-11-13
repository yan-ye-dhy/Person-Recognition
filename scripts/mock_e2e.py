#!/usr/bin/env python
"""Mock 端到端演示：不花一分钱跑通 采集(模拟)→标注(脚本)→导出→评测→评分→报告 全链路。

用法：.venv/bin/python scripts/mock_e2e.py
产物：data/demo/（演示数据集）与 results/reports/run_demo/（演示报告，内容为 Mock 数据，
仅用于验证管线，不代表任何真实模型结论）。
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from vcrb.annotation.rosters import CharacterEntry, Roster
from vcrb.config import load_settings, load_works
from vcrb.dataset.schema import load_dataset
from vcrb.dataset.store import AnnotationStore, CandidateItem, export_dataset
from vcrb.evaluation.matcher import CharacterMatcher
from vcrb.evaluation.scoring import save_score_result, score_run
from vcrb.models import ModelSpec
from vcrb.providers import MockProvider
from vcrb.reports.charts import render_charts
from vcrb.reports.html_report import render_html_report
from vcrb.reports.markdown_report import render_markdown_report
from vcrb.runner.executor import dataset_hash, execute_run
from vcrb.runner.store import RunStore

DEMO = Path("data/demo")
RUN_ID = "run_demo"


def make_noise(path: Path, seed: int) -> None:
    rng = random.Random(seed)
    img = Image.new("RGB", (640, 360))
    px = img.load()
    for y in range(0, 360, 4):
        for x in range(0, 640, 4):
            c = (rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for dy in range(4):
                for dx in range(4):
                    px[x + dx, y + dy] = c
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG")


def mock_model(mid: str) -> ModelSpec:
    return ModelSpec(id=mid, provider="mock", vendor="Mock", display_name=f"Mock·{mid}",
                     official_series="离线测试桩", tier="standard", supports_vision="doc",
                     enabled=True)


def main() -> None:
    works = load_works()
    settings = load_settings()
    if DEMO.exists():
        shutil.rmtree(DEMO)

    # 1) 模拟候选（两作品各 6 张，直接落 candidates 布局）
    candidates = DEMO / "candidates"
    plan = {
        "kuangbiao": [("高启强", "main", "normal", False), ("安欣", "main", "normal", False),
                      ("陈书婷", "key_supporting", "normal", True), ("高启盛", "key_supporting", "side_face", False),
                      ("李响", "supporting", "blurry", False), ("老默", "supporting", "occluded", True)],
        "qingyunian2": [("范闲", "main", "normal", False), ("庆帝", "main", "normal", True),
                       ("林婉儿", "key_supporting", "normal", False), ("王启年", "key_supporting", "costume_change", False),
                       ("范若若", "supporting", "normal", False), ("陈萍萍", "supporting", "occluded", True)],
    }
    store = AnnotationStore(DEMO / "annotations.db")
    seed = 0
    for slug, rows in plan.items():
        items = []
        for i, (char, imp, diff, multi) in enumerate(rows, start=1):
            seed += 1
            name = f"cand_{i:04d}.jpg"
            make_noise(candidates / slug / name, seed)
            items.append(CandidateItem(slug, name, 640, 360, source_type="official_still",
                                       source_url="mock://demo"))
        store.register_candidates(slug, items, candidates)
        # 直接写 verified 标注（演示中代替人工标注）
        for row, (char, imp, diff, multi) in zip(store.samples(slug), rows):
            store.update_annotation(row["pk"], {
                "status": "verified", "target_character": char,
                "character_importance": imp, "visual_difficulty": diff,
                "is_multi_person": int(multi),
                "target_bbox": [0.15, 0.1, 0.65, 0.9] if multi else None,
                "tags": [],
            })
            # 归属核验门禁：演示数据全部标记为已核实（真实流程须人工批量审核）
            store.set_attribution(
                [row["pk"]], "verified", ["Mock 演示数据，归属核验默认通过"],
                actor="user",
            )

    # 2) 导出匿名化正式集
    dataset_json = export_dataset(store, works, out_dir=DEMO / "final",
                                  candidates_dir=candidates)
    samples = load_dataset(dataset_json)
    print(f"[1/5] 演示数据集导出：{len(samples)} 样本 → {dataset_json}")

    # 3) Mock 评测（4 个桩模型 × A/B 两组）
    # mock-good 按 dataset 顺序给出正确答案（演示 correct 路径）
    good_answers = [
        {"text": json.dumps({"character_name": c, "is_uncertain": False, "reason": "mock"}, ensure_ascii=False)}
        for rows in plan.values()
        for (c, *_rest) in rows
    ]
    scripts = {
        "mock-good": good_answers,
        "mock-actor": [{"text": json.dumps({"character_name": "演员甲", "is_uncertain": False, "reason": "demo"}, ensure_ascii=False)}],
        "mock-refuse": [{"text": "很抱歉，我无法识别人脸。"}],
        "mock-flaky": [{"fail": "server_error"}],
    }
    run_store = RunStore(DEMO / "runs.db")
    if run_store.get_run(RUN_ID):
        print("已存在 run_demo，跳过评测（删除 data/demo 可重跑）")
    else:
        run_store.create_run(RUN_ID, ["A", "B"], list(scripts), "v1",
                             dataset_hash(samples), str(dataset_json),
                             {"note": "MOCK 演示数据，不代表真实模型"})
        tasks = [
            (mock_model(mid), MockProvider(script={mid: script} if script else None))
            for mid, script in scripts.items()
        ]
        summary = execute_run(
            run_id=RUN_ID, tasks=tasks, samples=samples,
            works_by_slug={w.slug: w for w in works}, experiments=["A", "B"],
            store=run_store, settings=settings, dataset_dir=DEMO / "final",
            marked_dir=DEMO / "marked", raw_dir=DEMO / "raw",
        )
        print(f"[2/5] Mock 评测：执行 {summary.executed} 成功 {summary.ok} 失败 {summary.api_failures}")

    # 4) 评分
    rosters = {}
    for w in works:
        roster = Roster(work_slug=w.slug, work_name=w.name)
        for slug_rows in plan.get(w.slug, []):
            roster.add(CharacterEntry(name=slug_rows[0], aliases=["强哥"] if slug_rows[0] == "高启强" else [],
                                      actors=["演员甲"], importance=slug_rows[1]))
        rosters[w.slug] = roster
    matcher = CharacterMatcher(rosters)
    result = score_run(run_store, RUN_ID, samples, matcher, settings)
    files = save_score_result(result, DEMO / "scores")
    print(f"[3/5] 评分：metrics/detail/summary → {files['metrics'].parent}")

    # 5) 报告三件套
    out = Path("results/reports") / RUN_ID
    charts = render_charts(result, out)
    md = render_markdown_report(result, out / "report.md",
                                models_info={m: {"vendor": "Mock", "tier": "standard",
                                                 "display_name": f"Mock·{m}", "official_series": ""}
                                             for m in scripts},
                                run_meta=run_store.get_run(RUN_ID),
                                chart_names=[p.name for p in charts])
    html_path = render_html_report(result, run_store, DEMO / "final", out / "errors.html")
    print(f"[4/5] 报告：{md}\n       错误浏览器：{html_path}")
    print("[5/5] 完成 —— 所有数字均为 Mock 数据，仅验证管线。")


if __name__ == "__main__":
    main()
