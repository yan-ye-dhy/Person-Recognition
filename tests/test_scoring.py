"""评分与统计测试：七类判定、双口径、Wilson CI、McNemar、按角色宏平均、改判。"""

from __future__ import annotations

import json
import math

from tests.conftest import make_sample
from vcrb.annotation.rosters import CharacterEntry, Roster
from vcrb.evaluation.matcher import CharacterMatcher
from vcrb.evaluation.scoring import (
    mcnemar,
    save_score_result,
    score_run,
    wilson_ci,
)
from vcrb.runner.store import RunStore


def _roster() -> Roster:
    return Roster(
        work_slug="kuangbiao",
        work_name="狂飙",
        characters=[
            CharacterEntry(name="高启强", aliases=["强哥"], actors=["张颂文"], importance="main"),
            CharacterEntry(name="高启盛", actors=["苏小玎"], importance="key_supporting"),
            CharacterEntry(name="安欣", aliases=["安警官"], actors=["张译"], importance="main"),
            CharacterEntry(name="陈书婷", aliases=["大嫂"], actors=["高叶"], importance="key_supporting"),
        ],
    )


def _samples() -> list:
    return [
        make_sample("kuangbiao_0001", target_character="高启强", character_aliases=["强哥"]),
        make_sample("kuangbiao_0002", target_character="高启强",
                    character_importance="supporting", character_aliases=[]),
        make_sample("kuangbiao_0003", target_character="安欣", character_aliases=[],
                    visual_difficulty="side_face"),
        make_sample("kuangbiao_0004", target_character="陈书婷", character_aliases=["大嫂"],
                    character_importance="key_supporting",
                    visual_difficulty="blurry", is_multi_person=True),
    ]


def _record(run_id, model, exp, sample_id, *, ok=True, raw_text="", error_type=None):
    return {
        "run_id": run_id, "model_id": model, "experiment": exp,
        "sample_id": sample_id, "work_slug": "kuangbiao", "attempt": 1,
        "ok": int(ok), "http_status": 200 if ok else 500, "latency_ms": 100,
        "error_type": error_type, "error_message": "boom" if not ok else None,
        "served_model": f"{model}#1.0", "usage": "{}",
        "raw_json": json.dumps({
            "sample_id": sample_id, "experiment": exp, "model_id": model,
            "request": {"user": "prompt"},
            "response": {"ok": ok, "raw_text": raw_text, "served_model": f"{model}#1.0",
                         "error_type": error_type},
        }, ensure_ascii=False),
        "raw_file": f"{model}_{exp}.jsonl", "created_at": "2026-01-01T00:00:00+00:00",
    }


def _build_run(tmp_path):
    store = RunStore(tmp_path / "runs.db")
    run_id = "run_test01"
    store.create_run(run_id, experiments=["A", "B"], models=["m1", "m2"],
                     prompt_version="v1", dataset_hash="x", dataset_path="y",
                     fairness_meta={})
    def ok_json(name):
        return json.dumps({"character_name": name, "is_uncertain": False, "reason": "测试"}, ensure_ascii=False)

    for exp, entries in {
        "A": [
            ("kuangbiao_0001", dict(ok=True, raw_text=ok_json("高启强"))),
            ("kuangbiao_0002", dict(ok=True, raw_text=ok_json("张颂文"))),
            ("kuangbiao_0003", dict(ok=True, raw_text=ok_json(None))),
            ("kuangbiao_0004", dict(ok=False, error_type="server_error")),
        ],
        "B": [
            ("kuangbiao_0001", dict(ok=True, raw_text=ok_json("高启强"))),
            ("kuangbiao_0002", dict(ok=True, raw_text=ok_json("高启盛"))),
            ("kuongbiao_0003".replace("kuong", "kuang"), dict(ok=True, raw_text="很抱歉，我无法识别人脸。")),
            ("kuangbiao_0004", dict(ok=True, raw_text=ok_json("大嫂"))),
        ],
    }.items():
        for sample_id, kwargs in entries:
            store.upsert_record(_record(run_id, "m1", exp, sample_id, **kwargs))
    return store, run_id


def _score(tmp_path, **kwargs):
    store, run_id = _build_run(tmp_path)
    matcher = CharacterMatcher({"kuangbiao": _roster()})
    result = score_run(store, run_id, _samples(), matcher, {"scoring": {"refusal_keywords": []}}, **kwargs)
    return store, run_id, result


def test_outcomes_and_dual_metrics(tmp_path):
    store, run_id, result = _score(tmp_path)
    by_key = {(r.model_id, r.experiment, r.sample_id): r.outcome for r in result.records}
    assert by_key[("m1", "A", "kuangbiao_0001")] == "correct"
    assert by_key[("m1", "A", "kuangbiao_0002")] == "actor_name"
    assert by_key[("m1", "A", "kuangbiao_0003")] == "uncertain"
    assert by_key[("m1", "A", "kuangbiao_0004")] == "api_failure"
    assert by_key[("m1", "B", "kuangbiao_0002")] == "wrong_name"
    assert by_key[("m1", "B", "kuangbiao_0003")] == "refused"
    assert by_key[("m1", "B", "kuangbiao_0004")] == "correct"  # 名单别名「大嫂」
    assert by_key[("m2", "A", "kuangbiao_0001")] == "not_run"

    a = result.metrics[("m1", "A")]
    assert a.planned == 4
    assert a.counts["correct"] == 1
    assert math.isclose(a.accuracy_valid, 1 / 3)
    assert math.isclose(a.accuracy_e2e, 1 / 4)
    assert math.isclose(a.api_failure_rate, 1 / 4)

    b = result.metrics[("m1", "B")]
    assert b.counts["correct"] == 2 and b.counts["refused"] == 1
    assert math.isclose(b.accuracy_valid, 0.5)

    m2 = result.metrics[("m2", "A")]
    assert m2.counts["not_run"] == 4 and m2.accuracy_valid == 0.0


def test_slices_and_per_character(tmp_path):
    _, _, result = _score(tmp_path)
    main_a = result.by_importance[("m1", "A")]["main"]
    assert main_a.planned == 2  # 0001/0003
    hard_a = result.by_difficulty[("m1", "A")]["blurry"]
    assert hard_a.planned == 1 and hard_a.accuracy_valid == 0.0  # api_failure
    multi_b = result.by_multi_person[("m1", "B")]["多人同框"]
    assert multi_b.counts["correct"] == 1
    work_a = result.by_work[("m1", "A")]["kuangbiao"]
    assert work_a.planned == 4

    pc = result.per_character[("m1", "A")]["characters"]
    assert pc["kuangbiao::高启强"]["n"] == 2
    assert pc["kuangbiao::高启强"]["correct"] == 1
    macro = result.per_character[("m1", "A")]["macro_accuracy"]
    assert math.isclose(macro, (0.5 + 0.0 + 0.0) / 3)


def test_ab_paired_test(tmp_path):
    _, _, result = _score(tmp_path)
    test = result.ab_tests["m1"]
    # 配对样本：0001(T,T) 0002(F,F) 0003(F,F)，无不一致对 → p=1
    assert test["n_discordant"] == 0
    assert test["p_value"] == 1.0
    assert math.isclose(test["accuracy_B_on_paired"], 1 / 3)


def test_override_changes_outcome(tmp_path):
    store, run_id, result = _score(tmp_path)
    before = result.metrics[("m1", "A")].counts["correct"]
    store.set_override(run_id, "m1", "A", "kuangbiao_0002", "correct", "实为别名")
    result2 = score_run(
        store, run_id, _samples(),
        CharacterMatcher({"kuangbiao": _roster()}),
        {"scoring": {"refusal_keywords": []}},
    )
    assert result2.metrics[("m1", "A")].counts["correct"] == before + 1
    rec = next(
        r for r in result2.records
        if (r.model_id, r.experiment, r.sample_id) == ("m1", "A", "kuangbiao_0002")
    )
    assert rec.outcome == "correct" and rec.overridden


def test_wilson_ci():
    assert wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = wilson_ci(5, 10)
    assert 0.0 < lo < 0.5 < hi < 1.0
    lo2, hi2 = wilson_ci(100, 100)
    assert lo2 > 0.9 and hi2 == 1.0
    # k>n 截断
    lo3, hi3 = wilson_ci(11, 10)
    assert 0 < lo3 < 1 and math.isclose(hi3, 1.0)


def test_mcnemar_exact():
    # 无不一致对
    r = mcnemar([True, False], [True, False])
    assert r["p_value"] == 1.0 and r["n_discordant"] == 0
    # b=2, c=0 → 精确二项 p=0.5
    r2 = mcnemar([False, False, True, True], [True, True, True, True])
    assert r2["b"] == 2 and r2["c"] == 0
    assert math.isclose(r2["p_value"], 0.5)
    # 完全对称不一致 b=c=1 → p=1
    r3 = mcnemar([True, False], [False, True])
    assert math.isclose(r3["p_value"], 1.0)


def test_save_score_result(tmp_path):
    _, _, result = _score(tmp_path)
    files = save_score_result(result, tmp_path / "scores")
    assert files["summary"].exists()
    assert files["metrics"].exists()
    assert files["detail"].exists()
    metrics_csv = files["metrics"].read_text(encoding="utf-8-sig")
    assert "m1,A,4,3,1" in metrics_csv.replace("\r", "") or ",m1,A," in metrics_csv
    detail_csv = files["detail"].read_text(encoding="utf-8-sig")
    assert "actor_name" in detail_csv and "高启强" in detail_csv
    payload = json.loads(files["summary"].read_text(encoding="utf-8"))
    assert payload["run_id"] == "run_test01"
    assert "m1|A" in payload["metrics"]
