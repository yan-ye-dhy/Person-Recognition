"""作品归属核验（screening）测试：证据规则、视觉启发式、编排与隔离。"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from tests.conftest import make_work
from vcrb.config import load_settings
from vcrb.dataset.store import AnnotationStore, CandidateItem
from vcrb.screening.rules import classify_sample
from vcrb.screening.service import report_screening, run_screening
from vcrb.screening.visual import analyze_image

CFG = load_settings().get("screening", {})


# ---------------------------------------------------------------- 规则

def test_rule_verified_needs_domain_and_token():
    work = make_work("langyabang")  # trusted: iqiyi/ent.sina/doubanio
    v = classify_sample(
        work,
        source_url="https://img9.doubanio.com/view/photo/l/abc.jpg",
        page_title="《琅琊榜》高清剧照_豆瓣电影",
        page_url="https://movie.douban.com/subject/25754848/",
    )
    assert v.status == "verified"
    assert v.evidence["trusted_domain"] is True
    assert v.evidence["matched_token"]


def test_rule_token_only_is_needs_review():
    """仅模糊关键词命中（作品标识但域不可信）不判 verified。"""
    work = make_work("kuangbiao")
    v = classify_sample(
        work,
        source_url="https://some.cdn.example.net/x.jpg",
        page_title="狂飙相关图片",
    )
    assert v.status == "needs_review"


def test_rule_trusted_domain_only_is_needs_review():
    work = make_work("kuangbiao")
    v = classify_sample(
        work,
        source_url="https://www.iqiyi.com/o.jpg",
        page_title="十年十佳国剧盘点",  # 可信域但无作品标识
    )
    assert v.status == "needs_review"


def test_rule_vague_keyword_not_token():
    """“功夫”“女足”这类模糊词不是有效证据：非标识词不给 verified。"""
    work = make_work("gongfunvzu")  # 标识 = 功夫女足/gongfunvzu/...
    v = classify_sample(
        work,
        source_url="https://img9.doubanio.com/kungfu.jpg",
        page_title="功夫足球 经典镜头",  # 只有“功夫”，不含“功夫女足”
    )
    assert v.status == "needs_review"


def test_rule_blocklist_rejected():
    work = make_work("jianlai")
    v = classify_sample(
        work,
        source_url="https://i.pximg.net/img-master/img/2026/a.png",
        page_title="剑来 同人图",
        blocklist_domains=list(CFG.get("blocklist_domains", [])),
    )
    assert v.status == "rejected"


def test_rule_animation_grayscale_rejected_but_liveaction_review():
    flags = ["grayscale_image"]
    anime = classify_sample(make_work("jianlai"), "https://x.cn/a.jpg", "剑来", visual_flags=flags)
    live = classify_sample(make_work("kuangbiao"), "https://x.cn/a.jpg", "狂飙", visual_flags=flags)
    assert anime.status == "rejected"
    assert live.status == "needs_review"


def test_rule_collage_needs_review_not_rejected():
    """拼贴启发式实测有误报（水墨云雾/贯穿字幕条也会触发）→ 不自动排除，
    也不允许自动 verified，一律交人工复核。"""
    # 无证据 + 拼贴 → needs_review
    v1 = classify_sample(
        make_work("gongfunvzu"), "https://x.cn/a.jpg", "功夫女足",
        visual_flags=["collage_grid"],
    )
    assert v1.status == "needs_review"
    # 可信域 + 作品标识 + 拼贴 → 仍 needs_review（多格组图不适合做单帧素材）
    v2 = classify_sample(
        make_work("gongfunvzu"),
        source_url="https://img9.doubanio.com/x.jpg",
        page_title="《功夫女足》官方组图",
        visual_flags=["collage_grid"],
    )
    assert v2.status == "needs_review"
    assert any("拼贴" in r for r in v2.reasons)


def test_rule_visual_never_verifies_alone():
    """纯视觉信号绝不产生 verified（即使看起来“正常”）。"""
    v = classify_sample(
        make_work("jianlai"), "https://random.net/a.jpg", "",
        visual_flags=[],
    )
    assert v.status == "needs_review"


def test_rule_poster_and_odd_aspect_go_to_review():
    v1 = classify_sample(
        make_work("gongfunvzu"), "https://x.cn/a.jpg", "功夫女足",
        visual_flags=["poster_layout"],
    )
    v2 = classify_sample(
        make_work("gongfunvzu"), "https://x.cn/a.jpg", "功夫女足",
        visual_flags=["odd_aspect"],
    )
    assert v1.status == "needs_review" and v2.status == "needs_review"


def test_rule_unreadable_pending():
    v = classify_sample(
        make_work("kuangbiao"), "https://x.cn/a.jpg", "", visual_flags=["unreadable"]
    )
    assert v.status == "pending"


def test_rule_subdomain_and_suffix_safety():
    work = make_work("kuangbiao")
    ok = classify_sample(
        work, source_url="https://pic.ent.sina.com.cn/a.jpg", page_title="狂飙 剧照"
    )
    assert ok.status == "verified"
    # 前缀相似但不属于可信域
    bad = classify_sample(
        work, source_url="https://ent.sina.com.cn.evil.net/a.jpg", page_title="狂飙 剧照"
    )
    assert bad.status == "needs_review"


# ---------------------------------------------------------------- 视觉

def _img(path, size, mode_fn):
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    mode_fn(arr)
    Image.fromarray(arr).save(path, format="JPEG")
    return path


def test_visual_grayscale_detection(tmp_path):
    def gray(a):
        a[:] = np.random.RandomState(0).randint(40, 200, a.shape[:2])[..., None]

    def color(a):
        rs = np.random.RandomState(1)
        a[..., 0] = rs.randint(0, 255, a.shape[:2])
        a[..., 1] = rs.randint(0, 255, a.shape[:2])
        a[..., 2] = rs.randint(0, 255, a.shape[:2])

    p1 = _img(tmp_path / "gray.jpg", (640, 360), gray)
    p2 = _img(tmp_path / "color.jpg", (640, 360), color)
    assert "grayscale_image" in analyze_image(p1, CFG).flags
    assert "grayscale_image" not in analyze_image(p2, CFG).flags


def _color_noise(seed):
    def fn(a):
        rs = np.random.RandomState(seed)
        a[..., 0] = rs.randint(20, 240, a.shape[:2])
        a[..., 1] = rs.randint(20, 240, a.shape[:2])
        a[..., 2] = rs.randint(20, 240, a.shape[:2])
    return fn


def test_visual_poster_and_aspect(tmp_path):
    p1 = _img(tmp_path / "poster.jpg", (400, 700), _color_noise(10))
    p2 = _img(tmp_path / "wide.jpg", (640, 360), _color_noise(11))
    assert "poster_layout" in analyze_image(p1, CFG).flags
    assert analyze_image(p2, CFG).flags == []


def test_visual_collage_grid(tmp_path):
    """3 联拼贴（两竖分割线）→ collage_grid；普通噪声图不误报。"""
    def collage(a):
        # 平滑色块格子 + 硬分割线（模拟宣传组图）
        h, w = a.shape[:2]
        cell_colors = [(180, 60, 60), (60, 180, 90), (70, 90, 200)]
        for i, c in enumerate(cell_colors):
            x0, x1 = i * w // 3, (i + 1) * w // 3
            a[:, x0:x1] = c
            rs = np.random.RandomState(i)
            noise = rs.randint(-4, 5, a[:, x0:x1].shape)
            a[:, x0:x1] = np.clip(
                a[:, x0:x1].astype(np.int16) + noise, 0, 255
            ).astype(np.uint8)
        for x in range(w // 3, w, w // 3):   # 两条竖分割线
            a[:, max(0, x - 2):x + 2] = 0

    p1 = _img(tmp_path / "collage.jpg", (640, 360), collage)
    p2 = _img(tmp_path / "plain.jpg", (640, 360), _color_noise(3))
    assert "collage_grid" in analyze_image(p1, CFG).flags
    assert "collage_grid" not in analyze_image(p2, CFG).flags


def test_visual_unreadable(tmp_path):
    p = tmp_path / "bad.jpg"
    p.write_bytes(b"not an image")
    assert "unreadable" in analyze_image(p, CFG).flags


# ---------------------------------------------------------------- 存储与编排

def _setup_work_store(tmp_path, work, n=3):
    cand = tmp_path / "candidates" / work.slug
    cand.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore(tmp_path / "db.sqlite")
    items = []
    for i in range(1, n + 1):
        name = f"cand_{i:04d}.jpg"
        img = Image.new("RGB", (640, 360), (60 + i * 40, 80, 140))
        img.save(cand / name, format="JPEG")
        items.append(CandidateItem(work.slug, name, 640, 360,
                                   source_url=f"https://www.iqiyi.com/img{i}.jpg"))
    store.register_candidates(work.slug, items, cand)
    (cand / "manifest.jsonl").write_text(
        "\n".join(
            json.dumps({"file": it.filename, "source_url": it.source_url}, ensure_ascii=False)
            for it in items
        ),
        encoding="utf-8",
    )
    return store, cand


def test_run_screening_and_audit_log(tmp_path):
    work = make_work("kuangbiao")
    store, cand = _setup_work_store(tmp_path, work)
    rep = run_screening(store, work, load_settings(), candidates_dir=cand)
    assert rep.total == 3 and rep.decided == 3
    # iqiyi 可信域但 URL 无“狂飙”标识 → 全部 needs_review（不自动放行）
    assert rep.counts.get("needs_review") == 3
    row = store.samples(work.slug)[0]
    assert row["attribution_status"] == "needs_review"
    assert row["attribution_reasons"]
    log = store.attribution_log(row["pk"])
    assert len(log) == 1 and log[0]["actor"] == "auto:rules"


def test_run_screening_keeps_user_decisions(tmp_path):
    work = make_work("kuangbiao")
    store, cand = _setup_work_store(tmp_path, work)
    row = store.samples(work.slug)[0]
    store.set_attribution([row["pk"]], "verified", ["人工确认"], actor="user")
    rep = run_screening(store, work, load_settings(), candidates_dir=cand)
    assert rep.kept_user == 1 and rep.decided == 2
    assert store.samples(work.slug)[0]["attribution_status"] == "verified"


def test_screening_report_numbers(tmp_path):
    work = make_work("kuangbiao")
    store, cand = _setup_work_store(tmp_path, work, n=4)
    rows = store.samples(work.slug)
    # 1 张人工核实归属+标注 verified → 可入评测；1 张自动排除；其余待复核
    store.set_attribution([rows[0]["pk"]], "verified", ["人工"], actor="user")
    store.update_annotation(
        rows[0]["pk"],
        {"status": "verified", "target_character": "高启强",
         "character_importance": "main", "visual_difficulty": "normal",
         "is_multi_person": 0},
    )
    store.set_attribution([rows[1]["pk"]], "rejected", ["测试排除"], actor="auto:rules")
    result = report_screening(store, [work], target_per_work=100)[0]
    assert result["candidates"] == 4
    assert result["attribution_verified"] == 1
    assert result["auto_rejected"] == 1
    assert result["needs_human"] == 2  # 剩余 needs_review/pending
    assert result["benchmark_ready"] == 1
    assert result["gap_to_100"] == 99


def test_set_attribution_validates_status(tmp_path):
    store = AnnotationStore(tmp_path / "db.sqlite")
    store.register_candidates("kuangbiao", [CandidateItem("kuangbiao", "a.jpg", 1, 1)], None)
    pk = store.samples("kuangbiao")[0]["pk"]
    with pytest.raises(ValueError):
        store.set_attribution([pk], "maybe", ["非法状态"])


# ---------------------------------------------------------------- purge

def _purge_fixture(tmp_path, work):
    """候选目录 + 原始目录 + 双清单 + 3 张图（1 verified / 1 rejected / 1 pending）。"""
    import shutil as _shutil

    cand = tmp_path / "candidates" / work.slug
    raw = tmp_path / "raw" / work.slug
    cand.mkdir(parents=True)
    raw.mkdir(parents=True)
    store = AnnotationStore(tmp_path / "db.sqlite")
    for i, name in enumerate(("raw_1.jpg", "raw_2.jpg", "raw_3.jpg"), start=1):
        Image.new("RGB", (64, 48), (i * 60, 30, 90)).save(raw / name)
        (raw / "manifest.jsonl").open("a", encoding="utf-8").write(
            json.dumps({"file": name, "source_url": f"https://x/{name}"}) + "\n"
        )
        cand_name = f"cand_{i:04d}.jpg"
        _shutil.copyfile(raw / name, cand / cand_name)
        (cand / "manifest.jsonl").open("a", encoding="utf-8").write(
            json.dumps({"file": cand_name, "original_file": name}) + "\n"
        )
    store.sync_candidates(work.slug, cand)
    rows = store.samples(work.slug)
    store.set_attribution([rows[0]["pk"]], "verified", ["可信证据"], actor="auto:rules")
    store.set_attribution([rows[1]["pk"]], "rejected", ["其他作品"], actor="auto:rules")
    return store, cand, raw


def test_purge_rejected_mode(tmp_path):
    from vcrb.screening.service import purge_candidates

    work = make_work("jianlai")
    store, cand, raw = _purge_fixture(tmp_path, work)
    rep = purge_candidates(store, work, "rejected", candidates_dir=cand, raw_dir=raw,
                           apply=True, reason="黑白漫画")
    assert rep.planned == 1
    assert rep.deleted_candidates == 1 and rep.deleted_raws == 1
    # 文件与原始文件都删除；verified/pending 保留
    assert not (cand / "cand_0002.jpg").exists()
    assert not (raw / "raw_2.jpg").exists()
    assert (cand / "cand_0001.jpg").exists() and (cand / "cand_0003.jpg").exists()
    # 库行删除、留痕完整
    assert len(store.samples(work.slug)) == 2
    log = store.purge_log(work.slug)
    assert len(log) == 1
    assert log[0]["reason"] == "黑白漫画" and log[0]["original_file"] == "raw_2.jpg"
    # 两份 manifest 同步重写
    assert "cand_0002.jpg" not in (cand / "manifest.jsonl").read_text(encoding="utf-8")
    assert "raw_2.jpg" not in (raw / "manifest.jsonl").read_text(encoding="utf-8")
    assert "cand_0001.jpg" in (cand / "manifest.jsonl").read_text(encoding="utf-8")


def test_purge_dry_run_deletes_nothing(tmp_path):
    from vcrb.screening.service import purge_candidates

    work = make_work("jianlai")
    store, cand, raw = _purge_fixture(tmp_path, work)
    rep = purge_candidates(store, work, "rejected", candidates_dir=cand, raw_dir=raw,
                           apply=False)
    assert rep.planned == 1 and rep.deleted_candidates == 0
    assert (cand / "cand_0002.jpg").exists()
    assert len(store.samples(work.slug)) == 3
    assert store.purge_log(work.slug) == []


def test_purge_all_protects_verified(tmp_path):
    from vcrb.screening.service import purge_candidates

    work = make_work("jianlai")
    store, cand, raw = _purge_fixture(tmp_path, work)
    rep = purge_candidates(store, work, "all", candidates_dir=cand, raw_dir=raw, apply=True)
    # verified 受保护，rejected + pending 删除
    assert rep.planned == 2
    assert (cand / "cand_0001.jpg").exists()
    assert not (cand / "cand_0002.jpg").exists() and not (cand / "cand_0003.jpg").exists()
    remaining = store.samples(work.slug)
    assert len(remaining) == 1 and remaining[0]["attribution_status"] == "verified"


def test_purge_all_force_removes_verified(tmp_path):
    """--force 解除 verified 保护（用户明确要求整池清空时）。"""
    from vcrb.screening.service import purge_candidates

    work = make_work("jianlai")
    store, cand, raw = _purge_fixture(tmp_path, work)
    rep = purge_candidates(store, work, "all", candidates_dir=cand, raw_dir=raw,
                           apply=True, force=True, reason="整池清空重采")
    assert rep.planned == 3
    assert not any(cand.glob("cand_*.jpg"))
    assert store.samples(work.slug) == []
    log = store.purge_log(work.slug)
    assert len(log) == 3 and all(l["reason"] == "整池清空重采" for l in log)


def test_purged_not_resurrected_by_clean_sync(tmp_path):
    """删除后 raw/candidates 都不再含该图，重跑 sync 不会复活。"""
    from vcrb.screening.service import purge_candidates

    work = make_work("jianlai")
    store, cand, raw = _purge_fixture(tmp_path, work)
    purge_candidates(store, work, "rejected", candidates_dir=cand, raw_dir=raw, apply=True)
    again = store.sync_candidates(work.slug, cand)
    assert again["registered"] == 0
    assert len(store.samples(work.slug)) == 2
    assert store.purge_log(work.slug)[0]["filename"] == "cand_0002.jpg"


def test_rejected_not_reregistered_after_resync(tmp_path):
    """排除的素材在重跑 sync 后不会复活（md5 身份保留原行与归属状态）。"""
    work = make_work("kuangbiao")
    store, cand = _setup_work_store(tmp_path, work, n=1)
    pk = store.samples(work.slug)[0]["pk"]
    store.set_attribution([pk], "rejected", ["其他作品"], actor="user")
    again = store.sync_candidates(work.slug, cand)
    assert again["registered"] == 0
    assert store.samples(work.slug)[0]["attribution_status"] == "rejected"


# ---------------------------------------------------------------- 隔离红线

def test_autoannot_isolation_from_evaluated_providers():
    """筛查组件不得依赖被评测厂商的模型/SDK（防评测污染）。"""
    import subprocess
    import sys

    code = (
        "import vcrb.screening, vcrb.screening.rules, vcrb.screening.visual, "
        "vcrb.screening.service, sys;"
        "mods = [m for m in sys.modules if m.startswith('vcrb.providers') "
        "or m.startswith('openai')];"
        "assert not mods, mods"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_runner_does_not_import_screening():
    """评测链路不引用筛查自动判定（筛查仅用于建库）。"""
    import subprocess
    import sys

    code = (
        "import vcrb.runner.executor, vcrb.evaluation.scoring, vcrb.reports.charts, sys; "
        "assert not [m for m in sys.modules if m.startswith('vcrb.screening')]"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
