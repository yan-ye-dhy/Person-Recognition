"""本地人工标注页面（Streamlit）。

启动：vcrb annotate  （等价 streamlit run vcrb/annotation/app.py）

四个页面：
1. 标注：浏览候选 → 框选目标人物 → 选择角色/三维度标签 → 状态保存
2. 角色名单：维护各作品角色候选（别名/演员/重要度）
3. 进度总览：四作品 × 状态计数 + 五组采样配额达成
4. 人工复核：查看评分待复核项并改判（写入 review_overrides）

坐标约定：target_bbox 为归一化 [x1,y1,x2,y2]，画布像素坐标按显示宽度换算。
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

# 以包形式运行时保证 vcrb 可导入
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vcrb.annotation.rosters import CharacterEntry, load_roster, save_roster  # noqa: E402
from vcrb.config import load_settings, load_works  # noqa: E402
from vcrb.dataset.schema import DIFFICULTY_LEVELS, IMPORTANCE_LEVELS, SAMPLING_GROUPS  # noqa: E402
from vcrb.dataset.store import AnnotationStore  # noqa: E402
from vcrb.paths import CANDIDATES_DIR, DATASET_JSON  # noqa: E402
from vcrb.runner.store import RunStore  # noqa: E402

try:
    from streamlit_drawable_canvas import st_canvas
    HAS_CANVAS = True
except ImportError:  # 未安装组件时退化为数值输入
    HAS_CANVAS = False

IMPORTANCE_LABELS = {
    "main": "主角",
    "key_supporting": "重要配角",
    "supporting": "普通配角",
}
DIFFICULTY_LABELS = {
    "normal": "正常清晰",
    "side_face": "侧脸",
    "occluded": "遮挡",
    "blurry": "模糊",
    "costume_change": "特殊造型",
}
GROUP_LABELS = {
    "main_character": "主角清晰正脸",
    "key_supporting_character": "重要配角",
    "supporting_character": "普通配角",
    "hard_case": "侧脸/遮挡/模糊/造型",
    "multi_person": "多人同框",
}
STATUS_LABELS = {
    "pending": "待标注",
    "verified": "已确认（有效）",
    "invalid": "无效图片",
    "uncertain": "待确认（有争议）",
}
TAG_OPTIONS = [
    "front_face", "side_face", "occluded", "blurry", "costume_change",
    "multiple_people", "night", "action", "close_up", "wide_shot", "indoor", "outdoor",
]

st.set_page_config(page_title="影视角色标注", layout="wide", page_icon="🎬")


@st.cache_resource
def get_store() -> AnnotationStore:
    return AnnotationStore()


@st.cache_resource
def get_run_store() -> RunStore:
    return RunStore()


@st.cache_data
def get_works():
    return load_works()


@st.cache_data
def get_settings():
    return load_settings()


# ================================================================ 标注页

def page_annotate() -> None:
    works = get_works()
    work = st.sidebar.selectbox(
        "选择作品", works, format_func=lambda w: f"{w.name}（{w.slug}）"
    )
    roster = load_roster(work)
    store = get_store()

    rows = store.samples(work.slug)
    if not rows:
        st.warning(
            f"《{work.name}》暂无候选图片。请先执行：vcrb clean --work {work.slug}"
        )
        return
    status_filter = st.sidebar.selectbox(
        "按状态筛选",
        ["全部", "pending", "verified", "invalid", "uncertain"],
        format_func=lambda s: "全部" if s == "全部" else STATUS_LABELS.get(s, s),
    )
    if status_filter != "全部":
        rows = [r for r in rows if r["status"] == status_filter]
    if not rows:
        st.info("该状态下没有样本。")
        return

    st.sidebar.metric("当前筛选样本数", len(rows))
    index = st.sidebar.number_input(
        "跳转到第几张", min_value=1, max_value=len(rows), value=1
    ) - 1
    row = rows[index]
    st.title(f"《{work.name}》 · 第 {index + 1}/{len(rows)} 张 · {row['filename']}")

    left, right = st.columns([3, 2])

    with left:
        img_path = CANDIDATES_DIR / work.slug / row["filename"]
        if not img_path.exists():
            st.error(f"图片缺失：{img_path}")
            return
        orig_w = row.get("width") or 1
        orig_h = row.get("height") or 1
        display_width = 640
        display_height = max(1, int(orig_h * display_width / orig_w))
        existing_bbox = row.get("target_bbox")

        # 已有 bbox 预渲染到背景图（红色实线框），便于核对/微调
        background = Image.open(img_path).convert("RGB")
        if existing_bbox:
            x1, y1, x2, y2 = existing_bbox
            draw = ImageDraw.Draw(background)
            draw.rectangle(
                [x1 * orig_w, y1 * orig_h, x2 * orig_w, y2 * orig_h],
                outline=(255, 0, 0), width=max(3, orig_w // 200),
            )
        st.caption("红框 = 已保存的目标框；重新框选并保存即覆盖。")

        if HAS_CANVAS:
            canvas_result = st_canvas(
                fill_color="rgba(255,0,0,0.15)",
                stroke_width=3,
                stroke_color="#ff0000",
                background_image=background,
                drawing_mode="rect",
                height=display_height,
                width=display_width,
                key=f"canvas_{row['pk']}",
                display_toolbar=False,
            )
            new_bbox = None
            source = "未修改（沿用已保存框）" if existing_bbox else "未绘制"
            if canvas_result.json_data and canvas_result.json_data.get("objects"):
                obj = canvas_result.json_data["objects"][-1]
                left_p = obj["left"] / display_width
                top_p = obj["top"] / display_height
                w_p = obj["width"] / display_width
                h_p = obj["height"] / display_height
                new_bbox = [
                    round(max(0.0, min(1.0, left_p)), 4),
                    round(max(0.0, min(1.0, top_p)), 4),
                    round(max(0.0, min(1.0, left_p + w_p)), 4),
                    round(max(0.0, min(1.0, top_p + h_p)), 4),
                ]
                source = "canvas"
        else:
            st.info("未安装 streamlit-drawable-canvas，使用数值输入 bbox（x1,y1,x2,y2，0~1）")
            default = existing_bbox or [0.1, 0.1, 0.9, 0.9]
            c1, c2, c3, c4 = st.columns(4)
            new_bbox = [
                c1.number_input("x1", 0.0, 1.0, float(default[0]), 0.01),
                c2.number_input("y1", 0.0, 1.0, float(default[1]), 0.01),
                c3.number_input("x2", 0.0, 1.0, float(default[2]), 0.01),
                c4.number_input("y2", 0.0, 1.0, float(default[3]), 0.01),
            ]
            source = "manual"
        st.caption(
            f"bbox={new_bbox if new_bbox else existing_bbox}（{source}）；"
            f"原图 {orig_w}×{orig_h}；素材类型 {row.get('source_type', 'unknown')}"
        )

    with right:
        with st.form("annotation_form", clear_on_submit=False):
            names = roster.names()
            current = row.get("target_character") or ""
            char_name = st.selectbox(
                "目标角色",
                ["（选择角色）"] + names + ["（其他/新角色）"],
                index=(
                    (["（选择角色）"] + names + ["（其他/新角色）"]).index(current)
                    if current in names
                    else 0
                ),
            )
            if char_name == "（其他/新角色）":
                char_name = st.text_input("输入角色名（将加入名单）")
            elif char_name == "（选择角色）":
                char_name = ""

            c1, c2 = st.columns(2)
            importance = c1.selectbox(
                "人物重要度",
                list(IMPORTANCE_LEVELS),
                index=list(IMPORTANCE_LEVELS).index(
                    row.get("character_importance") or "supporting"
                ),
                format_func=lambda v: IMPORTANCE_LABELS[v],
            )
            difficulty = c2.selectbox(
                "视觉难度",
                list(DIFFICULTY_LEVELS),
                index=list(DIFFICULTY_LEVELS).index(row.get("visual_difficulty") or "normal"),
                format_func=lambda v: DIFFICULTY_LABELS[v],
            )
            multi = st.checkbox(
                "多人同框", value=bool(row.get("is_multi_person")),
                help="与重要度/难度相互独立（调整 2）",
            )
            aliases = st.text_input(
                "本图角色别名（逗号分隔，可选）",
                value="、".join(row.get("character_aliases") or []),
            )
            tags = st.multiselect(
                "辅助标签", TAG_OPTIONS, default=list(row.get("tags") or [])
            )
            status = st.radio(
                "审核状态",
                ["verified", "invalid", "uncertain", "pending"],
                index=["verified", "invalid", "uncertain", "pending"].index(row["status"]),
                format_func=lambda s: STATUS_LABELS[s],
                horizontal=True,
            )
            note = st.text_input("备注", value=row.get("note") or "")
            submitted = st.form_submit_button("保存标注", type="primary", use_container_width=True)

        if submitted:
            problems = []
            final_bbox = new_bbox if new_bbox is not None else existing_bbox
            if status == "verified" and not char_name:
                problems.append("已确认样本必须选择目标角色")
            if status == "verified" and multi and not final_bbox:
                problems.append("多人同框样本必须绘制目标框")
            if problems:
                for p in problems:
                    st.warning(p)
            fields = {
                "status": status,
                "target_character": char_name,
                "character_importance": importance,
                "visual_difficulty": difficulty,
                "is_multi_person": int(multi),
                "character_aliases": [a.strip() for a in aliases.replace("，", ",").split(",") if a.strip()],
                "tags": tags,
                "target_bbox": final_bbox,
                "note": note,
            }
            store.update_annotation(row["pk"], fields)
            # 新角色自动入名单（needs_review 标记，等名单页确认）
            if char_name and roster.entry(char_name) is None:
                roster.add(
                    CharacterEntry(name=char_name, importance=importance, needs_review=True)
                )
                save_roster(roster)
            st.success("已保存。请用左侧序号或筛选切换下一张。")
            st.rerun()


# ================================================================ 名单页

def page_roster() -> None:
    works = get_works()
    work = st.sidebar.selectbox("选择作品", works, format_func=lambda w: w.name)
    roster = load_roster(work)
    st.title(f"角色名单 · 《{work.name}》")
    st.caption(roster.source_note or "")
    for i, char in enumerate(roster.characters):
        with st.expander(
            f"{'⚠️' if char.needs_review else '✅'} {char.name}（{IMPORTANCE_LABELS[char.importance]}）"
        ):
            c1, c2 = st.columns(2)
            aliases = c1.text_input("别名（逗号分隔）", value=",".join(char.aliases), key=f"al{i}")
            actors = c2.text_input("演员（逗号分隔）", value=",".join(char.actors), key=f"ac{i}")
            m1, m2 = st.columns(2)
            importance = m1.selectbox(
                "重要度", list(IMPORTANCE_LEVELS), index=list(IMPORTANCE_LEVELS).index(char.importance),
                format_func=lambda v: IMPORTANCE_LABELS[v], key=f"im{i}",
            )
            needs_review = m2.checkbox("待人工确认", value=char.needs_review, key=f"nr{i}")
            if st.button("保存该角色", key=f"sv{i}"):
                char.aliases = [a.strip() for a in aliases.split(",") if a.strip()]
                char.actors = [a.strip() for a in actors.split(",") if a.strip()]
                char.importance = importance
                char.needs_review = needs_review
                save_roster(roster)
                st.success(f"已保存 {char.name}")
                st.rerun()
    with st.form("add_character"):
        st.subheader("新增角色")
        c1, c2 = st.columns(2)
        name = c1.text_input("角色名")
        importance = c2.selectbox("重要度", list(IMPORTANCE_LEVELS), format_func=lambda v: IMPORTANCE_LABELS[v])
        aliases = st.text_input("别名（逗号分隔）")
        actors = st.text_input("演员（逗号分隔）")
        needs_review = st.checkbox("待人工确认（默认勾选，核实后取消）", value=True)
        if st.form_submit_button("添加") and name:
            roster.add(
                CharacterEntry(
                    name=name.strip(),
                    aliases=[a.strip() for a in aliases.split(",") if a.strip()],
                    actors=[a.strip() for a in actors.split(",") if a.strip()],
                    importance=importance,
                    needs_review=needs_review,
                )
            )
            save_roster(roster)
            st.rerun()


# ================================================================ 进度页

def page_progress() -> None:
    st.title("数据准备进度")
    store = get_store()
    settings = get_settings()
    target = int(settings.get("export", {}).get("target_per_group", 20))
    stats = store.stats()
    works = get_works()
    cols = st.columns(len(works))
    for col, work in zip(cols, works):
        bucket = stats.get(work.slug, {})
        verified = bucket.get("verified", 0)
        with col:
            st.metric(work.name, f"{sum(bucket.values())} 候选 / {verified} 已确认")
    st.divider()
    rows = store.samples(status="verified")
    table = {}
    for work in works:
        table[work.name] = {g: 0 for g in SAMPLING_GROUPS}
        table[work.name]["总计"] = 0
    for r in rows:
        multi = bool(r.get("is_multi_person"))
        difficulty = r.get("visual_difficulty") or "normal"
        importance = r.get("character_importance") or "supporting"
        group = (
            "multi_person" if multi
            else ("hard_case" if difficulty != "normal" else {
                "main": "main_character",
                "key_supporting": "key_supporting_character",
                "supporting": "supporting_character",
            }[importance])
        )
        name = next(w.name for w in works if w.slug == r["work_slug"])
        table[name][group] += 1
        table[name]["总计"] += 1
    st.subheader("五组采样配额达成（目标每组 20）")
    header = ["作品"] + [GROUP_LABELS[g] for g in SAMPLING_GROUPS] + ["总计"]
    data = []
    for name, bucket in table.items():
        data.append([name] + [bucket[g] for g in SAMPLING_GROUPS] + [bucket["总计"]])
    data.append(["目标"] + [target] * len(SAMPLING_GROUPS) + [target * len(SAMPLING_GROUPS)])
    st.table([header] + data)
    if DATASET_JSON.exists():
        import json as _json

        payload = _json.loads(DATASET_JSON.read_text(encoding="utf-8"))
        st.success(f"已导出正式集：{payload['total']} 样本（{DATASET_JSON}）")
    else:
        st.info("尚未导出正式集。全部四作品配额接近目标后执行：vcrb export-dataset")


# ================================================================ 复核页

def page_review() -> None:
    st.title("评分人工复核")
    run_store = get_run_store()
    run_id = st.sidebar.text_input("Run ID", value=run_store.latest_run_id() or "")
    if not run_id or not run_store.get_run(run_id):
        st.warning("请输入有效 Run ID（可先执行 vcrb score 生成评分）。")
        return
    from vcrb.evaluation.scoring import score_run
    from vcrb.config import load_settings
    from vcrb.annotation.rosters import load_roster as _lr
    from vcrb.evaluation.matcher import CharacterMatcher
    from vcrb.dataset.schema import load_dataset

    if not DATASET_JSON.exists():
        st.error("缺少 dataset.json，无法评分。")
        return
    samples = load_dataset(DATASET_JSON)
    works = {w.slug: w for w in get_works()}
    rosters = {slug: _lr(w) for slug, w in works.items()}
    matcher = CharacterMatcher(rosters)
    result = score_run(run_store, run_id, samples, matcher, load_settings())
    pending = [r for r in result.records if r.needs_review]
    st.metric("待复核（相似但未精确命中）", len(pending))
    for r in pending[:50]:
        with st.expander(f"{r.model_id} {r.experiment}组 {r.sample_id}：{r.gt_character} ← {r.predicted or '（空）'}"):
            st.caption(r.detail)
            outcome = st.selectbox(
                "改判为",
                ["correct", "wrong_name", "actor_name", "uncertain", "refused"],
                key=f"ov_{r.model_id}_{r.experiment}_{r.sample_id}",
            )
            note = st.text_input("复核说明", key=f"nt_{r.model_id}_{r.experiment}_{r.sample_id}")
            if st.button("保存改判", key=f"bt_{r.model_id}_{r.experiment}_{r.sample_id}"):
                run_store.set_override(
                    r.run_id, r.model_id, r.experiment, r.sample_id, outcome, note
                )
                st.success("已保存，重新执行 vcrb score 后生效。")
                st.rerun()


def main() -> None:
    st.sidebar.title("影视角色标注台")
    page = st.sidebar.radio(
        "页面", ["标注", "角色名单", "进度总览", "人工复核"]
    )
    if page == "标注":
        page_annotate()
    elif page == "角色名单":
        page_roster()
    elif page == "进度总览":
        page_progress()
    else:
        page_review()


if __name__ == "__main__":
    main()
