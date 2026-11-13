"""Local v2 annotation editor: canvas, exact coordinates, identities and audits.

Run: streamlit run vcrb/multi/annotation_app.py -- --dataset data/v2/dataset.json
All writes go to the selected v2 JSON and its audit log; v1 is never opened for writing.
"""

from __future__ import annotations

import argparse
import copy
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image, ImageDraw

from vcrb.annotation.rosters import load_roster
from vcrb.config import load_works
from vcrb.multi.schema import IgnoreRegion, MultiSample, PersonAnnotation, bbox_errors, renumber_persons
from vcrb.multi.store import ConcurrentModificationError, MultiAnnotationStore


def canvas_objects(sample: MultiSample, width: int, height: int, *, ignore: bool = False) -> dict[str, Any]:
    regions = sample.ignore_regions if ignore else sample.persons
    objects = []
    for region in regions:
        x1, y1, x2, y2 = region.bbox
        objects.append({"type": "rect", "version": "4.4.0", "originX": "left", "originY": "top", "left": x1 * width, "top": y1 * height, "width": (x2 - x1) * width, "height": (y2 - y1) * height, "fill": "rgba(255,160,0,0.12)" if ignore else "rgba(255,0,0,0.06)", "stroke": "#ff9900" if ignore else "#ff0000", "strokeWidth": 2, "scaleX": 1, "scaleY": 1, "angle": 0, "lockRotation": True, "lockScalingFlip": True})
    return {"version": "4.4.0", "objects": objects}


def canvas_bbox(obj: dict[str, Any], width: int, height: int) -> list[float]:
    if obj.get("type") != "rect" or abs(float(obj.get("angle", 0))) > 0.01:
        raise ValueError("只支持不旋转的人物矩形框")
    x1, y1 = float(obj["left"]), float(obj["top"])
    x2 = x1 + float(obj["width"]) * float(obj.get("scaleX", 1))
    y2 = y1 + float(obj["height"]) * float(obj.get("scaleY", 1))
    box = [round(max(0.0, min(1.0, v)), 6) for v in (min(x1, x2) / width, min(y1, y2) / height, max(x1, x2) / width, max(y1, y2) / height)]
    errors = bbox_errors(box)
    if errors:
        raise ValueError("; ".join(errors))
    return box


def apply_canvas_objects(sample: MultiSample, objects: list[dict[str, Any]], width: int, height: int, *, ignore: bool = False) -> MultiSample:
    """Canvas order retains metadata; toolbar removal is disabled, delete via named rows."""
    updated = MultiSample.from_dict(sample.to_dict())
    if ignore:
        updated.ignore_regions = [IgnoreRegion(canvas_bbox(obj, width, height), sample.ignore_regions[i].reason if i < len(sample.ignore_regions) else "不可辨识背景人物", copy.deepcopy(sample.ignore_regions[i].evidence) if i < len(sample.ignore_regions) else []) for i, obj in enumerate(objects)]
    else:
        people = []
        for i, obj in enumerate(objects):
            box = canvas_bbox(obj, width, height)
            person = PersonAnnotation.from_dict(sample.persons[i].to_dict()) if i < len(sample.persons) else PersonAnnotation(i + 1, box)
            if person.bbox != box:
                person.review_status = "pending"
            person.bbox = box
            people.append(person)
        updated.persons = renumber_persons(people)
    if updated.to_dict() != sample.to_dict():
        updated.review_status = "draft"
        updated.foreground_complete = False
    return updated


def mark_draft(sample: MultiSample, *, geometry: bool = False) -> None:
    sample.review_status = "draft"
    if geometry:
        sample.foreground_complete = False


def invalidate_ui_edit(state_key: str, annotation_id: str | None = None, geometry: bool = False) -> None:
    """Widget callbacks execute before rendering: prior approvals cannot survive new edits."""
    import streamlit as st

    sample = st.session_state[state_key]
    mark_draft(sample, geometry=geometry)
    version = st.session_state[state_key + ":canvas_version"]
    st.session_state[state_key + f":approve:{version}"] = False
    if geometry:
        st.session_state[state_key + f":complete:{version}"] = False
    if annotation_id:
        person = next(p for p in sample.persons if p.annotation_id == annotation_id)
        person.review_status = "pending"
        st.session_state[state_key + f":edit:{version}:person:" + annotation_id + ":review"] = "pending"


def render_preview(image_path: Path, sample: MultiSample, *, include_ignore: bool = True) -> Image.Image:
    # Same implementation used by A/B, whenever the runner's drawing module is available.
    from vcrb.imageutils import draw_target_boxes

    with tempfile.TemporaryDirectory(prefix="vcrb-v2-preview-") as directory:
        path = Path(directory) / "preview.jpg"
        draw_target_boxes(image_path, path, sample.persons)
        image = Image.open(path).convert("RGB")
    if include_ignore and sample.ignore_regions:
        draw = ImageDraw.Draw(image)
        for region in sample.ignore_regions:
            x1, y1, x2, y2 = region.bbox
            draw.rectangle((x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height), outline="#ff9900", width=max(2, image.width // 350))
    return image


def run_app(dataset_path: Path | None = None) -> None:
    import streamlit as st
    from streamlit.errors import StreamlitAPIException

    st.set_page_config(page_title="多人物 v2 标注", layout="wide", page_icon="🎬")
    if dataset_path is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=Path, default=Path("data/v2/dataset.json"))
        arguments, _ = parser.parse_known_args()
        dataset_path = arguments.dataset
    dataset_path = dataset_path.resolve()
    if not dataset_path.is_file():
        st.error("尚未找到v2数据集，请先执行多人物迁移命令。")
        return
    try:
        store = MultiAnnotationStore(dataset_path)
    except (ValueError, KeyError) as error:
        st.error(f"无法打开v2数据集：{error}")
        return

    st.title("多人物标注 · v2")
    st.caption("红框：本轮清晰可识别的主要角色，通常每图1–3人；虚化背景和残缺旁人不框。角色姓名依据独立证据。")
    st.info("先检查编号与角色对照，等你明确确认后再统一调用真实模型。下方已完成表示助手已代标；旧会话可点「恢复此图已保存标注」加载修订。")
    st.link_button("查看全部160张 A / B / C 输入图", "http://127.0.0.1:8765/results/multi/input_gallery/index.html")
    st.caption("A/B使用相同编号红框；C使用无框原图。C没有红框是实验设计。")
    stats = store.stats()
    metrics = st.columns(5)
    for col, label, value in zip(metrics, ["样本", "已完成", "待复核身份", "身份未知", "前景未补齐"], [stats["samples"], stats["approved_samples"], stats["needs_review_persons"], stats["unknown_persons"], stats["foreground_incomplete"]]):
        col.metric(label, value)
    all_samples = store.samples()
    work_names = {s.work_slug: s.work_name for s in all_samples}
    nav = f"v2nav:{dataset_path}:"
    query_sample = st.query_params.get("sample", "")
    requested = next((s for s in all_samples if s.sample_id == query_sample), None)
    if requested and st.session_state.get(nav + "last_query_sample") != query_sample:
        st.session_state[nav + "work"] = requested.work_slug
        st.session_state[nav + "status"] = "all"
        st.session_state[nav + "multi"] = False
        st.session_state[nav + "sample"] = requested.sample_id
        st.session_state[nav + "last_query_sample"] = query_sample
    elif query_sample and not requested:
        st.warning(f"没有找到指定样本：{query_sample}")
    multi_only = st.sidebar.checkbox("只看多人框（≥2）", key=nav + "multi")
    st.sidebar.caption(f"已补多人框：{sum(len(s.persons) >= 2 for s in all_samples)} / {len(all_samples)}张；仍需逐框复核身份。")
    work = st.sidebar.selectbox("作品", [""] + list(work_names), format_func=lambda v: "全部作品" if not v else work_names[v], key=nav + "work")
    status = st.sidebar.selectbox("样本状态", ["all", "draft", "approved"], format_func=lambda v: {"all": "全部", "draft": "草稿 / 待复核", "approved": "已批准"}[v], key=nav + "status")
    available = [s for s in all_samples if (not work or s.work_slug == work) and (status == "all" or s.review_status == status) and (not multi_only or len(s.persons) >= 2)]
    if not available:
        st.info("当前筛选没有样本。")
        return
    if st.session_state.get(nav + "sample") not in {s.sample_id for s in available}:
        st.session_state[nav + "sample"] = available[0].sample_id
    selected = st.sidebar.selectbox("样本", [s.sample_id for s in available], key=nav + "sample")
    state_key = f"v2draft:{dataset_path}:{selected}"
    revision_key, canvas_key = state_key + ":revision", state_key + ":canvas_version"
    if state_key not in st.session_state:
        st.session_state[state_key] = store.get(selected)
        st.session_state[revision_key] = store.revision
        st.session_state[canvas_key] = 0
    if st.sidebar.button("恢复此图已保存标注", help="丢弃此图未保存的草稿修改"):
        st.session_state[state_key] = store.get(selected)
        st.session_state[revision_key] = store.revision
        st.session_state[canvas_key] += 1
        for key in list(st.session_state):
            if str(key).startswith(state_key + ":edit:"):
                del st.session_state[key]
        st.rerun()
    sample: MultiSample = st.session_state[state_key]
    if st.session_state[revision_key] != store.revision:
        st.warning("标注文件已有新修订。请点左侧「恢复此图已保存标注」加载最新图片和人物框；当前未保存编辑会保留到你主动恢复。")
    image_path = dataset_path.parent / sample.image_path
    if not image_path.is_file():
        st.error("原图不存在；请恢复原图引用，标注器不会替换原图。")
        return
    original = Image.open(image_path).convert("RGB")
    st.subheader(f"《{sample.work_name}》 · {sample.sample_id}")
    st.caption(f"{len(sample.persons)}个人物框 · {len(sample.ignore_regions)}个忽略区 · 原图{original.width}×{original.height} · 编号按左到右、再上到下排序")
    if st.button("查看A/B编号图（红框＋编号）", key=state_key + ":show_preview", use_container_width=True):
        st.query_params["sample"] = selected
        st.query_params["view"] = "preview"
        st.session_state[nav + "last_query_sample"] = selected
        st.rerun()
    if not sample.foreground_complete:
        st.warning("此图尚未确认前景人物已补齐，不能进入正式付费测试。继承姓名仅为待复核候选。")

    view_col, edit_col = st.columns([2, 1])
    preview_requested = st.query_params.get("view") == "preview"
    with view_col:
        # Fabric measures its frame once on mount; it must start in a visible tab.
        tab_key = state_key + f":views:{'preview' if preview_requested else 'canvas'}:{st.session_state[canvas_key]}"
        if preview_requested:
            preview_tab, raw_tab, canvas_tab = st.tabs(["编号与忽略区预览", "原图", "画布编辑"], key=tab_key)
        else:
            canvas_tab, raw_tab, preview_tab = st.tabs(["画布编辑", "原图", "编号与忽略区预览"], key=tab_key)
        with raw_tab:
            st.image(original, width="stretch")
        with canvas_tab:
            if preview_requested:
                st.caption("进入编辑后会在可见区域加载画布，保留当前草稿。")
                if st.button("进入画布编辑", key=state_key + ":open_canvas"):
                    st.query_params["view"] = "canvas"
                    st.rerun()
            else:
                try:
                    # Compatibility with recent Streamlit: the optional canvas imports this old symbol.
                    import streamlit.elements.image as streamlit_image
                    if not hasattr(streamlit_image, "image_to_url"):
                        from streamlit.elements.lib.image_utils import image_to_url
                        from streamlit.elements.lib.layout_utils import LayoutConfig

                        def canvas_image_to_url(image, width, clamp, channels, output_format, image_id):
                            return image_to_url(image, LayoutConfig(width=width), clamp, channels, output_format, image_id)

                        streamlit_image.image_to_url = canvas_image_to_url
                    from streamlit_drawable_canvas import st_canvas
                except (ImportError, StreamlitAPIException):
                    st.info("画布组件未安装或版本不兼容；右侧数值编辑支持新增、调整和删除全部框。安装项目[ui]依赖后可启用画布。")
                else:
                    kind = st.radio("编辑区域类型", ["person", "ignore"], format_func=lambda v: "人物红框" if v == "person" else "背景忽略区（橙框）", horizontal=True)
                    mode = st.radio("画布操作", ["rect", "transform"], format_func=lambda v: "绘制新框" if v == "rect" else "移动 / 调整已有框", horizontal=True)
                    width = min(640, original.width)
                    height = max(1, round(original.height * width / original.width))
                    result = st_canvas(background_image=original, initial_drawing=canvas_objects(sample, width, height, ignore=kind == "ignore"), width=width, height=height, drawing_mode=mode, fill_color="rgba(255,160,0,0.12)" if kind == "ignore" else "rgba(255,0,0,0.06)", stroke_color="#ff9900" if kind == "ignore" else "#ff0000", stroke_width=2, display_toolbar=False, update_streamlit=True, key=f"{state_key}:canvas:{kind}:{st.session_state[canvas_key]}")
                    st.caption("绘制或调整后点击应用；删除请用右侧对应框的删除按钮，保留人物与身份对应关系。")
                    if st.button("应用画布修改到草稿", key=f"{state_key}:apply_canvas"):
                        try:
                            if result.json_data is None:
                                raise ValueError("画布尚未就绪，请稍后点击应用")
                            updated = apply_canvas_objects(sample, result.json_data.get("objects", []), width, height, ignore=kind == "ignore")
                            st.session_state[state_key] = updated
                            st.session_state[canvas_key] += 1
                            for key in list(st.session_state):
                                if str(key).startswith(state_key + ":edit:"):
                                    del st.session_state[key]
                            st.rerun()
                        except ValueError as error:
                            st.error(str(error))
        with preview_tab:
            try:
                show_ignore = st.checkbox("叠加背景忽略区（仅审核用）", value=False, key=state_key + ":show_ignore")
                st.image(render_preview(image_path, sample, include_ignore=show_ignore), width="stretch")
                st.caption("默认显示A/B使用的编号红框；橙色忽略区仅用于审核，不发送给模型。")
            except (ValueError, ImportError) as error:
                st.info(f"请先修正框坐标再查看预览：{error}")

    with edit_col:
        st.markdown("**人物与精确坐标**")
        configured = next((w for w in load_works() if w.slug == sample.work_slug), None)
        names = load_roster(configured).names() if configured else []
        st.caption("角色名单供人工查阅：" + "、".join(names))
        to_delete = []
        for person in sample.persons:
            prefix = state_key + f":edit:{st.session_state[canvas_key]}:person:" + person.annotation_id
            with st.expander(f"人物 {person.person_id} · {person.character or '身份未知'} · {person.identity_status}", expanded=True):
                before = person.to_dict()
                character = st.text_input("角色姓名（未知请留空）", value=person.character or "", key=prefix + ":name", on_change=invalidate_ui_edit, args=(state_key, person.annotation_id)).strip()
                identity = st.selectbox("身份状态", ["needs_review", "unknown", "verified"], index=["needs_review", "unknown", "verified"].index(person.identity_status), format_func=lambda v: {"needs_review": "待复核：可能有候选姓名", "unknown": "身份未知：可定位但无法可靠定名", "verified": "身份已独立核实"}[v], key=prefix + ":identity", on_change=invalidate_ui_edit, args=(state_key, person.annotation_id))
                review = st.selectbox("此框复核状态", ["pending", "approved"], index=["pending", "approved"].index(person.review_status), format_func=lambda v: "待人工复核" if v == "pending" else "已批准", key=prefix + ":review")
                coords = st.columns(4)
                bbox = [coords[i].number_input(label, min_value=0.0, max_value=1.0, value=float(person.bbox[i]), step=0.01, format="%.4f", key=prefix + ":" + label, on_change=invalidate_ui_edit, args=(state_key, person.annotation_id, True)) for i, label in enumerate(["x1", "y1", "x2", "y2"])]
                existing = next((e.get("text", "") for e in reversed(person.evidence) if e.get("type") == "human_review"), "")
                text = st.text_area("独立证据 / 人工复核说明", value=existing, help="可填原片集数与时间、姓名字幕、官方资料链接及核验结论。继承v1标签不算独立证据。", key=prefix + ":evidence", on_change=invalidate_ui_edit, args=(state_key, person.annotation_id)).strip()
                note = st.text_input("此框备注", value=person.note, key=prefix + ":note")
                person.character = None if identity == "unknown" else character or None
                person.identity_status, person.review_status, person.bbox, person.note = identity, review, bbox, note
                person.evidence = [e for e in person.evidence if e.get("type") != "human_review"]
                if text:
                    person.evidence.append({"type": "human_review", "actor": "user", "text": text})
                if before != person.to_dict():
                    mark_draft(sample, geometry=before["bbox"] != bbox)
                if identity == "unknown" and character:
                    st.caption("unknown状态保存时姓名为null，未采用输入的候选姓名。")
                for error in person.validate():
                    st.warning(error)
                if st.button(f"删除人物 {person.person_id}", key=prefix + ":delete"):
                    to_delete.append(person.annotation_id)
        add_person = st.button("新增人物框（数值编辑）", key=state_key + ":addperson")
        if to_delete or add_person:
            sample.persons = [p for p in sample.persons if p.annotation_id not in to_delete]
            if add_person:
                sample.persons.append(PersonAnnotation(len(sample.persons) + 1, [0.1, 0.1, 0.35, 0.9]))
            sample.persons = renumber_persons(sample.persons)
            mark_draft(sample, geometry=True)
            st.session_state[canvas_key] += 1
            st.rerun()

        st.markdown("**不可辨识背景忽略区**")
        remove_regions = []
        for index, region in enumerate(sample.ignore_regions):
            prefix = f"{state_key}:edit:{st.session_state[canvas_key]}:ignore:{index}"
            with st.expander(f"忽略区 {index + 1}"):
                before = region.to_dict()
                coords = st.columns(4)
                region.bbox = [coords[i].number_input(label, min_value=0.0, max_value=1.0, value=float(region.bbox[i]), step=0.01, format="%.4f", key=prefix + ":" + label, on_change=invalidate_ui_edit, args=(state_key, None, True)) for i, label in enumerate(["x1", "y1", "x2", "y2"])]
                region.reason = st.text_input("忽略原因", value=region.reason, key=prefix + ":reason", on_change=invalidate_ui_edit, args=(state_key, None, True))
                if region.to_dict() != before:
                    mark_draft(sample, geometry=True)
                if st.button("删除此忽略区", key=prefix + ":delete"):
                    remove_regions.append(index)
        add_ignore = st.button("新增忽略区（数值编辑）", key=state_key + ":addignore")
        if remove_regions or add_ignore:
            sample.ignore_regions = [r for i, r in enumerate(sample.ignore_regions) if i not in remove_regions]
            if add_ignore:
                sample.ignore_regions.append(IgnoreRegion([0.05, 0.05, 0.25, 0.4]))
            mark_draft(sample, geometry=True)
            st.session_state[canvas_key] += 1
            for key in list(st.session_state):
                if str(key).startswith(state_key + ":edit:"):
                    del st.session_state[key]
            st.rerun()

    st.divider()
    complete = st.checkbox("我已检查整张图：所有符合范围的前景人物均已补框，忽略区合理", value=sample.foreground_complete, key=state_key + f":complete:{st.session_state[canvas_key]}")
    sample.foreground_complete = complete
    approve = st.checkbox("将本图设为已批准（要求每框已批准、身份已核实或unknown）", value=sample.review_status == "approved", key=state_key + f":approve:{st.session_state[canvas_key]}")
    sample.review_status = "approved" if approve else "draft"
    sample.note = st.text_input("样本备注", value=sample.note, key=state_key + ":sample_note")
    actor = st.text_input("本次操作者", value="user", key=state_key + ":actor")
    reason = st.text_input("本次修改原因（必填，写入前后值审计）", key=state_key + ":reason")
    save_col, refresh_col = st.columns(2)
    if refresh_col.button("刷新编号预览（按左到右重新编号）", use_container_width=True):
        if any(bbox_errors(p.bbox) for p in sample.persons):
            st.error("请先修正无效坐标")
        else:
            sample.persons = renumber_persons(sample.persons)
            st.session_state[canvas_key] += 1
            st.rerun()
    if save_col.button("保存此图v2标注与审计", type="primary", use_container_width=True):
        try:
            saved = store.save_sample(sample, actor=actor, reason=reason, expected_revision=st.session_state[revision_key])
            st.session_state[state_key] = saved
            st.session_state[revision_key] = store.revision
            st.session_state[canvas_key] += 1
            st.success("已保存v2标注和前后值审计。v1数据及原图未修改。")
            st.rerun()
        except (ValueError, ConcurrentModificationError) as error:
            st.error(str(error))
    blockers = sample.validate(for_run=True)
    with st.expander(f"正式测试门禁：{'通过' if not blockers else str(len(blockers)) + '项待完成'}"):
        for error in blockers:
            st.write("• " + error)
        if not blockers:
            st.write("本图标注满足结构和复核状态要求；付费执行仍需用户单独授权。")


if __name__ == "__main__":
    run_app()
