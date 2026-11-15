#!/usr/bin/env python3
"""Export ten fixed web comparison cases from saved API inputs, without API calls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vcrb.multi.identity_review import apply_corrections


RUN_ID = "multi_v2_20261113_075729_2e4f15"
MODEL_ID = "qwen3.8-max"
SAMPLE_IDS = tuple(
    f"gongfunvzu_{suffix}"
    for suffix in (
        "0003", "0008", "0012", "0013", "0016", "0020", "0022", "0023", "0030", "0040"
    )
)
IMAGE_DIRS = {"B": "B_numbered", "C": "C_unmarked"}
DEFAULT_OUTPUT = Path("reports/2026-11-15-qwenmax-review/gongfunvzu_web_test_10")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _csv_bytes(fieldnames: list[str], rows: list[dict]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8-sig")


def export_web_test_pack(repo_root: Path, output_dir: Path | None = None) -> dict:
    """Export validated artifacts; never overwrite a user's web response records.

    All inputs are read from the frozen completed run and the existing input gallery.
    The output folder may be outside the repository. Relative output paths are resolved
    against repo_root. This function never changes labels, scores, or SQLite contents.
    """
    repo_root = repo_root.resolve()
    output_dir = output_dir or DEFAULT_OUTPUT
    output_dir = output_dir if output_dir.is_absolute() else repo_root / output_dir
    output_dir = output_dir.resolve()
    gallery_dir = repo_root / "results/multi/input_gallery"
    gallery_path = gallery_dir / "manifest.json"
    gallery_rows = {row["sample_id"]: row for row in _json_file(gallery_path)["rows"]}
    dataset_path = repo_root / "results/multi" / RUN_ID / "dataset/dataset.json"
    dataset_bytes = dataset_path.read_bytes()
    frozen_samples = {row["sample_id"]: row for row in json.loads(dataset_bytes)["samples"]}
    corrections_path=repo_root / 'reports/2026-11-15-qwenmax-review/annotation_corrections.json'
    corrections=_json_file(corrections_path)['corrections'] if corrections_path.exists() else []
    samples=apply_corrections(frozen_samples,corrections)
    db_path = repo_root / "results/multi/runs.db"

    images: list[tuple[Path, bytes]] = []
    exported_rows: list[dict] = []
    answers: list[dict] = []
    record_rows: list[dict] = []
    prompt_sets: dict[str, tuple[str, str]] = {}
    reference_settings: dict[str, dict] = {}
    roles: Counter = Counter()
    person_counts: Counter = Counter()

    with sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        for sample_id in SAMPLE_IDS:
            sample = samples[sample_id]
            gallery = gallery_rows[sample_id]
            if sample["work_slug"] != "gongfunvzu":
                raise ValueError(f"Unexpected work: {sample_id}")
            if sample["image_sha256"] != gallery["source_sha256"]:
                raise ValueError(f"Frozen source mismatch: {sample_id}")
            people_meta = {
                person["person_id"]: person
                for person in gallery["human_annotations"]["persons"]
            }
            person_counts[len(sample["persons"])] += 1
            for person in sample["persons"]:
                meta = people_meta[person["person_id"]]
                frozen_person=next(p for p in frozen_samples[sample_id]['persons'] if p['person_id']==person['person_id'])
                if meta["character"] != frozen_person["character"]:
                    raise ValueError(f"Gallery identity does not match frozen source: {sample_id}")
                if meta["visibility"] != "clear":
                    raise ValueError(f"Selected person is not clear: {sample_id}")
                if meta["face_orientation"] not in ("front", "three_quarter"):
                    raise ValueError(f"Selected person is not frontal/three-quarter: {sample_id}")
                roles[person["character"]] += 1
                answers.append({
                    "sample_id": sample_id,
                    "person_id": person["person_id"],
                    "character": person["character"],
                    "character_aliases": json.dumps(person["character_aliases"], ensure_ascii=False),
                    "bbox_normalized": json.dumps(person["bbox"]),
                    "face_orientation": meta["face_orientation"],
                    "visibility": meta["visibility"],
                    "note": person["note"],
                    "annotation_id": person["annotation_id"],
                })
            for experiment, image_subdir in IMAGE_DIRS.items():
                records = connection.execute(
                    "SELECT raw_json FROM records "
                    "WHERE run_id=? AND model_id=? AND experiment=? AND sample_id=?",
                    (RUN_ID, MODEL_ID, experiment, sample_id),
                ).fetchall()
                if len(records) != 1:
                    raise ValueError(f"Expected one API record: {sample_id}/{experiment}")
                raw = json.loads(records[0][0])
                request = raw["request"]
                response = raw["response"]
                gallery_input = gallery["experiments"][experiment]
                source_path = gallery_dir / gallery_input["image"]
                image_bytes = source_path.read_bytes()
                image_sha = _sha(image_bytes)
                if image_sha != request["image_meta"]["encoded_sha256"]:
                    raise ValueError(f"Saved JPEG differs from API input: {sample_id}/{experiment}")
                if image_sha != gallery_input["encoded_sha256"]:
                    raise ValueError(f"Gallery JPEG hash mismatch: {sample_id}/{experiment}")
                if request["image_meta"]["source_sha256"] != sample["image_sha256"]:
                    raise ValueError(f"Request source differs from frozen data: {sample_id}/{experiment}")
                if request["user"] != gallery_input["user_prompt"]:
                    raise ValueError(f"Gallery prompt differs from API request: {sample_id}/{experiment}")
                prompts = (request["system"], request["user"])
                if experiment in prompt_sets and prompt_sets[experiment] != prompts:
                    raise ValueError(f"Prompts differ between selected cases: {experiment}")
                prompt_sets[experiment] = prompts
                settings = {key: value for key, value in request.items() if key not in ("image_meta", "system", "user")}
                if experiment in reference_settings and reference_settings[experiment] != settings:
                    raise ValueError(f"API settings differ between selected cases: {experiment}")
                reference_settings[experiment] = settings
                relative_output = Path(image_subdir) / f"{sample_id}.jpg"
                images.append((relative_output, image_bytes))
                exported_rows.append({
                    "sample_id": sample_id,
                    "experiment": experiment,
                    "image": relative_output.as_posix(),
                    "encoded_sha256": image_sha,
                    "source_sha256": sample["image_sha256"],
                    "gallery_image": source_path.relative_to(repo_root).as_posix(),
                    "image_meta": request["image_meta"],
                    "person_count": len(sample["persons"]),
                    "face_orientations": [people_meta[p["person_id"]]["face_orientation"] for p in sample["persons"]],
                    "visibility": [people_meta[p["person_id"]]["visibility"] for p in sample["persons"]],
                    "requested_model": request["model"],
                    "served_model": response.get("served_model"),
                })
                record_rows.append({
                    "sample_id": sample_id,
                    "experiment": experiment,
                    "image": relative_output.as_posix(),
                    "website": "",
                    "displayed_model": "",
                    "thinking_mode": "",
                    "online_search": "",
                    "web_test_time": "",
                    "conversation_url_or_id": "",
                    "full_response": "",
                    "format_valid": "",
                    "correct_people": "",
                    "notes": "",
                })

    if person_counts != Counter({1: 8, 2: 2}) or len(roles) != 5:
        raise ValueError("Selected sample mix is not eight single-person/two double-person/five roles")
    # Complete all source validation before creating any output files.
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative_path, data in images:
        path = output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f"Refusing to replace a different existing input image: {path}")
        path.write_bytes(data)

    system_prompts = {prompts[0] for prompts in prompt_sets.values()}
    if len(system_prompts) != 1:
        raise ValueError("Selected experiments have different system prompts")
    (output_dir / "system_prompt_exact.txt").write_text(next(iter(system_prompts)), encoding="utf-8")
    for experiment, (system, user) in prompt_sets.items():
        (output_dir / f"{experiment}_user_prompt_exact.txt").write_text(user, encoding="utf-8")
        (output_dir / f"{experiment}_web_copy_prompt.txt").write_text(system + "\n\n" + user, encoding="utf-8")
    _write_json(output_dir / "api_reference_settings.json", {
        "run_id": RUN_ID,
        "model_id": MODEL_ID,
        "experiments": reference_settings,
        "note": "原始 API 的 system/user 分属不同消息；web_copy 文件仅为网页单输入框合并版本，不能保证网页后端模型或参数相同。",
    })
    (output_dir / "answers_do_not_upload.csv").write_bytes(_csv_bytes(list(answers[0]), answers))
    record_path = output_dir / "web_results_to_fill.csv"
    if not record_path.exists():
        record_path.write_bytes(_csv_bytes(list(record_rows[0]), record_rows))

    instructions = f"""功夫女足：10张图片的网页版对照测试

本文件夹包含10个底层样本，各有B编号版、C无框版，共20张JPEG。
选择：8张单人图、2张双人图；标注目标均为清晰正脸或四分之三侧脸，共覆盖5个角色。
样本：{', '.join(SAMPLE_IDS)}
图片与完成运行 {RUN_ID} 中 {MODEL_ID} 实际提交API的JPEG逐字节一致；没有重新压缩或重画框。

怎样开始
1. 先做B组：打开 B_numbered 文件夹，选择一张图片；网页版新建一个空白对话，只上传这一张图片。
2. 打开 B_web_copy_prompt.txt，全选复制，粘贴到网页版并发送。该提示词已经包含作品名，请勿再提供角色姓名或答案表。
3. 如果网页版可选择模型，记录页面实际显示的名称；第一轮关闭联网搜索。思考模式可关闭时先关闭，与原API设置接近；不支持关闭则如实记录。
4. 将完整原文回答、测试时间、显示模型、联网/思考开关、对话地址记入 web_results_to_fill.csv。不要只记录你概括后的结论。
5. 下一张图片重新新建空白对话；不要在上一张对话里继续测试，以免前文提示答案。
6. 做完B组再做C组：每张仍使用新对话，上传 C_unmarked 下同编号图片，粘贴 C_web_copy_prompt.txt。
7. 所有回答保存好后，再打开 answers_do_not_upload.csv 核对。该表是人工答案，不得上传给模型。

文件说明
- B_numbered / C_unmarked：仅有待上传图片，不含角色姓名与答案文件。
- system_prompt_exact.txt、B_user_prompt_exact.txt、C_user_prompt_exact.txt：当时API的原始提示词，原样保存。
- B_web_copy_prompt.txt、C_web_copy_prompt.txt：将原system与user内容拼在一起，供只有单输入框的网页使用。
- api_reference_settings.json：原API参数，包含temperature=0.1、max_tokens=4096、enable_thinking=false、tools=null等实际记录。
- answers_do_not_upload.csv：独立答案表，包含编号对应角色、允许别名与标注框；只供测试后人工核对。
- web_results_to_fill.csv：20次对照的空白记录模板。重复执行导出脚本不会覆盖此表中的已填结果。
- manifest.json：图片哈希、原始来源、姿态和清晰度，便于核验输入一致。

怎么理解结果
网页版可能使用不同后端版本、默认系统提示词、图像处理或工具设置；网页单输入框也无法原样复现API的消息角色分离。
所以记录网页显示的配置，不预先认定网页与API必为同一模型或同一参数。
如需观察联网搜索效果，应在第一轮之后另开一轮新对话，并另存记录，不与关闭搜索的结果混合。
此10张为清晰角色的定向诊断样本；不能替代《功夫女足》全部40张的正式分数，也不能单凭10张判断低分的根本原因。
这里沿用冻结标注并叠加用户明确确认的身份更正与既有评分规则；不因为网页答复不同就直接更改答案。
此导出过程不调用任何模型、不产生新的API费用。
"""
    (output_dir / "使用说明.txt").write_text(instructions, encoding="utf-8")
    manifest = {
        "kind": "gongfunvzu_web_comparison_10",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": RUN_ID,
        "model_id": MODEL_ID,
        "frozen_dataset": dataset_path.relative_to(repo_root).as_posix(),
        "frozen_dataset_file_sha256": _sha(dataset_bytes),
        "annotation_corrections":corrections,
        "input_gallery_manifest": gallery_path.relative_to(repo_root).as_posix(),
        "underlying_sample_count": len(SAMPLE_IDS),
        "exported_image_count": len(images),
        "sample_ids": list(SAMPLE_IDS),
        "single_person_images": person_counts[1],
        "two_person_images": person_counts[2],
        "target_person_count": len(answers),
        "distinct_character_count": len(roles),
        "selection_policy": "清晰人物优先；8张单人、2张双人；正脸/四分之三侧脸；排除缺少清晰样本的孖八。仅用于定向诊断，不代表全量分数。",
        "api_calls": 0,
        "inputs_verified_against_raw_request_sha256": True,
        "rows": exported_rows,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = export_web_test_pack(args.repo_root, args.output_dir)
    print(json.dumps({key: manifest[key] for key in ("run_id", "underlying_sample_count", "exported_image_count", "target_person_count", "distinct_character_count", "api_calls")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
