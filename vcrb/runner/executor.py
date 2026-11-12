"""实验执行器：遍历 (模型 × 实验 × 样本)，幂等持久化，断点续跑。

- Ground Truth（target_character）只用于评分阶段，绝不进入请求；
- 相同样本的「红框标注图」A/B 两组复用同一文件（同一函数生成）；
- 每次调用结果立即落库 + 追加 raw JSONL，中断后重跑自动跳过已完成项。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import Work
from ..dataset.schema import Sample
from ..imageutils import draw_target_box, encode_for_request
from ..models import ModelSpec
from ..paths import RESULTS_DIR
from ..prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt, describe_marking
from ..providers import GenerationParams, VisionProvider
from .store import RunStore

Experiment = str  # "A" | "B"


@dataclass
class RunSummary:
    run_id: str
    planned: int = 0
    executed: int = 0
    skipped_existing: int = 0
    ok: int = 0
    api_failures: int = 0
    interrupted: bool = False
    errors: list[str] = field(default_factory=list)


def dataset_hash(samples: list[Sample]) -> str:
    payload = json.dumps([s.to_dict() for s in samples], ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def select_pilot_samples(samples: list[Sample], per_work: int = 10) -> list[Sample]:
    """每作品按 sampling_group 分层轮转抽取 per_work 张（覆盖尽量多的组与角色）。"""
    by_work: dict[str, dict[str, list[Sample]]] = {}
    for s in sorted(samples, key=lambda x: x.sample_id):
        by_work.setdefault(s.work_slug, {}).setdefault(s.sampling_group, []).append(s)
    picked: list[Sample] = []
    for slug in sorted(by_work):
        buckets = by_work[slug]
        groups = sorted(buckets)
        cursor = {g: 0 for g in groups}
        count = 0
        while count < per_work:
            advanced = False
            for g in groups:
                idx = cursor[g]
                if idx < len(buckets[g]):
                    picked.append(buckets[g][idx])
                    cursor[g] += 1
                    count += 1
                    advanced = True
                    if count >= per_work:
                        break
            if not advanced:
                break
    return picked


def _fairness_meta(models: list[ModelSpec], settings: dict[str, Any]) -> dict[str, Any]:
    gen = settings.get("generation", {})
    return {
        "marking": describe_marking(),
        "prompt_version": PROMPT_VERSION,
        "generation_params": {
            "temperature": gen.get("temperature", 0.1),
            "max_tokens": gen.get("max_tokens", 512),
            "note": "Kimi k2.5/k2.6 不接受显式采样参数（官方文档），故未传 temperature，其余模型统一 temperature",
        },
        "per_model_notes": {
            m.id: {
                "extra_body": m.extra_body,
                "passes_temperature": m.passes_temperature,
                "official_series": m.official_series,
                "notes": m.notes,
            }
            for m in models
        },
        "image_encoding": "JPEG data-URL，长边≤1600，质量92（所有模型一致；详见各记录 image_meta）",
        "tools_disabled": "不传 tools/tool_choice；不启用联网搜索",
    }


def execute_run(
    run_id: str,
    tasks: list[tuple[ModelSpec, VisionProvider]],
    samples: list[Sample],
    works_by_slug: dict[str, Work],
    experiments: list[Experiment],
    store: RunStore,
    settings: dict[str, Any],
    dataset_dir: Path,
    marked_dir: Path | None = None,
    raw_dir: Path | None = None,
    retry_failed: bool = False,
    progress_cb: Callable[[int, str], None] | None = None,
) -> RunSummary:
    gen = GenerationParams(
        temperature=float(settings.get("generation", {}).get("temperature", 0.1)),
        max_tokens=int(settings.get("generation", {}).get("max_tokens", 512)),
    )
    marked_dir = marked_dir or (RESULTS_DIR / "marked")
    marked_dir.mkdir(parents=True, exist_ok=True)

    summary = RunSummary(run_id=run_id)
    summary.planned = len(tasks) * len(experiments) * len(samples)
    ok_keys = store.existing_ok_keys(run_id)
    failed_keys = store.failed_keys(run_id)

    # 预生成红框标注图（每样本一份，A/B 复用）
    marked_paths: dict[str, Path | None] = {}
    for sample in samples:
        src = dataset_dir / sample.image_path
        if not src.is_file():
            summary.errors.append(f"{sample.sample_id}: 图片缺失 {src}")
            continue
        dst = marked_dir / f"{sample.sample_id}.jpg"
        if not dst.exists():
            draw_target_box(src, dst, sample.target_bbox)
        marked_paths[sample.sample_id] = dst

    done = 0
    try:
        for model, provider in tasks:
            for exp in experiments:
                exp_norm = exp.upper()
                prompt_a: str | None = None          # A 组：全局唯一提示词
                prompt_b_by_work: dict[str, str] = {}  # B 组：按作品缓存
                for sample in samples:
                    key = (model.id, exp_norm, sample.sample_id)
                    if key in ok_keys:
                        summary.skipped_existing += 1
                        done += 1
                        continue
                    if not retry_failed and key in failed_keys:
                        # 已有失败记录且未要求重试：不重复付费调用
                        summary.skipped_existing += 1
                        done += 1
                        continue
                    marked = marked_paths.get(sample.sample_id)
                    if marked is None:
                        summary.errors.append(f"{sample.sample_id}: 无标注图，跳过")
                        done += 1
                        continue
                    work = works_by_slug.get(sample.work_slug)
                    if exp_norm == "B" and work is None:
                        summary.errors.append(f"{sample.sample_id}: 未知作品 {sample.work_slug}")
                        done += 1
                        continue
                    if exp_norm == "A":
                        if prompt_a is None:
                            prompt_a = build_user_prompt("A")
                        prompt = prompt_a
                    else:
                        if sample.work_slug not in prompt_b_by_work:
                            prompt_b_by_work[sample.work_slug] = build_user_prompt("B", work)
                        prompt = prompt_b_by_work[sample.work_slug]

                    data_url, img_meta = encode_for_request(marked)
                    result = provider.recognize(
                        model, data_url, SYSTEM_PROMPT, prompt, gen
                    )

                    raw_file = store.raw_file_for(run_id, model.id, exp_norm, base=raw_dir)
                    raw_payload = {
                        "sample_id": sample.sample_id,
                        "work_slug": sample.work_slug,
                        "experiment": exp_norm,
                        "model_id": model.id,
                        "request": {
                            "model": model.id,
                            "system": SYSTEM_PROMPT,
                            "user": prompt,
                            "max_tokens": gen.max_tokens,
                            "temperature": gen.temperature if model.passes_temperature else None,
                            "extra_body": model.extra_body or None,
                            "image_meta": img_meta,
                            "tools": None,
                        },
                        "response": {
                            "ok": result.ok,
                            "raw_text": result.raw_text,
                            "finish_reason": result.finish_reason,
                            "served_model": result.served_model,
                            "usage": result.usage,
                            "latency_ms": result.latency_ms,
                            "attempts": result.attempts,
                            "error_type": result.error_type,
                            "error_message": result.error_message,
                            "http_status": result.http_status,
                        },
                        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    }
                    record = {
                        "run_id": run_id,
                        "model_id": model.id,
                        "experiment": exp_norm,
                        "sample_id": sample.sample_id,
                        "work_slug": sample.work_slug,
                        "attempt": result.attempts,
                        "ok": int(result.ok),
                        "http_status": result.http_status,
                        "latency_ms": result.latency_ms,
                        "error_type": result.error_type,
                        "error_message": result.error_message,
                        "served_model": result.served_model,
                        "usage": json.dumps(result.usage, ensure_ascii=False),
                        "raw_json": json.dumps(raw_payload, ensure_ascii=False),
                        "raw_file": str(raw_file),
                        "created_at": raw_payload["created_at"],
                    }
                    store.upsert_record(record)
                    store.append_raw(raw_file, raw_payload)

                    if result.ok:
                        summary.ok += 1
                        ok_keys.add(key)
                    else:
                        summary.api_failures += 1
                        failed_keys.add(key)
                    summary.executed += 1
                    done += 1
                    if progress_cb:
                        progress_cb(done, f"{model.id}/{exp_norm}/{sample.sample_id}")
    except KeyboardInterrupt:
        summary.interrupted = True
        store.finish_run(run_id, status="interrupted", note="用户中断，可重跑续传")
        return summary
    status = "done" if not summary.errors else "done_with_warnings"
    store.finish_run(
        run_id,
        status=status,
        note=f"executed={summary.executed} skipped={summary.skipped_existing} "
        f"ok={summary.ok} api_failures={summary.api_failures}",
    )
    return summary
