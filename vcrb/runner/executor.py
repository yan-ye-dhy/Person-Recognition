"""实验执行器：遍历 (模型 × 实验 × 样本)，幂等持久化，断点续跑。

- Ground Truth（target_character）只用于评分阶段，绝不进入请求；
- 相同样本的「红框标注图」A/B 两组复用同一文件（同一函数生成）；
- 每次调用结果立即落库 + 追加 raw JSONL，中断后重跑自动跳过已完成项。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
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
from ..providers import GenerationParams, ProviderResult, VisionProvider
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
            "note": "K3.1固定temperature=1故不显式传；其余接受者统一0.1。可关闭型号关闭思考，K3.1/GLM5.3使用不可关闭的low思考；具体模式见per_model_notes。",
        },
        "per_model_notes": {
            m.id: {
                "extra_body": m.extra_body,
                "passes_temperature": m.passes_temperature,
                "token_limit_parameter": m.params.get("token_limit_parameter", "max_tokens"),
                "official_series": m.official_series,
                "notes": m.notes,
            }
            for m in models
        },
        "image_encoding": "JPEG data-URL，长边≤1600，质量92（所有模型一致；详见各记录 image_meta）",
        "tools_disabled": "不传 tools/tool_choice；不启用联网搜索",
        "execution": {
            "concurrency": max(1, int(settings.get("runner", {}).get("concurrency", 1))),
            "per_model_concurrency_limit": 2,
            "per_model_concurrency": {
                model.id: max(1, int(
                    settings.get("runner", {}).get("per_model_concurrency", {}).get(model.id, 2)
                ))
                for model in models
            },
            "persistence": "主线程实时写入 SQLite 与 raw JSONL",
        },
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
    """Run calls with bounded optional concurrency; all persistence stays here.

    ``runner.concurrency`` defaults to one. Parallel mode rotates between models
    and permits two outstanding calls per model by default. Individual caps are
    read from ``runner.per_model_concurrency``. No unbounded backlog is submitted,
    so an interrupt can stop further paid work immediately.
    """
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
    concurrency = max(1, int(settings.get("runner", {}).get("concurrency", 1)))
    configured_caps = settings.get("runner", {}).get("per_model_concurrency", {})
    model_caps = [max(1, int(configured_caps.get(model.id, 2))) for model, _ in tasks]
    if len({model.id for model, _ in tasks}) != len(tasks):
        raise ValueError("A run may contain each model only once")

    # Metadata-only queues. Images are encoded lazily on the main thread and
    # reused byte-for-byte for every model and both experiments.
    pending = []
    for model, provider in tasks:
        calls = deque()
        for exp in experiments:
            exp_norm = exp.upper()
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
                calls.append((model, provider, exp_norm, sample, marked, work))
        pending.append(calls)

    image_cache: dict[str, tuple[str, dict[str, Any]]] = {}
    prompt_cache: dict[tuple[str, str], str] = {}

    def prepare(call):
        model, provider, exp_norm, sample, marked, work = call
        prompt_key = (exp_norm, sample.work_slug if exp_norm == "B" else "")
        if prompt_key not in prompt_cache:
            prompt_cache[prompt_key] = build_user_prompt(
                exp_norm, work if exp_norm == "B" else None
            )
        if sample.sample_id not in image_cache:
            image_cache[sample.sample_id] = encode_for_request(marked)
        data_url, img_meta = image_cache[sample.sample_id]
        return model, provider, exp_norm, sample, prompt_cache[prompt_key], data_url, img_meta

    def invoke(prepared):
        model, provider, _exp, _sample, prompt, data_url, _meta = prepared
        try:
            return provider.recognize(model, data_url, SYSTEM_PROMPT, prompt, gen)
        except Exception as exc:
            # Expected HTTP failures are classified by the adapter. Do not print
            # unexpected exception text, which could contain request credentials.
            return ProviderResult(
                ok=False, error_type="internal_error", error_message=type(exc).__name__
            )

    def persist(prepared, result):
        nonlocal done
        model, _provider, exp_norm, sample, prompt, _data_url, img_meta = prepared
        key = (model.id, exp_norm, sample.sample_id)
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
                model.params.get("token_limit_parameter", "max_tokens"): gen.max_tokens,
                "temperature": gen.temperature if model.passes_temperature else None,
                "extra_body": model.extra_body or None,
                "image_meta": img_meta,
                "tools": None,
            },
            "response": {
                "ok": result.ok,
                "raw_text": result.raw_text,
                "reasoning_text": getattr(result, "reasoning_text", ""),
                "output_issue": getattr(result, "output_issue", None),
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

    if concurrency == 1:
        # Keep the historical model -> experiment -> sample order by default.
        try:
            for calls in pending:
                while calls:
                    prepared = prepare(calls.popleft())
                    persist(prepared, invoke(prepared))
        except KeyboardInterrupt:
            summary.interrupted = True
    else:
        # Construct HTTP clients before sharing adapters across worker threads.
        # This performs no network requests and avoids races in lazy client setup.
        for (_model, provider), calls in zip(tasks, pending):
            initialize = getattr(provider, "_get_client", None)
            if calls and callable(initialize):
                initialize()

        pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="vcrb-api")
        in_flight: dict[Future, tuple[int, Any]] = {}
        active_by_model: Counter[int] = Counter()
        cursor = 0

        def submit_available():
            nonlocal cursor
            while len(in_flight) < concurrency:
                chosen = None
                for offset in range(len(pending)):
                    index = (cursor + offset) % len(pending)
                    if pending[index] and active_by_model[index] < model_caps[index]:
                        chosen = index
                        break
                if chosen is None:
                    return
                prepared = prepare(pending[chosen].popleft())
                future = pool.submit(invoke, prepared)
                in_flight[future] = (chosen, prepared)
                active_by_model[chosen] += 1
                cursor = (chosen + 1) % len(pending)

        def collect(future):
            index, prepared = in_flight.pop(future)
            active_by_model[index] -= 1
            persist(prepared, future.result())

        try:
            submit_available()
            while in_flight:
                completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in completed:
                    collect(future)
                submit_available()
        except KeyboardInterrupt:
            summary.interrupted = True
            # Cancel requests that have not started. Drain actual in-flight calls
            # to persist paid responses, without scheduling any replacement work.
            for future in list(in_flight):
                if future.cancel():
                    index, _prepared = in_flight.pop(future)
                    active_by_model[index] -= 1
            while in_flight:
                try:
                    completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                    for future in completed:
                        collect(future)
                except KeyboardInterrupt:
                    # A second interrupt must not turn an already returned paid
                    # response into an unrecorded request that will be sent again.
                    summary.interrupted = True
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    if summary.interrupted:
        store.finish_run(run_id, status="interrupted", note="用户中断，已停止新调用并保存返回结果，可续跑")
        return summary
    status = "done" if not summary.errors else "done_with_warnings"
    store.finish_run(
        run_id,
        status=status,
        note=f"executed={summary.executed} skipped={summary.skipped_existing} "
        f"ok={summary.ok} api_failures={summary.api_failures}",
    )
    return summary
