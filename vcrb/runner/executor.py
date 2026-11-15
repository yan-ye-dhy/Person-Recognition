"""实验执行器：遍历 (模型 × 实验 × 样本)，幂等持久化，断点续跑。

- Ground Truth（target_character）只用于评分阶段，绝不进入请求；
- 相同样本的「红框标注图」A/B 两组复用同一文件（同一函数生成）；
- 每次调用结果立即落库 + 追加 raw JSONL，中断后重跑自动跳过已完成项。
"""

from __future__ import annotations

import hashlib
import json
import math
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
    search_enabled=any(m.params.get('search_enabled') is True or m.extra_body.get('enable_search') is True for m in models)
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
                "request_params": m.params,
                "http_timeout_seconds": _effective_http_timeout_seconds(m, settings),
            }
            for m in models
        },
        "image_encoding": "JPEG data-URL，长边≤1600，质量92（所有模型一致；详见各记录 image_meta）",
        "tools_disabled": "仅启用模型配置中明确声明的搜索能力；保留实际工具证据" if search_enabled else "不传 tools/tool_choice；不启用联网搜索",
        "search_policy":{m.id:_request_tool_metadata(m, settings) for m in models},
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


def _effective_http_timeout_seconds(model: ModelSpec, settings: dict[str, Any] | None = None) -> float:
    """Validate operational per-model HTTP timeouts without changing model inputs."""
    settings = settings or {}
    runner = settings.get('runner') or {}
    overrides = runner.get('per_model_http_timeout_seconds', {})
    if not isinstance(overrides, dict):
        raise ValueError('runner.per_model_http_timeout_seconds must be a mapping')
    for model_id, value in overrides.items():
        if (not isinstance(model_id, str) or not model_id or type(value) not in (int, float)
                or not math.isfinite(value) or value <= 0):
            raise ValueError('Per-model HTTP timeout values must be finite positive numbers keyed by model ID')
    value = overrides.get(model.id, (settings.get('retry') or {}).get('timeout_seconds', 120))
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Effective HTTP timeout must be a finite positive number')
    return float(value)


def _request_tool_metadata(model: ModelSpec, settings: dict[str, Any] | None = None) -> dict[str, Any]:
    enabled=model.params.get('search_enabled') is True or model.extra_body.get('enable_search') is True
    mode=model.params.get('api_mode','chat_completions')
    native=mode=='dashscope_anthropic'
    maximum=model.params.get('search_max_uses',1)
    forced=(model.extra_body.get('search_options') or {}).get('forced_search') is True
    tools = None
    if enabled:
        if native:
            tools = [{'type':'web_search_20250305','name':'web_search','max_uses':maximum}]
        elif mode == 'deepseek_native_search':
            tools = [{'type':'deepseek_web_search'}]
        elif mode == 'moonshot_responses':
            tools = [{'type':'web_search'}]
        elif mode == 'dashscope_multimodal_search':
            tools = [{'type':'native_web_search','search_strategy':'max','enable_source':True}]
        elif mode == 'zhipu_native_search':
            tools = [{'type':'web_search','web_search':{'enable':True,'search_engine':'search_pro','search_result':True}}]
        elif mode == 'zhipu_function_search':
            tools = [{'type':'function','name':'web_search','vendor':'zhipu','search_engine':'search_pro'}]
    return {
        'http_timeout_seconds': _effective_http_timeout_seconds(model, settings),
        'api_mode':mode,
        'tools':tools,
        'tool_choice':{'type':'tool','name':'web_search'} if enabled and native and forced else None,
        'stream':bool(mode=='dashscope_multimodal_search' or mode=='chat_completions' and (enabled or model.params.get('stream'))),
        'search_requested':enabled,
        'search_max_uses':maximum if enabled and native else None,
        'protocol_system_marker':'x-anthropic-billing-header: cc_entrypoint=cli;' if native else None,
        'search_instruction_version':'c-search-1' if enabled else None,
        'search_limit_semantics':'requested_not_enforced' if enabled and native else None,
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
    retry_failed_models: set[str] | None = None,
    retry_failed_keys: set[tuple[str, str, str]] | None = None,
    progress_cb: Callable[[int, str], None] | None = None,
    input_builder: Callable | None = None,
    system_prompt: str = SYSTEM_PROMPT,
    max_new_tasks: int | None = None,
    sample_first: bool = False,
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
        if input_builder is None and not dst.exists():
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
        pairs = ((exp, sample) for sample in samples for exp in experiments) if sample_first else (
            (exp, sample) for exp in experiments for sample in samples
        )
        for exp, sample in pairs:
            exp_norm = exp.upper()
            key = (model.id, exp_norm, sample.sample_id)
            if key in ok_keys:
                summary.skipped_existing += 1
                done += 1
                continue
            if key in failed_keys and (not retry_failed or
                    retry_failed_models is not None and model.id not in retry_failed_models or
                    retry_failed_keys is not None and key not in retry_failed_keys):
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
            if exp_norm in ("B", "C") and work is None:
                summary.errors.append(f"{sample.sample_id}: 未知作品 {sample.work_slug}")
                done += 1
                continue
            calls.append((model, provider, exp_norm, sample, marked, work))
        pending.append(calls)

    if max_new_tasks is not None:
        if max_new_tasks < 0:
            raise ValueError("max_new_tasks must be nonnegative")
        # Round-robin the per-model queues, preserving bounded fairness and
        # the existing idempotent task keys when a request/budget cap is used.
        limited = [deque() for _ in pending]
        remaining = max_new_tasks
        while remaining and any(pending):
            for index, calls in enumerate(pending):
                if calls and remaining:
                    limited[index].append(calls.popleft())
                    remaining -= 1
        pending = limited

    image_cache: dict[str, tuple[str, dict[str, Any]]] = {}
    prompt_cache: dict[tuple[str, str], str] = {}

    def prepare(call):
        model, provider, exp_norm, sample, marked, work = call
        if input_builder is not None:
            prompt, data_url, img_meta = input_builder(sample, exp_norm, work)
            return model, provider, exp_norm, sample, prompt, data_url, img_meta
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
            return provider.recognize(model, data_url, system_prompt, prompt, gen)
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
                "system": system_prompt,
                "user": prompt,
                model.params.get("token_limit_parameter", "max_tokens"): gen.max_tokens,
                "temperature": gen.temperature if model.passes_temperature else None,
                "extra_body": model.extra_body or None,
                "image_meta": img_meta,
                **_request_tool_metadata(model, settings),
            },
            "response": {
                "ok": result.ok,
                "raw_text": result.raw_text,
                "reasoning_text": getattr(result, "reasoning_text", ""),
                "output_issue": getattr(result, "output_issue", None),
                "finish_reason": result.finish_reason,
                "served_model": result.served_model,
                "usage": result.usage,
                "search_metadata":getattr(result,'search_metadata',{}),
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
