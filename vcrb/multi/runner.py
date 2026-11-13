"""Reuse v1 providers, scheduling and persistence for the separate v2 protocol."""
from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime, timezone
import base64
import hashlib
import json
import math
from pathlib import Path
import shutil
import fcntl
import sqlite3
import threading
from urllib.parse import urlsplit
from uuid import uuid4

from ..config import load_settings, load_works
from ..imageutils import draw_target_boxes, encode_for_request
from ..models import load_registry
from ..paths import ROOT, RESULTS_DIR
from ..providers.errors import classify_exception
from ..runner.executor import execute_run, _fairness_meta
from ..runner.store import RunStore
from .prompts import SYSTEM_PROMPT, PROMPT_VERSION, build_user_prompt
from .schema import load_dataset, dataset_hash, validate_dataset
from .stratification import SIDECAR_NAME, load_stratification, freeze_stratification, load_frozen_stratification

MODEL_IDS = (
    "deepseek-flash", "kimi-k3.1-pro", "glm-5.3-flash", "glm-4.6v",
    "qwen3.8-max", "qwen3.7-plus", "qwen3.8-flash",
)
MULTI_ROOT = RESULTS_DIR / "multi"
MULTI_DB = MULTI_ROOT / "runs.db"
REAL_CALL_GATE = ROOT / "data" / "v2" / "real_call_gate.json"


def _confirmed_real_gate(expected_hash=None):
    try:
        gate = json.loads(REAL_CALL_GATE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("真实调用确认门禁缺失或无法读取，拒绝联网") from error
    if not isinstance(gate, dict) or gate.get("state") != "confirmed_by_user":
        raise ValueError("标注正在等待用户确认，真实新建/续跑已暂停；旧--yes和approval-note不能越过确认门禁")
    if expected_hash is not None and gate.get("confirmed_dataset_hash") != expected_hash:
        raise ValueError("标注已变化或确认hash不匹配；须用户确认当前完整dataset后才可真实调用")
    return gate


def _explicit_budget_extension(gate, root_run_id, previous_maximum, requested_maximum):
    """A revision may raise the cumulative cap only after a separate user decision.

    All old known/uncertain costs stay in the same scope. This is neither a
    refund nor a new budget starting at zero; historical run metadata is kept.
    """
    approval = (gate or {}).get("budget_extension") or {}
    if (not isinstance(approval, dict) or approval.get("state") != "confirmed_by_user"
            or approval.get("root_run_id") != root_run_id
            or type(approval.get("previous_max_cost_cny")) not in (int, float)
            or type(approval.get("max_cost_cny")) not in (int, float)
            or approval["previous_max_cost_cny"] != previous_maximum
            or approval["max_cost_cny"] != requested_maximum
            or not requested_maximum > previous_maximum
            or not isinstance(approval.get("user_instruction"), str)
            or not approval["user_instruction"].strip()
            or not approval.get("confirmed_at")):
        raise ValueError("修订run必须保持原预算上限；提高累计上限须用户单独明确确认，旧未知费用不能释放或归零")
    return deepcopy(approval)


def resolve_models(ids=None):
    registry = load_registry()
    selected = list(ids or MODEL_IDS)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(MODEL_IDS):
        raise ValueError("v2只允许现有7个模型，不能重复或擅自新增型号")
    return [registry.get(mid) for mid in selected]


def select_samples(samples, per_work=0, limit=0):
    if per_work:
        selected = []
        for slug in sorted({s.work_slug for s in samples}):
            group = sorted((s for s in samples if s.work_slug == slug), key=lambda s: (len(s.persons), s.sample_id))
            # Alternate low/high counts so a pilot includes both simpler and
            # crowded annotations when they have actually been completed.
            chosen = []
            while group and len(chosen) < per_work:
                chosen.append(group.pop(0) if len(chosen) % 2 == 0 else group.pop())
            selected.extend(chosen)
        samples = selected
    return samples[:limit] if limit else samples


def pilot_first_samples(samples):
    """Execution order only: one single and one multi sample per work first.

    The snapshot/hash retain the caller's original order. A work lacking one
    category uses its earliest remaining sample; no person count is relabelled.
    """
    chosen = []
    for slug in dict.fromkeys(s.work_slug for s in samples):
        group = [s for s in samples if s.work_slug == slug]
        single = next((s for s in group if len(s.persons) == 1), None)
        multi = next((s for s in group if len(s.persons) >= 2), None)
        picks = [s for s in (single, multi) if s is not None]
        for sample in group:
            if len(picks) >= min(2, len(group)):
                break
            if sample not in picks:
                picks.append(sample)
        chosen.extend(picks)
    ids = {s.sample_id for s in chosen}
    return chosen + [s for s in samples if s.sample_id not in ids]


def task_upper_cost(model_id, settings, pricing):
    """Conservative reservation; absent/uncertain prices never imply free calls.

    Include all configured retry attempts. A smaller input bound must match a
    documented provider context ceiling; historical averages are not evidence.
    Prices must include visual input and non-cached token rates.
    """
    rate = (pricing or {}).get("models", {}).get(model_id) or {}
    values = [rate.get("input_cny_per_million"), rate.get("output_cny_per_million")]
    bound = rate.get("input_token_upper_bound")
    if (any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values)
            or not _valid_input_bound(model_id, rate)
            or not str(rate.get("source_url", "")).startswith("https://")):
        return None
    attempts = max(1, int(settings.get("retry", {}).get("max_attempts", 4)))
    tokens = int(settings.get("generation", {}).get("max_tokens", 4096))
    return (bound * values[0] + tokens * values[1]) / 1_000_000 * attempts


def _valid_input_bound(model_id, rate):
    bound = rate.get("input_token_upper_bound")
    if type(bound) is not int or bound < 1:
        return False
    if bound >= 1_048_576:
        return True  # Backward-compatible, conservative full-context ceiling.
    proof = rate.get("input_bound_evidence") or {}
    # These are provider ceilings verified in the checked-in pricing evidence,
    # not predictions based on the old single-person workload.
    ceilings = {"glm-4.6v": (131_072, {"docs.bigmodel.cn", "open.bigmodel.cn", "bigmodel.cn"})}
    ceiling, hosts = ceilings.get(model_id, (None, set()))
    url = urlsplit(str(proof.get("source_url", "")))
    return (ceiling is not None and bound >= ceiling
            and proof.get("kind") == "provider_context_limit"
            and type(proof.get("context_tokens")) is int and proof["context_tokens"] == ceiling
            and url.scheme == "https" and url.hostname in hosts and bool(proof.get("verified_on")))


class _BudgetPause(KeyboardInterrupt):
    """Internal scheduling pause: no request was sent for this task."""


class _RollingBudget:
    """Durable per-call reservations, with no DB work on the executor connection.

    A supporting provider admits each HTTP attempt separately. Uncertain actual
    attempts retain their bounds, while retries never sent occupy no budget.
    Legacy providers retain the entire task bound. Responses remain untouched.
    """
    def __init__(self, db_path, run_id, settings, pricing, maximum):
        self.db_path, self.run_id = Path(db_path), run_id
        self.settings, self.pricing, self.maximum = settings, pricing, maximum
        self.attempts = max(1, int(settings.get("retry", {}).get("max_attempts", 4)))
        self.max_output = int(settings.get("generation", {}).get("max_tokens", 4096))
        self.lock = threading.RLock()
        self.paused = self.user_interrupted = self.breached = False
        self.review_blocked = False
        self.blocked_providers = set()
        self.model_providers = {m.id: m.provider for m in load_registry().models}
        with self.lock, self._connect() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS multi_budget_calls (
                reservation_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, model_id TEXT NOT NULL,
                encoded_sha256 TEXT NOT NULL, prompt_sha256 TEXT NOT NULL,
                reserved_cny REAL NOT NULL, committed_cny REAL NOT NULL,
                known_cny REAL NOT NULL DEFAULT 0, uncertain_cny REAL NOT NULL DEFAULT 0,
                status TEXT NOT NULL, usage_json TEXT NOT NULL DEFAULT '{}',
                attempts INTEGER, task_count INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL, settled_at TEXT, note TEXT NOT NULL DEFAULT '')""")
            conn.execute("CREATE INDEX IF NOT EXISTS multi_budget_run ON multi_budget_calls(run_id)")
            columns = {row[1] for row in conn.execute("PRAGMA table_info(multi_budget_calls)")}
            for column, declaration in (("accounting_unit", "TEXT NOT NULL DEFAULT 'task'"),
                                        ("attempt_no", "INTEGER"),
                                        ("outcome_json", "TEXT NOT NULL DEFAULT '{}'")):
                if column not in columns:
                    conn.execute(f"ALTER TABLE multi_budget_calls ADD COLUMN {column} {declaration}")
            meta = json.loads(conn.execute("SELECT fairness_meta FROM runs WHERE run_id=?", (run_id,)).fetchone()[0])
            self.confirmed_hash = meta.get("confirmed_dataset_hash") or conn.execute("SELECT dataset_hash FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]
            previous = meta.get("cost_reservation") or {}
            self.breached = bool(previous.get("budget_breached"))
            for member in self._scope_runs(conn):
                # The execution lock covers this DB, so none of these requests
                # can still belong to another active v2 controller.
                conn.execute("UPDATE multi_budget_calls SET status='uncertain_after_restart', uncertain_cny=committed_cny WHERE run_id=? AND status='in_flight'", (member,))
                prior = json.loads(conn.execute("SELECT fairness_meta FROM runs WHERE run_id=?", (member,)).fetchone()[0]).get("cost_reservation") or {}
                count = conn.execute("SELECT count(*) FROM multi_budget_calls WHERE run_id=?", (member,)).fetchone()[0]
                amount = float(prior.get("own_run_reserved_cny", prior.get("reserved_cny", 0)))
                if not count and amount > 0:
                    conn.execute("INSERT INTO multi_budget_calls (reservation_id,run_id,model_id,encoded_sha256,prompt_sha256,reserved_cny,committed_cny,uncertain_cny,status,task_count,created_at,note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                                 (uuid4().hex, member, "legacy", "", "", amount, amount, amount, "legacy_uncertain", int(prior.get("reserved_tasks", 0)), self._now(), "Preserved legacy reservation; no automatic refund"))
            self._sync(conn)

    @staticmethod
    def _now():
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _scope_runs(self, conn):
        """All revisions sharing this explicit budget, including sibling runs."""
        own = json.loads(conn.execute("SELECT fairness_meta FROM runs WHERE run_id=?", (self.run_id,)).fetchone()[0])
        root = (own.get("budget_scope") or {}).get("root_run_id", self.run_id)
        members = []
        for rid, encoded in conn.execute("SELECT run_id,fairness_meta FROM runs"):
            meta = json.loads(encoded)
            if rid == root or ((meta.get("budget_scope") or {}).get("root_run_id") == root):
                if meta.get("benchmark_protocol") != "multi-v2" or meta.get("is_mock"):
                    raise ValueError("预算组不能混入非真实v2运行")
                members.append(rid)
                self.breached = self.breached or bool((meta.get("cost_reservation") or {}).get("budget_breached"))
        return sorted(members)

    def _sync(self, conn):
        scope = self._scope_runs(conn)
        placeholders = ",".join("?" for _ in scope)
        values = conn.execute("""SELECT coalesce(sum(task_count),0), coalesce(sum(committed_cny),0),
            coalesce(sum(reserved_cny),0), coalesce(sum(known_cny),0), coalesce(sum(uncertain_cny),0),
            coalesce(sum(CASE WHEN status='in_flight' THEN committed_cny ELSE 0 END),0),
            coalesce(sum(CASE WHEN accounting_unit='http_attempt' THEN 1 ELSE 0 END),0)
            FROM multi_budget_calls WHERE run_id IN (""" + placeholders + ")", scope).fetchone()
        own = conn.execute("SELECT coalesce(sum(committed_cny),0) FROM multi_budget_calls WHERE run_id=?", (self.run_id,)).fetchone()[0]
        meta = json.loads(conn.execute("SELECT fairness_meta FROM runs WHERE run_id=?", (self.run_id,)).fetchone()[0])
        meta["cost_reservation"] = {
            "policy_version": 3, "reserved_tasks": values[0], "reserved_cny": values[1],
            "reserved_total_cny": values[2], "known_usage_upper_cny": values[3],
            "uncertain_cny": values[4], "inflight_cny": values[5],
            "released_cny": max(0., values[2] - values[1]),
            "ledger_table": "multi_budget_calls", "budget_breached": self.breached,
            "scope_run_ids": scope, "own_run_reserved_cny": own,
            "reserved_http_attempts": values[6], "blocked_providers": sorted(self.blocked_providers),
            "real_call_gate_blocked": self.review_blocked,
            "note": "Scope includes inherited revision runs; known usage uses configured upper rates, uncertain and in-flight attempts retain bounds. Not a vendor bill; no automatic historical refunds.",
        }
        conn.execute("UPDATE runs SET fairness_meta=? WHERE run_id=?", (json.dumps(meta, ensure_ascii=False), self.run_id))
        return meta["cost_reservation"]

    def available(self):
        with self.lock, self._connect() as conn:
            state = self._sync(conn)
            return max(0., self.maximum - state["reserved_cny"])

    def is_blocked(self, model_id):
        return self.model_providers.get(model_id) in self.blocked_providers

    def reserve(self, model_id, image_data_url, prompt, *, attempt_no=None):
        try:
            _confirmed_real_gate(self.confirmed_hash)
        except ValueError as error:
            self.review_blocked = self.paused = True
            with self.lock, self._connect() as conn:
                self._sync(conn)
            raise _BudgetPause("User confirmation changed; no request sent") from error
        upper = task_upper_cost(model_id, self.settings, self.pricing)
        if upper is None:
            raise ValueError("No verified per-model budget bound")
        if attempt_no is not None:
            upper /= self.attempts
        with self.lock, self._connect() as conn:
            state = self._sync(conn)
            if self.paused or self.breached or self.user_interrupted or self.is_blocked(model_id) or state["reserved_cny"] + upper > self.maximum + 1e-10:
                self.paused = True
                raise _BudgetPause("Budget occupied/exhausted; no request sent")
            reservation = uuid4().hex
            encoded = hashlib.sha256(base64.b64decode(image_data_url.split(",", 1)[1], validate=True)).hexdigest()
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            conn.execute("INSERT INTO multi_budget_calls (reservation_id,run_id,model_id,encoded_sha256,prompt_sha256,reserved_cny,committed_cny,status,created_at,accounting_unit,attempt_no,task_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (reservation, self.run_id, model_id, encoded, prompt_hash, upper, upper, "in_flight", self._now(),
                          "http_attempt" if attempt_no is not None else "task", attempt_no,
                          1 if attempt_no in (None, 1) else 0))
            self._sync(conn)
            # Commit before the provider can send this actual HTTP request.
        return reservation

    def before_attempt(self, model, image_data_url, prompt, attempt_no):
        if type(attempt_no) is not int or not 1 <= attempt_no <= self.attempts:
            self.paused = self.breached = True
            with self.lock, self._connect() as conn:
                self._sync(conn)
            raise _BudgetPause("Provider exceeded configured attempt limit; no request sent")
        return self.reserve(model.id, image_data_url, prompt, attempt_no=attempt_no)

    def after_attempt(self, reservation, model, *, result=None, error=None):
        """Settle exactly one HTTP call. Failure never implies zero cost."""
        rate = self.pricing["models"][model.id]
        upper = task_upper_cost(model.id, self.settings, self.pricing) / self.attempts
        usage = getattr(result, "usage", None) if result is not None else None
        reliable = (result is not None and result.ok and isinstance(usage, dict)
                    and type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] > 0
                    and type(usage.get("completion_tokens")) is int and usage["completion_tokens"] >= 0)
        if reliable and "total_tokens" in usage:
            reliable = type(usage["total_tokens"]) is int and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        known, uncertain, committed = 0., upper, upper
        status, note = "attempt_uncertain", "One actual HTTP attempt retained at its bound; no assumption that an error is free"
        error_type = classify_exception(error)[0] if isinstance(error, Exception) else None
        if error_type == "insufficient_balance":
            with self.lock:
                self.blocked_providers.add(model.provider)
        if isinstance(error, KeyboardInterrupt):
            self.user_interrupted = True
        if reliable:
            known = (usage["prompt_tokens"] * rate["input_cny_per_million"] + usage["completion_tokens"] * rate["output_cny_per_million"]) / 1_000_000
            committed, uncertain = known, 0.
            status, note = "settled_attempt_usage", "This HTTP attempt's reliable usage priced at configured upper rates"
            if usage["prompt_tokens"] > rate["input_token_upper_bound"] or usage["completion_tokens"] > self.max_output or committed > upper + 1e-10:
                self.breached = self.paused = True
                status, note = "bound_breached", "Observed usage exceeded bound; retain observed cost and stop calls"
        outcome = {"ok": bool(result and result.ok), "http_status": getattr(result, "http_status", None) if error is None else getattr(error, "status_code", None),
                   "error_type": error_type, "interrupted": isinstance(error, BaseException) and not isinstance(error, Exception)}
        # Only protocol status/type are recorded; exception text may contain IDs.
        with self.lock, self._connect() as conn:
            conn.execute("UPDATE multi_budget_calls SET committed_cny=?,known_cny=?,uncertain_cny=?,status=?,usage_json=?,attempts=1,settled_at=?,note=?,outcome_json=? WHERE reservation_id=? AND run_id=?",
                         (committed, known, uncertain, status, json.dumps(usage or {}), self._now(), note,
                          json.dumps(outcome), reservation, self.run_id))
            self._sync(conn)

    def settle(self, reservation, model_id, result=None, interrupted=False):
        rate = self.pricing["models"][model_id]
        upper = task_upper_cost(model_id, self.settings, self.pricing)
        per_attempt = upper / self.attempts
        usage = getattr(result, "usage", None) if result is not None else None
        attempts = getattr(result, "attempts", None) if result is not None else None
        reliable = (result is not None and result.ok and isinstance(usage, dict)
                    and type(usage.get("prompt_tokens")) is int and usage["prompt_tokens"] > 0
                    and type(usage.get("completion_tokens")) is int and usage["completion_tokens"] >= 0
                    and type(attempts) is int and 1 <= attempts <= self.attempts)
        if reliable and "total_tokens" in usage:
            reliable = type(usage["total_tokens"]) is int and usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        known, uncertain, committed = 0., upper, upper
        status = "interrupted_uncertain" if interrupted else "failed_or_missing_usage"
        note = "Entire retry bound retained: failed, interrupted, or incomplete/unreliable usage"
        if reliable:
            known = (usage["prompt_tokens"] * rate["input_cny_per_million"] + usage["completion_tokens"] * rate["output_cny_per_million"]) / 1_000_000
            uncertain = (attempts - 1) * per_attempt
            committed = known + uncertain
            status, note = "settled_usage", "Final attempt usage priced conservatively; previous failed attempts retain individual full bounds"
            if usage["prompt_tokens"] > rate["input_token_upper_bound"] or usage["completion_tokens"] > self.max_output or committed > upper + 1e-10:
                self.breached = self.paused = True
                status, note = "bound_breached", "Provider usage exceeded configured bound; actual observed cost retained and new requests stopped"
        elif type(attempts) is int and attempts > self.attempts:
            self.breached = self.paused = True
            uncertain = committed = attempts * per_attempt
            status, note = "bound_breached", "Provider reported more attempts than configured; retain all reported attempt bounds and stop new calls"
        with self.lock, self._connect() as conn:
            conn.execute("UPDATE multi_budget_calls SET committed_cny=?,known_cny=?,uncertain_cny=?,status=?,usage_json=?,attempts=?,settled_at=?,note=? WHERE reservation_id=? AND run_id=?",
                         (committed, known, uncertain, status, json.dumps(usage or {}, ensure_ascii=False), attempts, self._now(), note, reservation, self.run_id))
            self._sync(conn)

    def wrap(self, provider, model_id):
        ledger = self
        class BudgetedProvider:
            name = getattr(provider, "name", "budgeted")

            def _get_client(self):
                initialize = getattr(provider, "_get_client", None)
                return initialize() if callable(initialize) else None

            def recognize(self, model, image_data_url, system_prompt, user_prompt, gen):
                if getattr(provider, "supports_attempt_observer", False):
                    return provider.recognize(model, image_data_url, system_prompt, user_prompt, gen,
                                              attempt_observer=ledger)
                reservation = ledger.reserve(model_id, image_data_url, user_prompt)
                try:
                    result = provider.recognize(model, image_data_url, system_prompt, user_prompt, gen)
                except BaseException as error:
                    if isinstance(error, KeyboardInterrupt) and not isinstance(error, _BudgetPause):
                        ledger.user_interrupted = True
                    ledger.settle(reservation, model_id, interrupted=True)
                    raise
                ledger.settle(reservation, model_id, result)
                return result
        return BudgetedProvider()


def plan(samples, models, experiments, settings, pricing=None):
    experiments = [e.upper() for e in experiments]
    if not experiments or len(set(experiments)) != len(experiments) or set(experiments) - {"A", "B", "C"}:
        raise ValueError("实验只能为不重复的A/B/C")
    costs = {m.id: task_upper_cost(m.id, settings, pricing) for m in models}
    blockers = []
    for sample in samples:
        errors = sample.validate(for_run=True)
        if errors:
            blockers.append({"sample_id": sample.sample_id, "errors": errors})
    n = len(samples) * len(experiments)
    return {
        "protocol": "multi-v2", "samples": len(samples), "persons": sum(len(s.persons) for s in samples),
        "models": [m.id for m in models], "experiments": experiments,
        "planned_requests": n * len(models), "annotation_blocked_samples": len(blockers),
        "annotation_blockers": blockers, "cost_upper_bound_per_task_cny": costs,
        "cost_upper_bound_cny": sum(v * n for v in costs.values()) if all(v is not None for v in costs.values()) else None,
        "cost_note": "价格未配置时不估造费用；含视觉输入的非缓存上界费率及有主源证据的上下文上限按实际HTTP尝试预留，可靠usage返回后结算，未知费用保留；不支持该hook的provider使用全重试保守预留。整批最坏上界不是费用预测或实时账单。",
    }


class InputBuilder:
    def __init__(self, dataset_dir, marked_dir):
        self.dataset_dir, self.marked_dir = Path(dataset_dir), Path(marked_dir)
        self.cache = {}

    def __call__(self, sample, experiment, work):
        experiment = experiment.upper()
        mode = "unmarked" if experiment == "C" else "numbered"
        key = sample.sample_id, mode
        if key not in self.cache:
            source = (self.dataset_dir / sample.image_path).resolve()
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            if digest != sample.image_sha256:
                raise ValueError(f"{sample.sample_id}: 原图哈希变化，拒绝调用")
            image = source
            if mode == "numbered":
                image = self.marked_dir / f"{sample.sample_id}.jpg"
                draw_target_boxes(source, image, sample.persons)
            data_url, meta = encode_for_request(image)
            meta.update(source_sha256=digest, input_file_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                        encoded_sha256=hashlib.sha256(base64.b64decode(data_url.split(",", 1)[1])).hexdigest(),
                        annotation_mode=mode)
            self.cache[key] = data_url, meta
        prompt = build_user_prompt(experiment, work if experiment != "A" else None,
                                   [p.person_id for p in sample.persons] if experiment != "C" else None)
        data_url, meta = self.cache[key]
        return prompt, data_url, dict(meta)


def run_multi(dataset, **kwargs):
    """One v2 scheduler per DB prevents concurrent resume/double reservation."""
    db_path = Path(kwargs.get("db_path", MULTI_DB))
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with db_path.with_suffix(".execution.lock").open("a") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("已有v2运行正在使用此数据库，不能并行重复调度") from error
        return _run_multi(dataset, **kwargs)


def _run_multi(dataset, *, experiments=("A", "B", "C"), model_ids=None, mock=True,
              yes=False, max_requests=None, max_cost_cny=None, approval_note="",
              pricing=None, per_work=0, limit=0, run_id=None, retry_failed=False,
              db_path=MULTI_DB, output_root=MULTI_ROOT, provider_overrides=None,
              budget_parent_run_id=None):
    """An explicit --yes is insufficient without approved annotations and budget.

    Mock accepts drafts to exercise the workflow, but their identities are not
    magically verified. Synthetic fixture results are labelled independently.
    """
    from ..cli import _build_provider
    from .mock import MultiMockProvider

    confirmed_gate = None
    complete_samples = None
    if not mock:
        gate = _confirmed_real_gate()
        complete_samples = load_dataset(Path(dataset).resolve())
        if gate.get("confirmed_dataset_hash") != dataset_hash(complete_samples):
            raise ValueError("标注已变化或确认hash不匹配；须用户确认当前完整dataset后才可真实调用")
        confirmed_gate = gate
    dataset, db_path, output_root = Path(dataset).resolve(), Path(db_path), Path(output_root).resolve()
    experiments = tuple(e.upper() for e in experiments)
    if per_work < 0 or limit < 0:
        raise ValueError("per_work/limit必须为非负数")
    models, settings = resolve_models(model_ids), load_settings()
    store = RunStore(db_path)
    existing = store.get_run(run_id) if run_id else None
    pending_budget_extension = None
    if run_id and existing is None:
        raise ValueError("续跑run-id不存在；拒绝因编号拼错而创建新的付费运行")
    if existing:
        meta = existing["fairness_meta"]
        if budget_parent_run_id is not None and budget_parent_run_id != (meta.get("budget_scope") or {}).get("parent_run_id"):
            raise ValueError("续跑不能修改已冻结的预算继承关系")
        if model_ids is None:
            models = resolve_models(existing["models"])
        if meta.get("benchmark_protocol") != "multi-v2" or bool(meta.get("is_mock")) != mock:
            raise ValueError("不能将旧单目标/真实与Mock响应复用为v2结果")
        if meta.get("prompt_version") != PROMPT_VERSION or existing["prompt_version"] != PROMPT_VERSION:
            raise ValueError("提示词版本已变化，不能用新口径续跑旧run；旧响应仅可离线查看/评分")
        if existing["models"] != [m.id for m in models] or existing["experiments"] != list(experiments):
            raise ValueError("续跑必须保持模型与实验范围")
        current = select_samples(complete_samples if complete_samples is not None else load_dataset(dataset), per_work, limit)
        if dataset_hash(current) != existing["dataset_hash"]:
            raise ValueError("标注已变化，不能续跑旧输入；请建立新版本run")
        samples = load_dataset(Path(existing["dataset_path"]))
        dataset_dir = Path(existing["dataset_path"]).parent
        if dataset_hash(samples) != existing["dataset_hash"]:
            raise ValueError("冻结快照已修改")
        load_frozen_stratification(existing, samples)
        if not mock and meta.get("confirmed_dataset_hash", dataset_hash(samples)) != confirmed_gate["confirmed_dataset_hash"]:
            raise ValueError("旧快照不属于本次已确认完整dataset，拒绝混入旧标注真实续跑")
        frozen_settings = deepcopy(meta["settings_snapshot"])
        frozen_settings["runner"] = settings.get("runner", {})
        settings = frozen_settings
        pricing = meta.get("pricing")
        requested_maximum, max_cost_cny = max_cost_cny, meta.get("max_cost_cny")
        if (not mock and requested_maximum is not None and requested_maximum != max_cost_cny
                and confirmed_gate.get("budget_extension")):
            root_id = (meta.get("budget_scope") or {}).get("root_run_id", run_id)
            pending_budget_extension = _explicit_budget_extension(
                confirmed_gate, root_id, max_cost_cny, requested_maximum)
            max_cost_cny = requested_maximum
        approval_note = meta.get("approval_note", "")
        for model in models:
            notes = meta["per_model_notes"][model.id]
            if (notes["extra_body"] != model.extra_body or notes["passes_temperature"] != model.passes_temperature
                    or notes.get("token_limit_parameter", "max_tokens") != model.params.get("token_limit_parameter", "max_tokens")):
                raise ValueError("模型请求参数变化，不能混入旧run")
        if not mock and not yes:
            raise ValueError("真实续跑仍需显式--yes")
    else:
        samples = select_samples(complete_samples if complete_samples is not None else load_dataset(dataset), per_work, limit)
        dataset_dir = dataset.parent
    stratification_payload = None
    if not existing and (dataset.parent / SIDECAR_NAME).is_file():
        stratification_payload = load_stratification(dataset.parent / SIDECAR_NAME,
                                                   complete_samples if complete_samples is not None else load_dataset(dataset))
    errors, _stats = validate_dataset(samples, dataset_dir, for_run=not mock)
    if errors:
        raise ValueError("标注/输入门禁未通过：" + "; ".join(errors[:8]))
    if not samples:
        raise ValueError("样本为空")
    execution_plan = plan(samples, models, experiments, settings, pricing)
    if not mock:
        if any(any(e.get("type") == "synthetic" for e in sample.evidence) for sample in samples):
            raise ValueError("合成Mock样例不能用于真实模型评测")
        if not yes or not approval_note.strip():
            raise ValueError("真实付费任务需要--yes及明确的approval-note；本轮尚未获授权")
        if (type(max_cost_cny) not in (int, float) or not math.isfinite(max_cost_cny) or max_cost_cny <= 0
                or any(c is None for c in execution_plan["cost_upper_bound_per_task_cny"].values())):
            raise ValueError("必须配置可核验价格及正的max-cost-cny，未知费用不能启动付费任务")
    if mock and budget_parent_run_id is not None:
        raise ValueError("Mock不能继承真实费用预算")
    if pending_budget_extension is not None:
        # Explicitly authorized amendment to this current run only. Preserve
        # the original parent run, all raw responses and every cost row.
        meta = deepcopy(meta)
        meta.setdefault("budget_amendments", []).append({
            "previous_max_cost_cny": meta["max_cost_cny"],
            "max_cost_cny": max_cost_cny, "approval": pending_budget_extension,
            "recorded_at": datetime.now(timezone.utc).isoformat()})
        meta["max_cost_cny"] = max_cost_cny
        meta.setdefault("budget_scope", {"root_run_id": run_id, "parent_run_id": None}).update(
            max_cost_cny=max_cost_cny, user_confirmed_extension=pending_budget_extension)
        store.conn.execute("UPDATE runs SET fairness_meta=? WHERE run_id=?",
                           (json.dumps(meta, ensure_ascii=False), run_id))
        store.conn.commit()
    budget_scope = None
    if not mock and not existing:
        paid_runs = {row[0]: json.loads(row[1]) for row in store.conn.execute("SELECT run_id,fairness_meta FROM runs")
                     if (json.loads(row[1]).get("benchmark_protocol") == "multi-v2" and not json.loads(row[1]).get("is_mock"))}
        if paid_runs and budget_parent_run_id is None:
            raise ValueError("已有真实v2费用，修订run必须显式设置budget_parent_run_id继承预算；不能归零重开")
        if budget_parent_run_id is not None:
            parent = paid_runs.get(budget_parent_run_id)
            if parent is None:
                raise ValueError("预算父run不存在或不是本库真实v2运行")
            inherited = parent.get("budget_scope") or {}
            maximum = inherited.get("max_cost_cny", parent.get("max_cost_cny"))
            if type(maximum) not in (int, float):
                raise ValueError("父运行缺少有效累计预算上限")
            root_run_id = inherited.get("root_run_id", budget_parent_run_id)
            extension = None
            if max_cost_cny != maximum:
                extension = _explicit_budget_extension(confirmed_gate, root_run_id, maximum, max_cost_cny)
            budget_scope = {"root_run_id": root_run_id,
                            "parent_run_id": budget_parent_run_id, "max_cost_cny": max_cost_cny,
                            "note": "All revisions share the original cap, including prior uncertain charges"}
            if extension:
                budget_scope["user_confirmed_extension"] = extension
    rid = run_id if existing else "multi_v2_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid4().hex[:6]
    run_dir = output_root / rid
    if not existing:
        snapshot_dir = run_dir / "dataset"
        snapshot_dir.mkdir(parents=True)
        for sample in samples:
            src = (dataset_dir / sample.image_path).resolve()
            dst = (snapshot_dir / sample.image_path).resolve()
            if not dst.is_relative_to(run_dir):
                raise ValueError("图片相对路径逃逸冻结run目录")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        snapshot = snapshot_dir / "dataset.json"
        snapshot.write_text(json.dumps({"schema_version": "2.0", "samples": [s.to_dict() for s in samples]}, ensure_ascii=False, indent=2))
        meta = _fairness_meta(models, settings)
        meta.update(benchmark_protocol="multi-v2", is_mock=mock,
                    synthetic_fixture=all(any(e.get("type") == "synthetic" for e in s.evidence) for s in samples),
                    dataset_snapshot_path=str(snapshot), settings_snapshot=settings, image_root=str(snapshot_dir),
                    pricing=pricing, max_cost_cny=max_cost_cny, approval_note=approval_note,
                    cost_reservation_policy="Per-HTTP-attempt reservation when supported; retain every uncertain actual attempt, with legacy full-task fallback; inherited revisions share the original cap",
                    prompt_version=PROMPT_VERSION, sample_ids=[s.sample_id for s in samples],
                    initial_quality=execution_plan, image_hashes={s.sample_id: s.image_sha256 for s in samples})
        if stratification_payload is not None:
            meta["stratification"] = freeze_stratification(stratification_payload, samples, snapshot_dir)
        if not mock:
            meta["budget_scope"] = budget_scope or {"root_run_id": rid, "parent_run_id": None, "max_cost_cny": max_cost_cny}
            meta["confirmed_dataset_hash"] = confirmed_gate["confirmed_dataset_hash"]
            meta["real_call_confirmation"] = {key: confirmed_gate.get(key) for key in
                                               ("state", "confirmed_dataset_hash", "confirmed_at", "user_instruction")}
        store.create_run(rid, list(experiments), [m.id for m in models], PROMPT_VERSION,
                         dataset_hash(samples), str(snapshot), meta, label="MOCK / 非真实模型成绩" if mock else "多人物A/B/C")
        dataset_dir = snapshot_dir
    cap = max_requests
    if cap is not None and (type(cap) is not int or cap < 0):
        raise ValueError("max-requests必须为非负整数")
    builder = InputBuilder(dataset_dir, run_dir / "marked")
    execution_samples = pilot_first_samples(samples)
    pilot_count = sum(min(2, sum(s.work_slug == slug for s in samples)) for slug in {s.work_slug for s in samples})
    pilot_samples = execution_samples[:pilot_count]
    if provider_overrides:
        providers = [provider_overrides[m.id] for m in models]
        if mock and any(getattr(p, "name", "") != "multi_mock" for p in providers):
            raise ValueError("Mock模式不能接真实Provider")
    elif mock:
        from .mock import build_demo_responses
        scripted = build_demo_responses(samples, dataset_dir, builder) if meta.get("synthetic_fixture") else {}
        providers = [MultiMockProvider(scripted) for _ in models]
    else:
        registry = load_registry()
        providers = [_build_provider(m, registry, settings) for m in models]
    initial_ok, initial_failed = store.existing_ok_keys(rid), store.failed_keys(rid)
    skipped = sum((m.id, e, s.sample_id) in initial_ok or
                  (not retry_failed and (m.id, e, s.sample_id) in initial_failed)
                  for m in models for e in experiments for s in samples)
    executed, interrupted, budget = 0, False, None
    works = {w.slug: w for w in load_works()}
    if mock:
        store.conn.execute("update runs set status='running' where run_id=?", (rid,));store.conn.commit()
        summary = execute_run(rid, list(zip(models, providers)), execution_samples, works, list(experiments), store, settings, dataset_dir,
                              marked_dir=run_dir / "marked", raw_dir=output_root / "raw", retry_failed=retry_failed,
                              input_builder=builder, system_prompt=SYSTEM_PROMPT, max_new_tasks=cap, sample_first=True)
        executed, interrupted = summary.executed, summary.interrupted
    else:
        budget = _RollingBudget(db_path, rid, settings, pricing, max_cost_cny)
        attempted = set()
        class InvocationStore:
            """A failed task is retried at most once during this CLI invocation.

            Budget contention may restart the scheduler after draining existing
            calls. This in-memory skip set never changes stored ok/failure data.
            """
            def __getattr__(self, name):
                return getattr(store, name)

            def existing_ok_keys(self, run_id):
                return store.existing_ok_keys(run_id) | attempted

            def upsert_record(self, record):
                store.upsert_record(record)
                attempted.add((record["model_id"], record["experiment"], record["sample_id"]))
        invocation_store = InvocationStore()
        wrapped = {m.id: budget.wrap(p, m.id) for m, p in zip(models, providers)}
        admission_costs = {m.id: execution_plan["cost_upper_bound_per_task_cny"][m.id] /
                           (budget.attempts if getattr(p, "supports_attempt_observer", False) else 1)
                           for m, p in zip(models, providers)}
        while cap is None or executed < cap:
            ok_keys, failed_keys = invocation_store.existing_ok_keys(rid), store.failed_keys(rid)
            # Keep a pilot phase barrier even across internal budget pauses;
            # fast models must not consume a 168-task pilot cap on later images.
            pilot_pending = any((m.id, e, s.sample_id) not in ok_keys and
                                (retry_failed or (m.id, e, s.sample_id) not in failed_keys)
                                for m in models if not budget.is_blocked(m.id)
                                for e in experiments for s in pilot_samples)
            active_samples = pilot_samples if pilot_pending else execution_samples
            pending_models = [m for m in models if any(
                (m.id, e, s.sample_id) not in ok_keys and (retry_failed or (m.id, e, s.sample_id) not in failed_keys)
                for e in experiments for s in active_samples)]
            available = budget.available()
            eligible = [m for m in pending_models if not budget.is_blocked(m.id) and admission_costs[m.id] <= available + 1e-10]
            if not eligible or budget.breached or budget.review_blocked:
                break
            budget.paused = False
            store.conn.execute("update runs set status='running' where run_id=?", (rid,));store.conn.commit()
            summary = execute_run(rid, [(m, wrapped[m.id]) for m in eligible], active_samples, works, list(experiments),
                                  invocation_store, settings, dataset_dir, marked_dir=run_dir / "marked", raw_dir=output_root / "raw",
                                  retry_failed=retry_failed, input_builder=builder, system_prompt=SYSTEM_PROMPT,
                                  max_new_tasks=None if cap is None else cap - executed, sample_first=True)
            executed += summary.executed
            if budget.user_interrupted or (summary.interrupted and not budget.paused):
                interrupted = True
                break
            if not summary.executed and not budget.paused:
                break
        if not interrupted:
            complete = store.run_progress(rid)["total"] >= execution_plan["planned_requests"]
            store.finish_run(rid, "done" if complete else "limited",
                             "Annotation confirmation changed; remaining tasks were not called" if budget.review_blocked else
                             "Rolling per-model budget settled; unexecuted tasks remain explicit" if not budget.breached
                             else "Provider violated a token/cost bound; new calls stopped, observed cost retained")
    if mock and store.run_progress(rid)["total"] < execution_plan["planned_requests"] and not interrupted:
        store.finish_run(rid, "limited", "Request/budget cap; unexecuted tasks remain explicit")
    return {"run_id": rid, "is_mock": mock, "planned": execution_plan["planned_requests"],
            "executed": executed, "skipped": skipped,
            "progress": store.run_progress(rid), "directory": str(run_dir),
            "db_path": str(db_path), "plan": execution_plan,
            "budget": store.get_run(rid)["fairness_meta"].get("cost_reservation")}
