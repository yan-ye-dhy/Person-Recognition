"""vcrb 命令行入口。

付费红线：`run --yes`、`verify --yes` 才会发起真实 API 调用，且需要用户事先授权；
默认 dry-run 只打印计划与预估调用量。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import paths
from .config import (
    ConfigError,
    get_api_key,
    load_settings,
    load_works,
)
from .models import ModelRegistry, ModelSpec, load_registry
from .paths import CANDIDATES_DIR, DATASET_JSON, FINAL_DIR, RAW_DIR, SCORES_DIR, REPORTS_DIR
from .providers import MockProvider, OpenAICompatProvider, VisionProvider

app = typer.Typer(
    help="国产多模态大模型影视角色识别 Benchmark",
    no_args_is_help=True,
    add_completion=False,
)
models_app = typer.Typer(help="模型注册表", no_args_is_help=True)
collect_app = typer.Typer(help="素材采集", no_args_is_help=True)
roster_app = typer.Typer(help="角色名单", no_args_is_help=True)
review_app = typer.Typer(help="评分人工复核", no_args_is_help=True)
app.add_typer(models_app, name="models")
app.add_typer(collect_app, name="collect")
app.add_typer(roster_app, name="roster")
app.add_typer(review_app, name="review")

console = Console()

WORK_SLUGS = ["kuangbiao", "langyabang", "gongfunvzu", "jianlai"]


def _fail(msg: str) -> None:
    console.print(f"[red]错误：[/red]{msg}")
    raise typer.Exit(1)


# ================================================================ models


@models_app.command("list")
def models_list(all: bool = typer.Option(False, "--all", help="包含可选与文本模型")):
    registry = load_registry()
    table = Table(title="模型注册表（2026-11 官方文档核实）")
    for col in ["模型 ID", "厂商", "档位", "视觉", "启用", "官方系列定位"]:
        table.add_column(col)
    rows = registry.all_models() if all else registry.runnable_models()
    for m in rows:
        table.add_row(
            m.id,
            m.vendor,
            m.tier,
            {"doc": "✓(文档)", "none": "✗", "unverified": "待验证"}[m.supports_vision],
            "✓" if m.enabled else "—",
            (m.official_series[:44] + "…") if len(m.official_series) > 44 else m.official_series,
        )
    console.print(table)
    ineligible = registry.vision_ineligible_models()
    if ineligible:
        console.print(
            "[yellow]不支持视觉输入（不参与评测）：[/yellow]"
            + "、".join(f"{m.id}({m.vendor})" for m in ineligible)
        )


@models_app.command("verify")
def models_verify(
    yes: bool = typer.Option(False, "--yes", help="真实调用（每模型 1 张小图，产生少量费用）"),
    models: Optional[str] = typer.Option(None, "--models", help="逗号分隔，默认全部启用模型"),
):
    """分级验证①：每模型 1 张合成小图连通性检查（图片输入能力实测）。"""
    from .imageutils import encode_for_request
    from .providers import GenerationParams
    from .runner.store import RunStore
    from .prompts import SYSTEM_PROMPT

    registry = load_registry()
    settings = load_settings()
    targets = registry.runnable_models()
    if models:
        wanted = {m.strip() for m in models.split(",")}
        targets = [m for m in targets if m.id in wanted]
        missing = wanted - {m.id for m in targets}
        if missing:
            _fail(f"未找到可运行的启用模型：{missing}")

    table = Table(title="连通性检查计划")
    table.add_column("模型")
    table.add_column("Key")
    for m in targets:
        spec = registry.provider_spec(m)
        key_ok = get_api_key(spec.api_key_env) is not None
        table.add_row(m.id, "✓" if key_ok else "[red]缺失[/red]")
    console.print(table)
    missing_keys = [
        m.id for m in targets if get_api_key(registry.provider_spec(m).api_key_env) is None
    ]
    if missing_keys:
        console.print(f"[yellow]以下模型缺少 API Key，将被跳过：{missing_keys}[/yellow]")
    if not yes:
        console.print("[cyan]dry-run：加 --yes 执行真实检查（每模型 1 次小图请求）[/cyan]")
        return

    # 合成小图（不依赖数据集）
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (320, 200), (40, 90, 160)).save(buf, format="JPEG")
    tmp = paths.RESULTS_DIR / "verify_probe.jpg"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(buf.getvalue())
    data_url, _meta = encode_for_request(tmp)

    store = RunStore()
    gen = GenerationParams(
        temperature=float(settings["generation"]["temperature"]),
        max_tokens=128,
    )
    probe_prompt = '请描述这张图片的主体颜色。输出 JSON：{"character_name": null, "is_uncertain": true, "reason": "..."}'
    results = Table(title="连通性检查结果")
    for col in ["模型", "结果", "实际版本", "错误"]:
        results.add_column(col)
    for m in targets:
        spec = registry.provider_spec(m)
        if get_api_key(spec.api_key_env) is None:
            store.save_verify(m.id, False, "missing_api_key", "API Key 未配置")
            results.add_row(m.id, "[red]跳过[/red]", "-", "缺少 Key")
            continue
        provider = OpenAICompatProvider(spec, settings=settings)
        r = provider.recognize(m, data_url, SYSTEM_PROMPT, probe_prompt, gen)
        store.save_verify(
            m.id, r.ok, r.error_type, r.error_message, r.served_model, r.raw_text
        )
        results.add_row(
            m.id,
            "[green]通过[/green]" if r.ok else f"[red]失败({r.error_type})[/red]",
            r.served_model or "-",
            (r.error_message or "")[:60],
        )
    console.print(results)
    console.print("结果已写入 results/runs.db 的 verify 表（不影响 models.yaml）。")


# ================================================================ collect


@collect_app.command("web")
def collect_web_cmd(
    work: str = typer.Option(..., "--work", help=f"作品 slug：{'/'.join(WORK_SLUGS)}"),
    limit: int = typer.Option(0, "--limit", help="本次采集上限（0=用 settings 上限）"),
):
    """网络图片采集（Bing 搜索 + data/url_lists/{slug}.txt 清单）。"""
    from .collectors import collect_web

    works = {w.slug: w for w in load_works()}
    if work not in works:
        _fail(f"未知作品：{work}（可选 {list(works)}）")
    settings = load_settings()
    report = collect_web(works[work], settings, RAW_DIR / work, limit=limit or None)
    table = Table(title=f"《{works[work].name}》网络采集报告")
    for col in ["命中", "下载", "过小跳过", "重复跳过", "失败"]:
        table.add_column(col, justify="right")
    table.add_row(
        str(report.hits), str(report.downloaded), str(report.skipped_small),
        str(report.skipped_duplicate), str(report.failed),
    )
    console.print(table)
    for err in report.errors[:5]:
        console.print(f"  [dim]{err}[/dim]")
    if report.failed:
        console.print("[yellow]部分来源下载失败属预期（反爬/失效链接）；可补充 data/url_lists 手动清单或投放 data/manual_import。[/yellow]")


@collect_app.command("video")
def collect_video_cmd(
    work: str = typer.Option(..., "--work"),
    video: Path = typer.Option(..., "--video", help="本地视频文件路径"),
    strategy: str = typer.Option("scene", "--strategy", help="scene|interval"),
    interval: float = typer.Option(10.0, "--interval", help="interval 策略的秒数"),
    max_frames: int = typer.Option(200, "--max-frames"),
):
    """本地视频抽帧（需 ffmpeg 或 pip install "vcrb[video]"）。"""
    from .collectors import extract_frames
    from .collectors.video_frames import FfmpegNotAvailable

    works = {w.slug: w for w in load_works()}
    if work not in works:
        _fail(f"未知作品：{work}")
    try:
        report = extract_frames(
            video_path=video,
            out_dir=RAW_DIR / work,
            strategy=strategy,
            interval_seconds=interval,
            max_frames=max_frames,
        )
    except FfmpegNotAvailable as exc:
        _fail(str(exc))
    if report.errors:
        console.print(f"[red]抽帧失败：[/red]{report.errors[0][:300]}")
        raise typer.Exit(1)
    console.print(f"完成：{report.frames} 帧已写入 {RAW_DIR / work}")


@collect_app.command("import")
def collect_import_cmd(
    work: str = typer.Option(..., "--work", help="扫描 data/manual_import/{work}/ 下手动投放的图片"),
):
    """手动投放通道：把 data/manual_import/{slug}/ 的图片登记进 raw（source_type=manual）。"""
    import shutil
    from datetime import datetime, timezone

    src_dir = paths.MANUAL_IMPORT_DIR / work
    if not src_dir.exists():
        _fail(f"目录不存在：{src_dir}")
    files = [p for p in src_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not files:
        _fail("目录为空")
    raw_dir = RAW_DIR / work
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = raw_dir / "manifest.jsonl"
    count = 0
    with manifest.open("a", encoding="utf-8") as fh:
        for src in sorted(files):
            dst = raw_dir / f"manual_{src.name}"
            shutil.copyfile(src, dst)
            fh.write(json.dumps({
                "file": dst.name,
                "source_type": "manual",
                "source_url": str(src),
                "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "license_note": "手动投放，使用者自行确认权利",
            }, ensure_ascii=False) + "\n")
            count += 1
    console.print(f"已登记 {count} 张（已复制到 {raw_dir}）。原目录保留，可清空避免重复导入。")


# ================================================================ roster


@roster_app.command("show")
def roster_show(work: str = typer.Option(..., "--work")):
    from .annotation.rosters import load_roster

    works = {w.slug: w for w in load_works()}
    if work not in works:
        _fail(f"未知作品：{work}")
    roster = load_roster(works[work])
    table = Table(title=f"《{works[work].name}》角色名单（{len(roster.characters)}）")
    for col in ["角色", "重要度", "别名", "演员", "状态"]:
        table.add_column(col)
    for c in roster.characters:
        table.add_row(
            c.name, c.importance, "、".join(c.aliases) or "—", "、".join(c.actors) or "—",
            "⚠️待确认" if c.needs_review else "✅",
        )
    console.print(table)


@roster_app.command("build")
def roster_build(work: str = typer.Option(..., "--work")):
    """生成空白名单模板（真实名单在 config/rosters/ 中维护或经调研填充）。"""
    from .annotation.rosters import build_template

    works = {w.slug: w for w in load_works()}
    if work not in works:
        _fail(f"未知作品：{work}")
    path = build_template(works[work])
    console.print(f"模板已生成：{path}（请人工核实后填写，不确定条目保留 needs_review: true）")


# ================================================================ clean / annotate / export


@app.command("clean")
def clean_cmd(
    work: Optional[str] = typer.Option(None, "--work", help="不指定则清洗全部作品"),
):
    """清洗流水线：有效性→去重→质量→data/candidates/{slug}/。"""
    from .dataset.cleaning import clean_work

    settings = load_settings()
    works = load_works()
    selected = [w for w in works if work in (None, w.slug)]
    table = Table(title="清洗报告")
    for col in ["作品", "原始", "保留", "无效", "质量剔除", "精确重复", "近重复"]:
        table.add_column(col, justify="right")
    for w in selected:
        report = clean_work(w.slug, RAW_DIR / w.slug, CANDIDATES_DIR / w.slug, settings)
        table.add_row(
            w.name, str(report.total), str(report.kept), str(report.invalid),
            str(report.quality_rejected), str(report.exact_duplicates),
            str(report.near_duplicates),
        )
    console.print(table)
    console.print("下一步：vcrb annotate 进入人工标注。")


@app.command("annotate")
def annotate_cmd():
    """启动本地标注页面（Streamlit）。"""
    cmd = [sys.executable, "-m", "streamlit", "run", str(Path(__file__).parent / "annotation" / "app.py"), "--server.headless", "true"]
    console.print(f"[cyan]启动标注页面：[/cyan]{' '.join(cmd)}")
    subprocess.run(cmd, check=False)


@app.command("export-dataset")
def export_dataset_cmd(
    target_per_group: int = typer.Option(20, "--target", help="配额目标（仅记录到 dataset.json）"),
):
    """导出匿名化正式测试集（仅 annotation_status=verified 样本）。"""
    from .dataset.store import AnnotationStore, export_dataset

    store = AnnotationStore()
    verified = store.verified()
    if not verified:
        _fail("没有任何 verified 样本，请先在标注页确认。")
    stats = store.stats()
    for slug in WORK_SLUGS:
        n = stats.get(slug, {}).get("verified", 0)
        if n < 20:
            console.print(f"[yellow]《{slug}》仅 {n} 张已确认（目标 100），将如实导出并记录缺口。[/yellow]")
    path = export_dataset(store, load_works(), FINAL_DIR, target_per_group=target_per_group)
    payload = json.loads(path.read_text(encoding="utf-8"))
    console.print(f"[green]已导出 {payload['total']} 样本 → {path}[/green]")


@app.command("validate-dataset")
def validate_dataset_cmd():
    """校验正式集合法性 + 配额达成 + 防泄露检查。"""
    from .dataset.schema import load_dataset, validate_dataset

    if not DATASET_JSON.exists():
        _fail(f"缺少 {DATASET_JSON}，请先 export-dataset")
    samples = load_dataset(DATASET_JSON)
    errors, stats = validate_dataset(samples, DATASET_JSON.parent)
    table = Table(title="数据集统计")
    for col in ["作品", "样本", "主角", "重要配角", "普通配角", "复杂", "多人"]:
        table.add_column(col, justify="right")
    for slug, bucket in stats["by_work"].items():
        table.add_row(
            slug, str(bucket["total"]), str(bucket.get("main_character", 0)),
            str(bucket.get("key_supporting_character", 0)),
            str(bucket.get("supporting_character", 0)),
            str(bucket.get("hard_case", 0)), str(bucket.get("multi_person", 0)),
        )
    console.print(table)
    # 防泄露检查
    leak = [
        s.sample_id
        for s in samples
        if s.target_character and s.target_character in s.image_path
    ]
    if leak:
        errors.append(f"文件名泄露角色名：{leak[:5]}")
    if errors:
        for e in errors[:20]:
            console.print(f"[red]✗ {e}[/red]")
        raise typer.Exit(1)
    console.print("[green]✓ 校验通过（schema/图片存在/无文件名泄露）[/green]")


# ================================================================ run


def _resolve_models(
    registry: ModelRegistry,
    models: Optional[str],
    vendors: Optional[str],
) -> list[ModelSpec]:
    if models:
        wanted = [m.strip() for m in models.split(",") if m.strip()]
        try:
            specs = [registry.get(m) for m in wanted]
        except ConfigError as exc:
            _fail(str(exc))
            raise  # unreachable
        ineligible = [m.id for m in specs if not m.is_vision_capable]
        if ineligible:
            _fail(
                f"以下模型不支持图片输入，禁止视觉评测：{ineligible}"
                "（见 config/models.yaml 的 supports_vision 标记）"
            )
        return specs
    runnable = registry.runnable_models()
    if vendors:
        wanted = {v.strip() for v in vendors.split(",")}
        runnable = [m for m in runnable if m.vendor in wanted]
    return runnable


def _build_provider(model: ModelSpec, registry: ModelRegistry, settings: dict) -> VisionProvider:
    if model.provider == "mock":
        return MockProvider()
    spec = registry.provider_spec(model)
    return OpenAICompatProvider(spec, settings=settings)


@app.command("run")
def run_cmd(
    models: Optional[str] = typer.Option(None, "--models", help="逗号分隔模型 ID（缺省=全部启用）"),
    vendors: Optional[str] = typer.Option(None, "--vendors", help="按厂商过滤：DeepSeek,Kimi,GLM,Qwen"),
    works: Optional[str] = typer.Option(None, "--works", help="逗号分隔作品 slug"),
    experiment: str = typer.Option("both", "--experiment", help="A|B|both"),
    pilot: bool = typer.Option(False, "--pilot", help="正式预检：每作品 10 张分层抽样"),
    per_work: int = typer.Option(0, "--per-work", help="每作品抽取 N 张（0=全部）"),
    small: bool = typer.Option(False, "--small", help="小规模预检：每作品 2 张"),
    retry_failed: bool = typer.Option(False, "--retry-failed", help="重试 api_failure 记录"),
    yes: bool = typer.Option(False, "--yes", help="确认执行真实 API 调用（付费！需已获授权）"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="续跑指定 run"),
    label: str = typer.Option("", "--label", help="备注标签"),
    limit: int = typer.Option(0, "--limit", help="总样本上限（0=不限）"),
):
    """执行评测（默认 dry-run；--yes 才真实调用）。"""
    from .dataset.schema import load_dataset
    from .runner.executor import dataset_hash, execute_run, select_pilot_samples
    from .runner.store import RunStore, new_run_id

    exps = {"A": ["A"], "B": ["B"], "both": ["A", "B"], "a": ["A"], "b": ["B"]}.get(experiment.lower())
    if not exps:
        _fail("--experiment 取值：A|B|both")

    if not DATASET_JSON.exists():
        _fail(f"缺少 {DATASET_JSON}。流程：collect → clean → annotate → export-dataset")
    samples = load_dataset(DATASET_JSON)
    if works:
        wanted = {w.strip() for w in works.split(",")}
        samples = [s for s in samples if s.work_slug in wanted]
    if per_work > 0 or pilot or small:
        n = per_work or (2 if small else 10)
        samples = select_pilot_samples(samples, per_work=n)
    if limit > 0:
        samples = samples[:limit]
    if not samples:
        _fail("样本为空")

    registry = load_registry()
    settings = load_settings()
    model_specs = _resolve_models(registry, models, vendors)
    if not model_specs:
        _fail("没有匹配的模型（检查 --models/--vendors 与 models.yaml enabled）")
    # Key 预检
    skipped = []
    tasks = []
    for m in model_specs:
        if m.provider == "mock":
            tasks.append((m, _build_provider(m, registry, settings)))
            continue
        spec = registry.provider_spec(m)
        if get_api_key(spec.api_key_env) is None:
            skipped.append(f"{m.id}（缺 {spec.api_key_env}）")
        else:
            tasks.append((m, _build_provider(m, registry, settings)))

    store = RunStore()
    resume = run_id and store.get_run(run_id)
    active_run_id = run_id if resume else new_run_id()
    existing_ok = store.existing_ok_keys(active_run_id) if resume else set()
    existing_failed = store.failed_keys(active_run_id) if resume else set()

    todo = len(tasks) * len(exps) * len(samples)
    if resume:
        todo = sum(
            1
            for m, _ in tasks
            for e in exps
            for s in samples
            if (m.id, e, s.sample_id) not in existing_ok
            and (retry_failed or (m.id, e, s.sample_id) not in existing_failed)
        )

    table = Table(title=f"执行计划（run={active_run_id}{' 续跑' if resume else ''}）")
    for col in ["模型", "实验", "样本", "预计调用"]:
        table.add_column(col)
    for m, _ in tasks:
        for e in exps:
            table.add_row(m.id, e, str(len(samples)), str(len(samples)))
    console.print(table)
    console.print(f"样本来源：{DATASET_JSON}（{len(samples)} 张，hash 前 8 位 {dataset_hash(samples)[:8]}）")
    if skipped:
        console.print(f"[yellow]以下模型缺少 Key 已跳过：{skipped}[/yellow]")
    if todo == 0 and resume:
        console.print("[green]该 run 已全部完成，无待执行任务。[/green]")
        return
    if not yes:
        console.print(
            f"[cyan]dry-run：共 {todo} 次真实调用待执行。确认已获授权后加 --yes 执行。[/cyan]"
        )
        return

    works_by_slug = {w.slug: w for w in load_works()}
    if not resume:
        from .prompts import PROMPT_VERSION
        from .runner.executor import _fairness_meta

        store.create_run(
            run_id=active_run_id,
            experiments=exps,
            models=[m.id for m, _ in tasks],
            prompt_version=PROMPT_VERSION,
            dataset_hash=dataset_hash(samples),
            dataset_path=str(DATASET_JSON),
            fairness_meta=_fairness_meta([m for m, _ in tasks], settings),
            label=label,
        )
    with console.status("[bold green]执行中…（Ctrl-C 可中断，重跑自动续传）[/bold green]"):
        summary = execute_run(
            run_id=active_run_id,
            tasks=tasks,
            samples=samples,
            works_by_slug=works_by_slug,
            experiments=exps,
            store=store,
            settings=settings,
            dataset_dir=DATASET_JSON.parent,
            retry_failed=retry_failed,
        )
    color = "green" if not summary.interrupted else "yellow"
    console.print(
        f"[{color}]完成：执行 {summary.executed}（成功 {summary.ok} / API失败 {summary.api_failures}），"
        f"跳过已完成 {summary.skipped_existing}[/{color}]"
    )
    if summary.errors:
        for e in summary.errors[:10]:
            console.print(f"[red]{e}[/red]")
    console.print(f"下一步：vcrb score --run-id {active_run_id}")


# ================================================================ score / report


@app.command("score")
def score_cmd(
    run_id: Optional[str] = typer.Option(None, "--run-id", help="缺省=最近 run"),
):
    """评分（只读 raw 记录，绝不重新调用 API；可反复重评）。"""
    from .annotation.rosters import load_roster
    from .dataset.schema import load_dataset
    from .evaluation.matcher import CharacterMatcher
    from .evaluation.scoring import save_score_result, score_run
    from .runner.store import RunStore

    store = RunStore()
    rid = run_id or store.latest_run_id()
    if not rid or not store.get_run(rid):
        _fail("没有可评分的 run")
    if not DATASET_JSON.exists():
        _fail("缺少 dataset.json")
    samples = load_dataset(DATASET_JSON)
    run = store.get_run(rid)
    from .runner.executor import dataset_hash as _dh

    if run["dataset_hash"] != _dh(samples):
        console.print("[yellow]警告：数据集与该 run 快照不一致，评分可能失真。[/yellow]")
    works = {w.slug: w for w in load_works()}
    rosters = {slug: load_roster(w) for slug, w in works.items()}
    matcher = CharacterMatcher(rosters, fuzzy_threshold=load_settings()["scoring"]["fuzzy_review_threshold"])
    result = score_run(store, rid, samples, matcher, load_settings())
    out = SCORES_DIR / rid
    files = save_score_result(result, out)
    console.print(f"[green]评分完成 → {out}[/green]")
    for f in files.values():
        console.print(f"  {f}")
    needs = [r for r in result.records if r.needs_review]
    if needs:
        console.print(f"[yellow]{len(needs)} 条相似匹配待人工复核：vcrb review list --run-id {rid}[/yellow]")


@app.command("report")
def report_cmd(
    run_id: Optional[str] = typer.Option(None, "--run-id", help="缺省=最近 run"),
):
    """生成图表 + Markdown 报告 + HTML 错误案例浏览器。"""
    from .annotation.rosters import load_roster
    from .dataset.schema import dataset_stats, load_dataset
    from .evaluation.matcher import CharacterMatcher
    from .evaluation.scoring import score_run
    from .reports.charts import render_charts
    from .reports.html_report import render_html_report
    from .reports.markdown_report import render_markdown_report
    from .runner.store import RunStore

    store = RunStore()
    rid = run_id or store.latest_run_id()
    if not rid or not store.get_run(rid):
        _fail("没有可出报告的 run")
    if not DATASET_JSON.exists():
        _fail("缺少 dataset.json")
    samples = load_dataset(DATASET_JSON)
    settings = load_settings()
    works = {w.slug: w for w in load_works()}
    rosters = {slug: load_roster(w) for slug, w in works.items()}
    matcher = CharacterMatcher(rosters, fuzzy_threshold=settings["scoring"]["fuzzy_review_threshold"])
    result = score_run(store, rid, samples, matcher, settings)
    registry = load_registry()
    models_info = {
        m.id: {
            "vendor": m.vendor,
            "tier": m.tier,
            "display_name": m.display_name,
            "official_series": m.official_series,
        }
        for m in registry.all_models()
    }
    run = store.get_run(rid)
    out_dir = REPORTS_DIR / rid
    charts = render_charts(result, out_dir)
    md = render_markdown_report(
        result,
        out_dir / "report.md",
        dataset_stats=dataset_stats(samples),
        models_info=models_info,
        run_meta=run,
        overrides=[
            {"model_id": m, "experiment": e, "sample_id": s, **v}
            for (m, e, s), v in store.overrides(rid).items()
        ],
        chart_names=[p.name for p in charts],
    )
    html_path = render_html_report(result, store, DATASET_JSON.parent, out_dir / "errors.html")
    console.print(f"[green]报告已生成 → {out_dir}[/green]")
    console.print(f"  Markdown：{md}\n  图表：{len(charts)} 张\n  错误浏览器：{html_path}")


# ================================================================ review


@review_app.command("list")
def review_list(
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    all_errors: bool = typer.Option(False, "--all", help="列出全部错误（不只是待复核）"),
):
    from .annotation.rosters import load_roster
    from .dataset.schema import load_dataset
    from .evaluation.matcher import CharacterMatcher
    from .evaluation.review import error_records, needs_review_records
    from .evaluation.scoring import score_run
    from .runner.store import RunStore

    store = RunStore()
    rid = run_id or store.latest_run_id()
    if not rid or not store.get_run(rid):
        _fail("没有可复核的 run")
    if not DATASET_JSON.exists():
        _fail("缺少 dataset.json")
    samples = load_dataset(DATASET_JSON)
    settings = load_settings()
    works = {w.slug: w for w in load_works()}
    matcher = CharacterMatcher(
        {slug: load_roster(w) for slug, w in works.items()},
        fuzzy_threshold=settings["scoring"]["fuzzy_review_threshold"],
    )
    result = score_run(store, rid, samples, matcher, settings)
    rows = error_records(result) if all_errors else needs_review_records(result)
    if not rows:
        console.print("[green]没有待复核项。[/green]")
        return
    table = Table(title=f"待复核（{len(rows)} 条）")
    for col in ["模型", "实验", "样本", "正确", "预测", "当前判定", "说明"]:
        table.add_column(col)
    for r in rows[:50]:
        table.add_row(
            r["model_id"], r["experiment"], r["sample_id"], r["gt_character"],
            r["predicted"] or "（空）", r["outcome_label"], r["detail"][:40],
        )
    console.print(table)


@review_app.command("set")
def review_set(
    run_id: str = typer.Option(..., "--run-id"),
    model: str = typer.Option(..., "--model"),
    experiment: str = typer.Option(..., "--experiment"),
    sample: str = typer.Option(..., "--sample"),
    outcome: str = typer.Option(..., "--outcome", help="correct|wrong_name|actor_name|uncertain|refused|parse_error|api_failure"),
    note: str = typer.Option("", "--note"),
):
    from .evaluation.review import set_override
    from .runner.store import RunStore

    store = RunStore()
    set_override(store, run_id, model, experiment.upper(), sample, outcome, note)
    console.print("[green]已保存改判。重新 vcrb score 后生效。[/green]")


# ================================================================ status


@app.command("status")
def status_cmd():
    """四作品数据准备进度 + 运行历史 + 模型验证状态。"""
    from .dataset.schema import SAMPLING_GROUPS
    from .dataset.store import AnnotationStore
    from .runner.store import RunStore

    works = load_works()
    table = Table(title="数据准备进度")
    for col in ["作品", "raw", "candidates", "已确认", "无效", "待标注", "待确认"]:
        table.add_column(col, justify="right")
    store = None
    try:
        store = AnnotationStore()
        stats = store.stats()
    except Exception:
        stats = {}
    for w in works:
        def _count(folder: Path, prefix: str | None = None) -> int:
            if not folder.exists():
                return 0
            return len(
                [
                    p
                    for p in folder.iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
                    and (prefix is None or p.name.startswith(prefix))
                ]
            )

        raw = _count(RAW_DIR / w.slug)
        cand = _count(CANDIDATES_DIR / w.slug, prefix="cand_")
        bucket = stats.get(w.slug, {})
        table.add_row(
            w.name, str(raw), str(cand),
            str(bucket.get("verified", 0)), str(bucket.get("invalid", 0)),
            str(bucket.get("pending", 0)), str(bucket.get("uncertain", 0)),
        )
    console.print(table)
    if DATASET_JSON.exists():
        payload = json.loads(DATASET_JSON.read_text(encoding="utf-8"))
        console.print(f"正式集：{payload['total']} 样本（{DATASET_JSON}）")
    else:
        console.print("[yellow]正式集未导出[/yellow]")

    try:
        run_store = RunStore()
        runs = run_store.conn.execute(
            "SELECT run_id, created_at, label, status, note FROM runs ORDER BY created_at DESC LIMIT 10"
        ).fetchall()
        if runs:
            t2 = Table(title="最近运行")
            for col in ["Run ID", "时间", "标签", "状态", "进度"]:
                t2.add_column(col)
            for r in runs:
                prog = run_store.run_progress(r["run_id"])
                t2.add_row(
                    r["run_id"], r["created_at"][:19], r["label"] or "—", r["status"],
                    f"{prog['ok']}ok/{prog['failed']}fail/{prog['total']}total",
                )
            console.print(t2)
        verify = run_store.latest_verify()
        if verify:
            t3 = Table(title="模型连通性验证")
            for col in ["模型", "结果", "实际版本", "时间"]:
                t3.add_column(col)
            for mid, v in sorted(verify.items()):
                t3.add_row(
                    mid, "✓" if v["ok"] else f"✗ {v['error_type'] or ''}",
                    v["served_model"] or "-", (v["checked_at"] or "")[:19],
                )
            console.print(t3)
    except Exception as exc:  # 无数据库时静默
        console.print(f"[dim]（运行记录不可用：{exc}）[/dim]")


def main() -> None:  # python -m vcrb
    app()


if __name__ == "__main__":
    main()
