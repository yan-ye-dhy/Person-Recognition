"""Local v2 annotation, planning, Mock and separately authorized real execution."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Optional

import typer

from ..config import load_settings, load_works, load_yaml
from ..paths import ROOT, DATASET_JSON
from ..runner.store import RunStore
from ..annotation.rosters import load_roster
from ..evaluation.matcher import CharacterMatcher
from ..imageutils import draw_target_boxes
from .schema import load_dataset, validate_dataset, dataset_stats, dataset_hash
from .runner import MULTI_DB, MULTI_ROOT, resolve_models, select_samples, plan, run_multi
from .migration import migrate_v1

app = typer.Typer(help="v2多人物A/B/C；默认仅计划，Mock零API，真实运行需新授权与费用门禁", no_args_is_help=True)
DEFAULT_DATASET = ROOT / "data/v2/dataset.json"


def emit(value):
    typer.echo(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def exps(value):
    if value.upper() in {"ALL", "ABC"}:
        return ["A", "B", "C"]
    result = value.upper().split(",")
    if set(result) - {"A", "B", "C"} or len(result) != len(set(result)):
        raise ValueError("experiment必须是all或A,B,C的不重复子集")
    return result


@app.command("init")
def initialize(source: Path = typer.Option(DATASET_JSON, "--source"),
               dataset: Path = typer.Option(DEFAULT_DATASET, "--dataset")):
    """从v1继承待复核草稿，拒绝覆盖，不把旧标签自动当作verified。"""
    try:
        payload = migrate_v1(source, dataset)
        emit({"dataset": dataset, "samples": len(payload["samples"]), "quality": dataset_stats(load_dataset(dataset))})
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error))


@app.command("status")
def status(dataset: Path = typer.Option(DEFAULT_DATASET, "--dataset")):
    emit(dataset_stats(load_dataset(dataset)))


@app.command("validate")
def validate(dataset: Path = typer.Option(DEFAULT_DATASET, "--dataset"),
             for_run: bool = typer.Option(False, "--for-run", help="应用正式付费标注门禁")):
    errors, stats = validate_dataset(load_dataset(dataset), dataset.parent, for_run=for_run)
    emit({"dataset": dataset, "formal_gate": for_run, "errors": errors, "stats": stats})
    if errors:
        raise typer.Exit(1)


@app.command("annotate")
def annotate(dataset: Path = typer.Option(DEFAULT_DATASET, "--dataset"),
             port: int = typer.Option(8502, "--port", min=1024, max=65535)):
    """启动仅本机可访问的多框标注工具。"""
    if not dataset.is_file():
        raise typer.BadParameter("请先vcrb multi init")
    emit({"url": f"http://127.0.0.1:{port}/", "dataset": dataset.resolve()})
    result = subprocess.call([sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("annotation_app.py")),
                              "--server.address", "127.0.0.1", "--server.port", str(port),
                              "--browser.gatherUsageStats", "false", "--", "--dataset", str(dataset.resolve())])
    raise typer.Exit(result)


@app.command("preview")
def preview(sample_id: str = typer.Argument(...), dataset: Path = typer.Option(DEFAULT_DATASET, "--dataset"),
            experiment: str = typer.Option("A", "--experiment"), output: Optional[Path] = typer.Option(None, "--output")):
    """生成编号红框或C原图副本；从不覆盖源图。"""
    sample = next((s for s in load_dataset(dataset) if s.sample_id == sample_id), None)
    if sample is None:
        raise typer.BadParameter("未知sample_id")
    exp = exps(experiment)
    if len(exp) != 1:
        raise typer.BadParameter("preview一次选择一个实验")
    source = (dataset.parent / sample.image_path).resolve()
    output = output or MULTI_ROOT / "previews" / f"{sample_id}_{exp[0]}.jpg"
    if output.resolve() == source:
        raise typer.BadParameter("不能覆盖原图")
    if exp[0] == "C":
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output)
    else:
        draw_target_boxes(source, output, sample.persons)
    emit({"preview": output.resolve(), "experiment": exp[0], "numbered": exp[0] != "C"})


@app.command("plan")
def plan_cmd(dataset: Path = typer.Option(DEFAULT_DATASET, "--dataset"),
             models: Optional[str] = typer.Option(None, "--models"),
             experiment: str = typer.Option("all", "--experiment"),
             per_work: int = typer.Option(0, "--per-work", min=0),
             limit: int = typer.Option(0, "--limit", min=0),
             pricing: Optional[Path] = typer.Option(None, "--pricing")):
    samples = select_samples(load_dataset(dataset), per_work, limit)
    result = plan(samples, resolve_models(models.split(",") if models else None), exps(experiment),
                  load_settings(), load_yaml(pricing) if pricing else None)
    path = MULTI_ROOT / "plans/latest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    emit({k: v for k, v in result.items() if k != "annotation_blockers"} | {"full_plan": path})


@app.command("gallery")
def gallery_cmd(dataset: Path = typer.Option(DEFAULT_DATASET, "--dataset"),
                output: Path = typer.Option(MULTI_ROOT / "input_gallery", "--output")):
    """生成全部样本A/B/C实际输入图的本地并排总览，零模型API。"""
    from .gallery import generate_gallery

    try:
        emit(generate_gallery(dataset, output))
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error))


@app.command("run")
def run_cmd(dataset: Path = typer.Option(DEFAULT_DATASET, "--dataset"),
            models: Optional[str] = typer.Option(None, "--models"),
            experiment: str = typer.Option("all", "--experiment"),
            mock: bool = typer.Option(False, "--mock", help="全链路本地Mock，不调用API"),
            yes: bool = typer.Option(False, "--yes", help="真实付费请求的显式确认，仍须通过标注/费用门禁"),
            per_work: int = typer.Option(0, "--per-work", min=0), limit: int = typer.Option(0, "--limit", min=0),
            run_id: Optional[str] = typer.Option(None, "--run-id"),
            retry_failed: bool = typer.Option(False, "--retry-failed"),
            max_requests: Optional[int] = typer.Option(None, "--max-requests", min=0),
            max_cost_cny: Optional[float] = typer.Option(None, "--max-cost-cny"),
            budget_parent_run_id: Optional[str] = typer.Option(None, "--budget-parent-run-id", help="修订运行继承旧真实运行的共享费用上限"),
            pricing: Optional[Path] = typer.Option(None, "--pricing"),
            approval_note: str = typer.Option("", "--approval-note")):
    if not mock and not yes:
        plan_cmd(dataset, models, experiment, per_work, limit, pricing)
        typer.echo("dry-run：未调用任何模型；尚需标注复核、新的真实试跑授权与费用配置。")
        return
    try:
        result = run_multi(dataset, model_ids=models.split(",") if models else None,
                           experiments=exps(experiment), mock=mock, yes=yes,
                           per_work=per_work, limit=limit, run_id=run_id, retry_failed=retry_failed,
                           max_requests=max_requests, max_cost_cny=max_cost_cny,
                           budget_parent_run_id=budget_parent_run_id,
                           pricing=load_yaml(pricing) if pricing else None, approval_note=approval_note)
        brief = dict(result)
        brief["plan"] = {k: v for k, v in result["plan"].items() if k != "annotation_blockers"}
        emit(brief)
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error))


def score_and_report(run_id, iou):
    from .scoring import score_run
    from .reports import save_reports
    store = RunStore(MULTI_DB)
    run = store.get_run(run_id)
    if not run or run["fairness_meta"].get("benchmark_protocol") != "multi-v2":
        raise ValueError("必须指定独立multi-v2 run，不能使用单目标旧响应")
    samples = load_dataset(Path(run["dataset_path"]))
    if dataset_hash(samples) != run["dataset_hash"]:
        raise ValueError("运行快照哈希不一致")
    settings = dict(run["fairness_meta"].get("settings_snapshot") or load_settings())
    settings["multi"] = {"scoring": {"iou_threshold": iou, "ignore_overlap_threshold": .5}}
    matcher = CharacterMatcher({w.slug: load_roster(w) for w in load_works()})
    result = score_run(store, run_id, samples, matcher, settings)
    return save_reports(result, MULTI_ROOT / run_id / "reports", run_meta=run)


@app.command("score")
@app.command("report")
def score_cmd(run_id: str = typer.Option(..., "--run-id"),
              iou: float = typer.Option(.5, "--iou", min=.01, max=1.0)):
    """从v2原响应重评并生成HTML/CSV/JSON，零API。"""
    try:
        emit(score_and_report(run_id, iou))
    except (ValueError, OSError) as error:
        raise typer.BadParameter(str(error))


@app.command("demo")
def demo():
    """独立合成8图、7型号×ABC的168任务Mock演示；不是实际模型能力。"""
    from .mock import create_demo_dataset
    dataset = create_demo_dataset(MULTI_ROOT / "demo_fixture")
    result = run_multi(dataset, mock=True)
    files = score_and_report(result["run_id"], .5)
    emit({"mock_only": True, "real_api_calls": 0, "dataset": dataset, "run_id": result["run_id"],
          "planned_tasks": result["planned"], "executed": result["executed"], "reports": files})
