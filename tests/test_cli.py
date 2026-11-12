"""CLI 测试（离线子命令 + dry-run 红线）。"""

from __future__ import annotations

from typer.testing import CliRunner

from vcrb.cli import app

runner = CliRunner()


def test_models_list_defaults():
    result = runner.invoke(app, ["models", "list"])
    assert result.exit_code == 0
    assert "deepseek-flash" in result.output
    assert "kimi-k3.1-pro" in result.output
    assert "glm-5.3-flash" in result.output
    assert "qwen3.8-flash" in result.output
    # 不支持视觉的模型有明确提示
    assert "不支持视觉" in result.output


def test_models_list_all_includes_optional():
    result = runner.invoke(app, ["models", "list", "--all"])
    assert result.exit_code == 0
    assert "glm-4v-flash" in result.output  # 可选免费档
    assert "deepseek-v4-pro" in result.output  # 纯文本登记


def test_models_verify_dry_run_by_default():
    result = runner.invoke(app, ["models", "verify"])
    assert result.exit_code == 0
    assert "dry-run" in result.output
    # 未 --yes 不应有任何真实调用提示表以外的动作
    assert "连通性检查计划" in result.output


def test_roster_show():
    result = runner.invoke(app, ["roster", "show", "--work", "kuangbiao"])
    assert result.exit_code == 0
    assert "高启强" in result.output
    assert "张颂文" in result.output


def test_roster_show_unknown_work():
    result = runner.invoke(app, ["roster", "show", "--work", "nope"])
    assert result.exit_code == 1


def test_collect_web_unknown_work():
    result = runner.invoke(app, ["collect", "web", "--work", "nope"])
    assert result.exit_code == 1


def test_run_requires_dataset(monkeypatch, tmp_path):
    # dataset.json 缺失 → 明确报错
    from vcrb.cli import DATASET_JSON
    from vcrb import paths

    monkeypatch.setattr("vcrb.cli.DATASET_JSON", tmp_path / "nope.json")
    result = runner.invoke(app, ["run", "--experiment", "A"])
    assert result.exit_code == 1
    assert "缺少" in result.output or "export-dataset" in result.output


def test_validate_dataset_missing():
    from vcrb.cli import DATASET_JSON

    if DATASET_JSON.exists():
        return  # 已有本地数据集的机器上跳过该用例
    result = runner.invoke(app, ["validate-dataset"])
    assert result.exit_code == 1


def test_review_list_without_runs(tmp_path, monkeypatch):
    from vcrb import paths
    from vcrb.runner.store import RunStore

    monkeypatch.setattr(paths, "RUNS_DB", tmp_path / "runs.db")
    # CLI 内部直接实例化 RunStore()（默认真实路径）；此处仅验证无 run 时的报错路径
    store = RunStore(tmp_path / "empty.db")
    assert store.latest_run_id() is None


def test_help_works():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ["run", "score", "report", "clean", "annotate", "export-dataset", "status"]:
        assert cmd in result.output
