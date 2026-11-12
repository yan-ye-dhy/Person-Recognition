"""统一目录约定。

所有运行时数据都落在项目根目录下的 data/ 与 results/（均被 .gitignore 忽略）。
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG_DIR = ROOT / "config"
ROSTERS_DIR = CONFIG_DIR / "rosters"

# ---- 数据侧 ----
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # 采集落地（网络下载 / 抽帧）
CANDIDATES_DIR = DATA_DIR / "candidates"  # 清洗去重后待标注
FINAL_DIR = DATA_DIR / "final"          # 匿名化后的正式测试集
ANNOTATIONS_DB = DATA_DIR / "annotations.db"
DATASET_JSON = FINAL_DIR / "dataset.json"
MANUAL_IMPORT_DIR = DATA_DIR / "manual_import"  # 手动投放图片目录（采集受阻时的兜底通道）

# ---- 结果侧 ----
RESULTS_DIR = ROOT / "results"
RUNS_DB = RESULTS_DIR / "runs.db"
RAW_RESPONSES_DIR = RESULTS_DIR / "raw"    # 每次请求的原始响应 JSONL，按 run 分目录
SCORES_DIR = RESULTS_DIR / "scores"
REPORTS_DIR = RESULTS_DIR / "reports"


def ensure_dirs() -> None:
    """创建全部运行时目录（幂等）。"""
    for d in (
        DATA_DIR,
        RAW_DIR,
        CANDIDATES_DIR,
        FINAL_DIR,
        MANUAL_IMPORT_DIR,
        RESULTS_DIR,
        RAW_RESPONSES_DIR,
        SCORES_DIR,
        REPORTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def raw_dir_for_work(slug: str) -> Path:
    return RAW_DIR / slug


def candidates_dir_for_work(slug: str) -> Path:
    return CANDIDATES_DIR / slug


def final_dir_for_work(slug: str) -> Path:
    return FINAL_DIR / slug
