"""作品归属核验与素材清洗（screening）。

背景：关键词图片搜索的候选池混入了其他作品海报、宣传拼贴、漫画/二创等
污染素材。本包在人工批量审核前先做两级筛查：

1. 证据规则（rules.py）——来源 URL / 页面标题 / 可信域 / 拉黑域。
   只有「可信域 + 作品标识」等来源证据才能给出 verified；
   纯视觉信号永远不能单独给出 verified（用户要求 5）。
2. 视觉启发式（visual.py）——灰度/黑白图（动画类→漫画截图嫌疑）、
   网格拼贴、竖版海报、非正片画幅、插画留白等明显异常标记。

本包与被评测的四家厂商模型完全隔离（不 import providers/openai），
不发起任何模型 API 调用；任何自动判定都可以被人工覆盖（审计日志
attribution_log 记录每次变更的 actor 与原因）。
"""

from .rules import classify_sample, evidence_tokens
from .service import backfill_evidence, report_screening, run_screening
from .visual import VisualReport, analyze_image

__all__ = [
    "analyze_image",
    "backfill_evidence",
    "classify_sample",
    "evidence_tokens",
    "report_screening",
    "run_screening",
    "VisualReport",
]
