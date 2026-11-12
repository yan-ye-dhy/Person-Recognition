"""统一提示词模板（A/B 两组仅差「作品名称」一句，其余逐字相同）。

公平性约束（用户批准的调整 6）：
- 所有模型使用同一份模板与同一画框标记；
- 不向任何模型提供角色名单、集数、剧情背景；
- A 组不提供作品名称，B 组仅在首句给出作品类型与名称。
"""

from __future__ import annotations

from .config import Work

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "你是影视画面分析助手。你的任务是根据给定的影视截图判断画面中被标注人物的身份。"
    "只输出一个 JSON 对象，不要输出任何解释、前缀或代码块标记。"
)

_TARGET_RULES = """请识别这张影视画面中红色方框标注的人物的影视角色姓名。如果画面中没有红色方框，请识别画面中最主要、最清晰的人物。
要求：
1. 只回答该人物在影视作品中出演的角色姓名，不要回答扮演该角色的演员的真实姓名。
2. 如果无法确定角色姓名，请将 character_name 设为 null，并将 is_uncertain 设为 true。
3. reason 字段用一句话说明你的判断依据。
输出格式（只输出如下 JSON 对象）：
{"character_name": "角色姓名或null", "is_uncertain": false, "reason": "简短依据"}"""

WORK_PREFIX_TEMPLATE = "这张截图来自{media_type}《{work_name}》。\n"


def build_user_prompt(experiment: str, work: Work | None = None) -> str:
    """experiment ∈ {A, B}。B 组必须提供 work；A 组不得提供任何作品信息。"""
    experiment = experiment.upper()
    if experiment not in {"A", "B"}:
        raise ValueError(f"未知实验组：{experiment}")
    if experiment == "A":
        if work is not None:
            raise ValueError("实验 A 不得携带作品信息（防泄露）")
        return _TARGET_RULES
    if work is None:
        raise ValueError("实验 B 需要提供作品信息")
    return WORK_PREFIX_TEMPLATE.format(media_type=work.media_type, work_name=work.name) + _TARGET_RULES


def build_messages(experiment: str, work: Work | None = None) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(experiment, work)},
    ]


def describe_marking() -> str:
    """画框标记说明（写入 run 元数据，保证可追溯）。"""
    return (
        "目标人物以红色实线方框标注（PIL 绘制，线宽 max(3, 宽//200)），"
        "A/B 两组与所有模型使用同一函数与同一线宽规则；无 bbox 的样本不画框，"
        "提示词统一说明『无红框时识别画面主要人物』。"
    )
