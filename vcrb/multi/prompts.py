"""Multi-person v2 prompts. No identities, boxes, or counts enter task C."""

from __future__ import annotations

from typing import Any

PROMPT_VERSION = "multi-v2-principal-abc-2"
SYSTEM_PROMPT = (
    "你是影视画面分析助手。请识别影视角色，不要回答演员真实姓名。"
    "无法确定身份时返回JSON null，不要猜测或列出多个候选。只输出一个JSON对象。"
)
AB_RULES = """请识别图片中每个红色编号方框对应人物的影视角色姓名。按照编号逐个回答，不要遗漏或添加编号。
无法确认时character返回null，不要猜测，不要列出多个候选。每个编号只回答一次。
只输出JSON：{"persons":[{"person_id":1,"character":null}]}。示意编号不是目标数量，须以图片实际编号为准。"""
C_RULES = """请自主找出画面中清晰可辨、占据主要位置或参与主要对话的影视角色，并分别给出位置和影视角色姓名。
本轮只识别主要角色，每张最多返回3人，不凑人数。虚焦背景、远处群众或侍卫、只剩衣物肢体的残缺旁人不属于本轮目标。
按画面主体、清晰度和剧情重要性选择人物；主要人物身份无法确认时仍返回位置并将character设为null，不因不认识姓名而遗漏明显主体。
每个实际人物只返回一次，不按角色名合并不同位置的人物；不要列出多个身份候选。
bbox必须是归一化[x1,y1,x2,y2]，左上角为原点，x向右、y向下，满足0<=x1<x2<=1、0<=y1<y2<=1；框住完整可见人物范围。
只输出JSON：{"persons":[{"bbox":[0.1,0.1,0.4,0.9],"character":null}]}。
示意坐标和条目仅说明格式，不代表画面位置或人物数量；若没有符合范围的人物，返回{"persons":[]}。"""


def build_user_prompt(experiment: str, work: Any = None, person_ids: list[int] | None = None) -> str:
    experiment = experiment.upper()
    if experiment not in {"A", "B", "C"}:
        raise ValueError(f"Unknown multi-person experiment: {experiment}")
    if experiment == "A":
        if work is not None:
            raise ValueError("Experiment A must not include work information")
        return AB_RULES
    if work is None:
        raise ValueError(f"Experiment {experiment} requires work information")
    name = work.get("name", work.get("work_name")) if isinstance(work, dict) else work.name
    media = work.get("media_type", "影视作品") if isinstance(work, dict) else work.media_type
    # person_ids intentionally never enter any prompt, especially C.
    return f"这张图片出自{media}《{name}》。\n" + (C_RULES if experiment == "C" else AB_RULES)
