"""模型注册表：从 config/models.yaml 加载，供所有模块统一查询。

supports_vision 语义（与 models.yaml 头注释一致）：
  doc        官方文档确认支持图片输入
  none       官方确认不支持（禁止视觉评测）
  unverified 存疑，禁止进入正式评测
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import ConfigError, config_dir, load_yaml

VALID_VISION_STATUS = {"doc", "none", "unverified"}
VALID_TIERS = {
    "flagship",
    "flagship_fast",
    "standard",
    "flash",
    "legacy",
    "text",
}


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    base_url: str
    api_key_env: str


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    vendor: str
    display_name: str
    official_series: str
    tier: str
    supports_vision: str
    enabled: bool
    notes: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def is_vision_capable(self) -> bool:
        return self.supports_vision == "doc"

    @property
    def passes_temperature(self) -> bool:
        return bool(self.params.get("pass_temperature", True))

    @property
    def extra_body(self) -> dict[str, Any]:
        value = self.params.get("extra_body") or {}
        return dict(value)


class ModelRegistry:
    def __init__(
        self,
        providers: dict[str, ProviderSpec],
        models: list[ModelSpec],
    ) -> None:
        self.providers = providers
        self.models = models
        self._by_id = {m.id: m for m in models}

    # ---- 加载 ----
    @classmethod
    def from_yaml(cls, path) -> "ModelRegistry":
        raw = load_yaml(path)
        providers: dict[str, ProviderSpec] = {}
        for name, cfg in (raw.get("providers") or {}).items():
            missing = {"base_url", "api_key_env"} - set(cfg or {})
            if missing:
                raise ConfigError(f"providers.{name} 缺少字段 {sorted(missing)}")
            providers[str(name)] = ProviderSpec(
                name=str(name),
                base_url=str(cfg["base_url"]),
                api_key_env=str(cfg["api_key_env"]),
            )
        models: list[ModelSpec] = []
        seen: set[str] = set()
        for item in raw.get("models") or []:
            missing = {
                "id",
                "provider",
                "vendor",
                "display_name",
                "official_series",
                "tier",
                "supports_vision",
                "enabled",
            } - set(item or {})
            if missing:
                raise ConfigError(f"models 条目缺少字段 {sorted(missing)}：{item}")
            model_id = str(item["id"])
            if model_id in seen:
                raise ConfigError(f"模型 ID 重复：{model_id}")
            seen.add(model_id)
            provider = str(item["provider"])
            if provider not in providers:
                raise ConfigError(f"模型 {model_id} 引用了未定义的 provider {provider}")
            vision = str(item["supports_vision"])
            if vision not in VALID_VISION_STATUS:
                raise ConfigError(
                    f"模型 {model_id} supports_vision 非法：{vision}（合法值 {sorted(VALID_VISION_STATUS)}）"
                )
            tier = str(item["tier"])
            if tier not in VALID_TIERS:
                raise ConfigError(
                    f"模型 {model_id} tier 非法：{tier}（合法值 {sorted(VALID_TIERS)}）"
                )
            models.append(
                ModelSpec(
                    id=model_id,
                    provider=provider,
                    vendor=str(item["vendor"]),
                    display_name=str(item["display_name"]),
                    official_series=str(item.get("official_series", "")),
                    tier=tier,
                    supports_vision=vision,
                    enabled=bool(item["enabled"]),
                    notes=str(item.get("notes", "") or ""),
                    params=dict(item.get("params") or {}),
                )
            )
        if not models:
            raise ConfigError("models.yaml 未注册任何模型")
        return cls(providers, models)

    # ---- 查询 ----
    def get(self, model_id: str) -> ModelSpec:
        try:
            return self._by_id[model_id]
        except KeyError:
            raise ConfigError(f"未注册的模型 ID：{model_id}") from None

    def all_models(self) -> list[ModelSpec]:
        return list(self.models)

    def enabled_models(self) -> list[ModelSpec]:
        return [m for m in self.models if m.enabled]

    def runnable_models(self) -> list[ModelSpec]:
        """enabled 且官方文档确认支持视觉的模型 —— 评测候选。"""
        return [m for m in self.models if m.enabled and m.is_vision_capable]

    def vision_ineligible_models(self) -> list[ModelSpec]:
        """不支持视觉的登记模型（用于『不支持视觉测试』清单）。"""
        return [m for m in self.models if m.supports_vision == "none"]

    def unverified_models(self) -> list[ModelSpec]:
        return [m for m in self.models if m.supports_vision == "unverified"]

    def by_vendor(self, vendor: str) -> list[ModelSpec]:
        return [m for m in self.models if m.vendor == vendor]

    def provider_spec(self, model: ModelSpec) -> ProviderSpec:
        return self.providers[model.provider]


def load_registry() -> ModelRegistry:
    return ModelRegistry.from_yaml(config_dir() / "models.yaml")
