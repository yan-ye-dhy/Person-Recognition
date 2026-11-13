"""模型注册表测试：加载、校验、查询。"""

from __future__ import annotations

import pytest

from vcrb.config import ConfigError
from vcrb.models import ModelRegistry, load_registry


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return load_registry()


def test_registry_loads(registry: ModelRegistry):
    assert len(registry.all_models()) >= 15
    ids = {m.id for m in registry.all_models()}
    # 每家首批代表模型必须在册
    for expected in [
        "deepseek-flash",
        "kimi-k3.1-pro",
        "kimi-k2.6",
        "kimi-k2.5",
        "glm-5.3-flash",
        "glm-4.6v",
        "glm-4.6v-flash",
        "qwen3.8-max",
        "qwen3.7-plus",
        "qwen3.8-flash",
    ]:
        assert expected in ids, f"缺少已核实的模型 {expected}"


def test_every_model_has_provider_and_base_url(registry: ModelRegistry):
    for model in registry.all_models():
        spec = registry.provider_spec(model)
        assert spec.base_url.startswith("https://")
        assert spec.api_key_env


def test_runnable_models_exclude_text_only(registry: ModelRegistry):
    runnable = {m.id for m in registry.runnable_models()}
    assert "deepseek-v4-pro" not in runnable
    assert "glm-5.3" not in runnable
    assert "deepseek-flash" in runnable
    # 不支持视觉的模型有专门的清单
    ineligible = {m.id for m in registry.vision_ineligible_models()}
    assert "deepseek-v4-pro" in ineligible


def test_enabled_batch_two_to_three_per_vendor(registry: ModelRegistry):
    vendors = {}
    for m in registry.runnable_models():
        vendors.setdefault(m.vendor, []).append(m.id)
    assert set(vendors) == {"DeepSeek", "Kimi", "GLM", "Qwen"}
    for vendor, ids in vendors.items():
        if vendor in {"DeepSeek", "Kimi"}:
            # DeepSeek仅一个视觉型号；本账号K2.5/K2.6实测404，Kimi只跑K3.1。
            assert len(ids) == 1, f"{vendor} 当前可测型号应为1个: {ids}"
        else:
            assert 2 <= len(ids) <= 3, f"{vendor} 首批模型数 {len(ids)} 不在 2-3 范围: {ids}"


def test_kimi_k3_1_pro_registered_with_notes(registry: ModelRegistry):
    model = registry.get("kimi-k3.1-pro")
    assert model.is_vision_capable
    assert model.enabled
    assert "verify" in model.notes or "连通" in model.notes


def test_qwen_models_disable_thinking(registry: ModelRegistry):
    for mid in ["qwen3.8-max", "qwen3.7-plus", "qwen3.8-flash"]:
        model = registry.get(mid)
        assert model.extra_body.get("enable_thinking") is False


def test_kimi_sampling_params_not_passed(registry: ModelRegistry):
    assert not registry.get("kimi-k2.6").passes_temperature
    assert not registry.get("kimi-k2.5").passes_temperature
    assert registry.get("deepseek-flash").passes_temperature


def test_unknown_model_raises(registry: ModelRegistry):
    with pytest.raises(ConfigError):
        registry.get("gpt-not-registered")


def test_duplicate_model_id_rejected(tmp_path):
    yaml_text = """
providers:
  p:
    base_url: https://example.com/v1
    api_key_env: X_KEY
models:
  - id: m1
    provider: p
    vendor: V
    display_name: M1
    official_series: ""
    tier: standard
    supports_vision: doc
    enabled: true
  - id: m1
    provider: p
    vendor: V
    display_name: M1 dup
    official_series: ""
    tier: standard
    supports_vision: doc
    enabled: false
"""
    path = tmp_path / "models.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError, match="重复"):
        ModelRegistry.from_yaml(path)


def test_invalid_vision_status_rejected(tmp_path):
    yaml_text = """
providers:
  p:
    base_url: https://example.com/v1
    api_key_env: X_KEY
models:
  - id: m1
    provider: p
    vendor: V
    display_name: M1
    official_series: ""
    tier: standard
    supports_vision: maybe
    enabled: true
"""
    path = tmp_path / "models.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ConfigError, match="supports_vision"):
        ModelRegistry.from_yaml(path)
