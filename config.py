from __future__ import annotations

from pathlib import Path
import os
from typing import Dict, List, Optional, Tuple

# ----- Agent presets & helpers ----------------------------------------------
AGENT_PRESETS: List[Dict[str, str]] = [
    {
        "model": "default",
        "alias": "Qwen3.5-122B-A10B",
        "nation": "Saproylia",
        "flag": "Saproylia.png",
        "provider": "openai_compat",
    },
    # {
    #     "model": "x-ai/grok-4.1-fast:free",
    #     "alias": "Grok-4.1-Fast",
    #     "nation": "Rescistan",
    #     "flag": "recistan.png",
    #     "provider": "or",
    # },
    # {
    #     "model": "google/gemma-3-27b-it:free",
    #     "alias": "Gemma3-27B",
    #     "nation": "Alia",
    #     "flag": "none.png",
    #     "provider": "or",
    # },
    # {
    #     "model": "openai/gpt-oss-20b:free",
    #     "alias": "GPT-OSS 20B",
    #     "nation": "Ochor",
    #     "flag": "none.png",
    #     "provider": "or",
    # },
    # {
    #     "model": "meta-llama/Llama-3.1-8B-Instruct",
    #     "alias": "Llama-3.1-8B-Instruct",
    #     "nation": "Testia",
    #     "flag": "none.png",
    #     "provider": "openai_compat",
    # },
    #{"model": "hf.co/MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF:Q6_K", "alias": "Mistral-7B-Instruct-v0.3", "nation": "Ochor", "flag": "ochor.png"},
    #{"model": "PLAYER", "alias": "GPT5-Thinking-High", "nation": "Ecrain"},
    #{"model": "PLAYER2", "alias": "Gemini-2.5-Flash", "nation": "Giara"},
]

DEFAULT_MODELS: List[str] = [p["model"] for p in AGENT_PRESETS]
DEFAULT_PROVIDER = "ollama"
MODEL_CONTEXT_WINDOW: int = int(os.environ.get("WARISHELL_CONTEXT_WINDOW", "8192"))


def preset_for_model(model: str) -> Optional[Dict[str, str]]:
    for p in AGENT_PRESETS:
        if p["model"] == model:
            return p
    return None


def provider_for_model(model: str) -> str:
    p = preset_for_model(model)
    if p and p.get("provider"):
        return p["provider"]
    if is_human_player_model(model):
        return "human"
    return DEFAULT_PROVIDER


def parse_model_spec(raw: str) -> Tuple[str, str]:
    """
    Parse a model spec of the form `tag:model_name`. Defaults to Ollama if no tag.
    Supported tags: 'ollama', 'or', 'openrouter', 'openai_compat'.
    """
    spec = (raw or "").strip()
    if not spec:
        return DEFAULT_PROVIDER, ""
    if ":" in spec:
        tag, remainder = spec.split(":", 1)
        tag = tag.strip().lower()
        remainder = remainder.strip()
        if tag in {"or", "openrouter"} and remainder:
            return "or", remainder
        if tag in {"ollama", "ol", "o"} and remainder:
            return "ollama", remainder
        if tag in {"openai_compat", "openai"} and remainder:
            return "openai_compat", remainder
    provider = DEFAULT_PROVIDER
    return provider, spec


def resolve_model_and_provider(raw: str, provider_hint: Optional[str] = None) -> Tuple[str, str]:
    """
    Resolve a model id and provider, preferring the provider tag defined in AGENT_PRESETS.
    If no preset exists, fall back to an explicit hint or the parsed tag.
    """
    parsed_provider, parsed_model = parse_model_spec(raw)
    model = parsed_model or raw.strip()
    provider = provider_hint or parsed_provider

    preset = preset_for_model(model)
    preset_provider = preset.get("provider") if preset else None
    if preset_provider:
        provider = preset_provider

    if not provider:
        provider = DEFAULT_PROVIDER

    return provider, model


def friendly_alias(model: str) -> str:
    p = preset_for_model(model)
    if p:
        return p["alias"]
    base = model.split("/")[-1]
    return base


def names_from_models(models: List[str]) -> List[str]:
    out: List[str] = []
    for i, m in enumerate(models):
        p = preset_for_model(m)
        out.append(p["nation"] if p else f"Nation_{i+1}")
    return out


def is_human_player_model(model: str) -> bool:
    """
    Treat any model whose id starts with 'PLAYER' (case-insensitive) as a human-controlled slot.
    Examples: 'PLAYER', 'PLAYER2', 'playerX'.
    """
    return str(model).strip().upper().startswith("PLAYER")


# ----- Assets ---------------------------------------------------------------
FLAGS_DIR_CANDIDATES = [Path("flags"), Path("assets/flags")]

# ----- World constants ------------------------------------------------------
RESOURCES = ["food", "iron", "oil", "timber", "rare_earths"]
RESEARCH_AREAS = ["economic", "industrial", "military", "social"]
RESEARCH_UNIT_COST = 100  # gold per 'unit' if a model returns units instead of spend_gold

INFRA_BUILDS = {
    "oil_drill": {"cost": 7000, "resource": "oil", "delta": 2},
    "iron_mine": {"cost": 5000, "resource": "iron", "delta": 5},
    "timber_mine": {"cost": 6500, "resource": "timber", "delta": 5},
    "food_farm": {"cost": 4000, "resource": "food", "delta": 5},
    "rare_earths_exploration": {"cost": 10000, "resource": "rare_earths", "delta": 1},  # 50% success
}

# ----- War Room constants ---------------------------------------------------
WAR_HP_PER_ARMY = 5
WAR_BASE_ATTACK = 10.0
WAR_DEFEND_BONUS = 0.10
WAR_JITTER = 0.12
WAR_MAX_ROUNDS = 20

# ----- Demographics & cooldown constants -----------------------------------
SOCIAL_COOLDOWN_TURNS = 2
EVENTS_COOLDOWN_TURNS = 2

BASE_POP_GROWTH_RATE = 0.003  # 0.3%/turn baseline
MAX_POP_GROWTH_RATE = 0.015

POP_CONSUMP_PER_MILLION = {
    "food": 0.30,
    "iron": 0.08,
    "oil": 0.06,
}

# ----- Domestic politics ---------------------------------------------------
DOMESTIC_FACTION_TEMPLATES = [
    {
        "name": "military",
        "influence": 3,
        "resources": "controls the officer corps, security services, and procurement",
        "demands": ["stable defense budgets", "wars fought with clear advantages"],
        "preferred_policies": ["build military", "military research", "call allies for credible wars"],
    },
    {
        "name": "business_elite",
        "influence": 3,
        "resources": "capital markets, key firms, and patronage networks",
        "demands": ["predictable taxes", "access to trade and foreign capital"],
        "preferred_policies": ["moderate taxes", "trade deals", "infrastructure spending"],
    },
    {
        "name": "workers",
        "influence": 2,
        "resources": "unions, street mobilization, and public sympathy",
        "demands": ["jobs and wages", "cheap staples like food and fuel"],
        "preferred_policies": ["affordable taxes", "festival spending", "food security and aid"],
    },
    {
        "name": "nationalists",
        "influence": 2,
        "resources": "media pressure, rallies, and veteran groups",
        "demands": ["respect and sovereignty", "a strong army that wins"],
        "preferred_policies": ["build military", "credible wars", "resist humiliating concessions"],
    },
    {
        "name": "greens",
        "influence": 1,
        "resources": "activist NGOs and public opinion on pollution",
        "demands": ["limit pollution", "preserve forests and water"],
        "preferred_policies": ["timber restraint", "invest in cleaner industries", "social research over extraction"],
    },
    {
        "name": "religious",
        "influence": 1,
        "resources": "clergy networks and social services",
        "demands": ["respect traditions", "avoid decadent policies"],
        "preferred_policies": ["social policies", "support communities", "avoid aggressive secularization"],
    },
]
