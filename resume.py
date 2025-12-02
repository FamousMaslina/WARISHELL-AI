from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config import (
    DEFAULT_MODELS,
    friendly_alias,
    preset_for_model,
    resolve_model_and_provider,
    provider_for_model,
)
from models import Agent, Alliance, Country, DomesticFaction, Faction, Research, World, build_domestic_factions


TURN_DIR_RE = re.compile(r"^turn_(\d{3})$")


def find_resume_point(out_dir: Path) -> Optional[Tuple[int, Path]]:
    if not out_dir.exists():
        return None
    best: Tuple[int, Optional[Path]] = (-1, None)
    for p in out_dir.iterdir():
        if p.is_dir():
            m = TURN_DIR_RE.match(p.name)
            if m:
                n = int(m.group(1))
                w = p / "world.json"
                if w.exists() and n > best[0]:
                    best = (n, w)
    return best if best[0] >= 0 else None


def load_world_snapshot(world_path: Path) -> Tuple[World, Dict[str, str]]:
    data = json.loads(world_path.read_text(encoding="utf-8"))
    countries: Dict[str, Country] = {}
    for name, c in data["countries"].items():
        research = Research(**c.get("research", {}))
        if c.get("domestic_factions"):
            domestic = [DomesticFaction(**f) for f in c.get("domestic_factions", [])]
        else:
            domestic = build_domestic_factions(anchor=c.get("stability", 60))
        countries[name] = Country(
            name=c["name"],
            surface_km2=c["surface_km2"],
            production=c["production"],
            stock=c["stock"],
            gold=c["gold"],
            army=c.get("army", 0),
            stability=c.get("stability", 60),
            stability_prev=c.get("stability_prev", c.get("stability", 60)),
            research=research,
            loans_out={k: tuple(v) for k, v in c.get("loans_out", {}).items()},
            loans_in={k: tuple(v) for k, v in c.get("loans_in", {}).items()},
            population=c.get("population", 10_000_000),
            tax_rate=c.get("tax_rate", 0.20),
            tax_rate_prev=c.get("tax_rate_prev", c.get("tax_rate", 0.20)),
            last_events_turn=c.get("last_events_turn", -999),
            last_social_turn=c.get("last_social_turn", -999),
            social_growth_bonus=c.get("social_growth_bonus", 0.0),
            overlord=c.get("overlord"),
            puppet_since_turn=c.get("puppet_since_turn"),
            wars_won=c.get("wars_won", 0),
            wars_lost=c.get("wars_lost", 0),
            last_war_turn=c.get("last_war_turn", -1),
            last_war_result=c.get("last_war_result"),
            last_war_against=c.get("last_war_against"),
            econ_income_pct_active=c.get("econ_income_pct_active", 0.0),
            econ_income_pct_next=c.get("econ_income_pct_next", 0.0),
            domestic_factions=domestic,
            faction_tension=c.get("faction_tension", 0.0),
        )
    world = World(
        countries=countries,
        alliances=[Alliance(**a) for a in data.get("alliances", [])],
        factions=[Faction(**f) for f in data.get("factions", [])],
        wars=[tuple(w) for w in data.get("wars", [])],
        turn=data.get("turn", 1),
        news=data.get("news", []),
        war_log=data.get("war_log", []),
    )
    for c in world.countries.values():
        c.recompute_stability()
    model_map = {
        name: {
            "model": c.get("model"),
            "provider": c.get("model_provider")
            or provider_for_model(c.get("model", "")),
        }
        for name, c in data["countries"].items()
    }
    return world, model_map


def attach_agents_from_map(world: World, model_map: Dict[str, Optional[str]], fallback_models: List[str]) -> List[Agent]:
    agents: List[Agent] = []
    fb_iter = iter(fallback_models or DEFAULT_MODELS)
    for cname in world.countries.keys():
        entry = model_map.get(cname) or {}
        model = entry.get("model")
        provider = entry.get("provider")
        if not model:
            try:
                spec = next(fb_iter)
            except StopIteration:
                spec = fallback_models[-1] if fallback_models else DEFAULT_MODELS[-1]
            provider, model = resolve_model_and_provider(spec)
        else:
            provider, model = resolve_model_and_provider(model, provider_hint=provider)
        p = preset_for_model(model)
        flag = p.get("flag") if p else None
        agents.append(
            Agent(
                country=cname,
                model=model,
                alias=friendly_alias(model),
                flag=flag,
                provider=provider,
            )
        )
    return agents


def load_memories(history_dir: Path, agents: List[Agent], cap: int) -> None:
    if not history_dir.exists():
        return
    by_name = {a.country: a for a in agents}
    for p in history_dir.glob("*.txt"):
        name = p.stem
        if name in by_name:
            lines = p.read_text(encoding="utf-8").splitlines()
            by_name[name].memory_private = lines[-cap:]
