from __future__ import annotations

import dataclasses
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from config import DOMESTIC_FACTION_TEMPLATES


@dataclass
class Faction:
    name: str
    members: List[str]
    secret: bool = True
    created_by: str = ""


@dataclass
class DomesticFaction:
    name: str
    approval: int
    influence: int
    resources: str
    demands: List[str]
    preferred_policies: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class Research:
    economic: int = 0
    industrial: int = 0
    military: int = 0
    social: int = 0

    def as_dict(self) -> Dict[str, int]:
        return dataclasses.asdict(self)


@dataclass
class Country:
    name: str
    surface_km2: int
    production: Dict[str, int]  # per‑turn production of resources
    stock: Dict[str, int]  # current resource stockpiles
    gold: int  # treasury in GOLD
    army: int = 0
    stability: int = 60
    stability_prev: int = 60
    research: Research = field(default_factory=Research)
    loans_out: Dict[str, Tuple[int, float]] = field(default_factory=dict)
    loans_in: Dict[str, Tuple[int, float]] = field(default_factory=dict)
    population: int = 0
    tax_rate: float = 0.20
    tax_rate_prev: float = 0.20
    last_events_turn: int = -999
    last_social_turn: int = -999
    social_growth_bonus: float = 0.0
    overlord: Optional[str] = None
    puppet_since_turn: Optional[int] = None
    wars_won: int = 0
    wars_lost: int = 0
    last_war_turn: int = -1
    last_war_result: Optional[str] = None
    last_war_against: Optional[str] = None
    econ_income_pct_active: float = 0.0
    econ_income_pct_next: float = 0.0
    zero_gold_streak: int = 0
    bankrupt: bool = False
    bankruptcy_turn: Optional[int] = None
    domestic_factions: List[DomesticFaction] = field(default_factory=list)
    faction_tension: float = 0.0

    def public_summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "surface_km2": self.surface_km2,
            "research": self.research.as_dict(),
            "production": self.production,
            "population": self.population,
            "tax_rate": round(self.tax_rate, 3),
            "military_tier": self.army_tier(),
            "stability": self.stability,
            "faction_tension": self.faction_tension,
        }

    def army_tier(self) -> str:
        a = int(self.army)
        if a <= 0:
            return "none"
        if a <= 10:
            return "weak"
        if a <= 20:
            return "average"
        if a <= 35:
            return "strong"
        return "overwhelming"

    def recompute_stability(self) -> None:
        if not self.domestic_factions:
            self.faction_tension = 0.0
            self.stability = max(0, min(100, int(self.stability)))
            return
        weights = [max(1, f.influence) for f in self.domestic_factions]
        total_w = sum(weights)
        avg = sum(f.approval * w for f, w in zip(self.domestic_factions, weights)) / max(1, total_w)
        spread = max(f.approval for f in self.domestic_factions) - min(f.approval for f in self.domestic_factions)
        tension_penalty = spread * 0.35
        self.faction_tension = round(spread, 2)
        self.stability = max(0, min(100, int(round(avg - tension_penalty))))


def build_domestic_factions(rng: Optional[random.Random] = None, anchor: Optional[int] = None) -> List[DomesticFaction]:
    rng = rng or random.Random()
    anchor = 60 if anchor is None else max(20, min(90, int(anchor)))
    factions: List[DomesticFaction] = []
    for tpl in DOMESTIC_FACTION_TEMPLATES:
        wobble = rng.randint(-12, 12)
        base = max(15, min(95, anchor + wobble))
        factions.append(
            DomesticFaction(
                name=tpl["name"],
                approval=base,
                influence=tpl.get("influence", 1),
                resources=tpl.get("resources", ""),
                demands=list(tpl.get("demands", [])),
                preferred_policies=list(tpl.get("preferred_policies", [])),
            )
        )
    return factions


@dataclass
class Alliance:
    members: List[str]
    secret: bool


@dataclass
class World:
    countries: Dict[str, Country]
    alliances: List[Alliance] = field(default_factory=list)
    factions: List[Faction] = field(default_factory=list)
    wars: List[Tuple[str, str]] = field(default_factory=list)
    turn: int = 1
    news: List[str] = field(default_factory=list)
    war_log: List[Dict[str, Any]] = field(default_factory=list)

    def puppets_of(self, name: str) -> List[str]:
        return [cname for cname, c in self.countries.items() if c.overlord == name]

    def country(self, name: str) -> Country:
        return self.countries[name]

    def faction_of(self, name: str) -> Optional[Faction]:
        for f in self.factions:
            if name in f.members:
                return f
        return None

    def public_faction_label(self, name: str) -> Optional[str]:
        f = self.faction_of(name)
        if not f:
            return None
        return f.name if not f.secret else None

    def add_news(self, msg: str) -> None:
        self.news.append(msg)


@dataclass
class Agent:
    country: str
    model: str
    memory_private: List[str] = field(default_factory=list)
    alias: str = ""
    flag: Optional[str] = None
    provider: str = "ollama"
