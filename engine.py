from __future__ import annotations

import asyncio
import dataclasses
import difflib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from pydantic import ValidationError
from rich import box
from rich.table import Table

from config import (
    BASE_POP_GROWTH_RATE,
    DEFAULT_PROVIDER,
    EVENTS_COOLDOWN_TURNS,
    INFRA_BUILDS,
    MAX_POP_GROWTH_RATE,
    MODEL_CONTEXT_WINDOW,
    POP_CONSUMP_PER_MILLION,
    RESEARCH_AREAS,
    RESEARCH_UNIT_COST,
    RESOURCES,
    SOCIAL_COOLDOWN_TURNS,
    WAR_BASE_ATTACK,
    WAR_DEFEND_BONUS,
    WAR_HP_PER_ARMY,
    WAR_JITTER,
    WAR_MAX_ROUNDS,
    friendly_alias,
    is_human_player_model,
    preset_for_model,
    resolve_model_and_provider,
)
from console_utils import console
from decisions import (
    AidBid,
    AllianceOrder,
    AllianceVote,
    BuildOrder,
    InfraOrder,
    LoanOrder,
    ModelDecision,
    PolicyOrder,
    PuppetControl,
    ResearchOrder,
    TaxOrder,
    TradeDecision,
    TradeOffer,
    WarDecision,
    WarOrder,
)
from io_utils import RunIO
from market import Market, MarketOrder
from models import Agent, Alliance, Country, Faction, Research, World, build_domestic_factions
from ollama_client import OllamaClient
from openrouter_client import OpenRouterClient
from parser_utils import (
    ALLIANCE_NEWS_RE,
    BUILD_NEWS_RE,
    FACTION_FORM_RE,
    FACTION_LEFT_RE,
    LOAN_NEWS_RE,
    RESEARCH_NEWS_RE,
    THINK_RE,
    TRADE_NEWS_RE,
    VOTE_MEMBER_RE,
    VOTE_OUTCOME_ADMIT_RE,
    VOTE_OUTCOME_DECLINE_RE,
    WAR_CONCLUDED_RE,
    WAR_DECLARED_RE,
    WAR_NEWS_RE,
    extract_and_coerce_decision,
)
from prompts import BASE_SYSTEM, build_user_prompt


def resolve_war(world: World, a: Country, b: Country) -> Tuple[str, str]:
    mult_a = stability_military_multiplier(a.stability)
    mult_b = stability_military_multiplier(b.stability)

    rnd = random.random() * 0.2 + 0.9
    eff_a = a.army * (1 + a.research.military / 100) * mult_a * rnd
    eff_b = b.army * (1 + b.research.military / 100) * mult_b * (2 - rnd)
    if abs(eff_a - eff_b) < 1e-9:
        eff_b *= 1.01

    winner = a.name if eff_a > eff_b else b.name
    loser = b.name if winner == a.name else a.name
    wa, wb = world.country(winner), world.country(loser)

    wa.army = max(0, int(wa.army * 0.85))
    wb.army = max(0, int(wb.army * 0.60))

    gold_take = max(0, int(wb.gold * 0.35))
    wb.gold -= gold_take
    wa.gold += gold_take

    res = random.choice(RESOURCES)
    steal = min(wb.stock.get(res, 0), random.randint(10, 100))
    wb.stock[res] = max(0, wb.stock.get(res, 0) - steal)
    wa.stock[res] = wa.stock.get(res, 0) + steal

    if wa.domestic_factions:
        for f in wa.domestic_factions:
            f.approval = max(0, min(100, f.approval + 7))
        wa.recompute_stability()
    else:
        wa.stability = min(100, wa.stability + 5)

    if wb.domestic_factions:
        for f in wb.domestic_factions:
            f.approval = max(0, min(100, f.approval - 10))
        wb.recompute_stability()
    else:
        wb.stability = max(0, wb.stability - 10)

    for k in RESOURCES:
        wa.production[k] = int(max(0, round(wa.production.get(k, 0) * 1.10)))
    world.add_news(f"Expansion incentive: {winner} production increased by +10% after victory.")

    def label(name: str) -> str:
        f = world.faction_of(name)
        if f:
            return f"[{f.name}]"
        return name

    headline = f"War: {label(a.name)} vs {label(b.name)} → Winner: {winner} (+{gold_take} gold, stole {steal} {res})."
    return winner, headline


def stability_econ_multiplier(s: int) -> float:
    s = max(0, min(100, int(s)))
    if s >= 100:
        return 1.15
    if s >= 50:
        return 1.0 + 0.15 * (s - 50) / 50.0
    if s > 30:
        return 1.0
    steps = 0
    if s <= 30:
        steps += 1
    if s <= 20:
        steps += 1
    if s <= 10:
        steps += 1
    if s == 0:
        steps += 1
    return max(0.4, 1.0 - 0.10 * steps)


def stability_military_multiplier(s: int) -> float:
    if s > 60:
        return 1.05
    if s < 30:
        return 0.95
    return 1.00


class Engine:
    MAX_PRIVATE_MEM_LINES = 320
    MEMORY_LINES_IN_PROMPT = 75

    def __init__(
        self,
        world: World,
        agents: List[Agent],
        out_dir: Path,
        ollama_url: str,
        openrouter_url: str,
        openrouter_key: Optional[str],
        context_window: Optional[int],
        seed: int = 0,
    ):
        self.world = world
        self.agents = {a.country: a for a in agents}
        self.io = RunIO(out_dir)
        ctx = context_window or MODEL_CONTEXT_WINDOW
        self.ollama = OllamaClient(ollama_url, default_ctx=ctx)
        self.openrouter: Optional[OpenRouterClient]
        if openrouter_key:
            self.openrouter = OpenRouterClient(openrouter_url, openrouter_key, default_ctx=ctx)
        else:
            self.openrouter = None
        self.provider_clients = {"ollama": self.ollama}
        if self.openrouter:
            self.provider_clients["or"] = self.openrouter
        self.random = random.Random(seed)

    def _client_for_provider(self, provider: str):
        client = self.provider_clients.get(provider)
        if not client:
            raise RuntimeError(f"No client configured for provider '{provider}'.")
        return client

    async def _stop_all_clients(self) -> None:
        for client in set(self.provider_clients.values()):
            try:
                await client.stop_all()
            except Exception:
                pass

    async def _unload_model(self, provider: str, model: str) -> None:
        client = self.provider_clients.get(provider)
        if not client:
            return
        try:
            await client.unload(model)
        except Exception:
            pass

    async def _chat_model(
        self,
        agent: Agent,
        messages: List[Dict[str, str]],
        options: Optional[Dict[str, Any]] = None,
        stream: bool = False,
    ) -> str:
        return await self._client_for_provider(agent.provider).chat(
            agent.model,
            messages,
            options=options,
            stream=stream,
        )

    @staticmethod
    def _clamp(v: float, lo: float = 0, hi: float = 100) -> int:
        return int(max(lo, min(hi, round(v))))

    def _adjust_faction_approvals(self, c: Country, deltas: Dict[str, float], default: float = 0.0) -> None:
        """Shift domestic faction approvals, then recompute stability."""
        if not c.domestic_factions:
            # Fallback: mimic legacy behaviour by nudging stability directly.
            c.stability = self._clamp(c.stability + (default if default else sum(deltas.values()) / max(1, len(deltas) or 1)))
            return
        name_map = {f.name.lower(): f for f in c.domestic_factions}
        for f in c.domestic_factions:
            delta = deltas.get(f.name.lower(), default)
            f.approval = self._clamp(f.approval + delta)
        c.recompute_stability()

    def _broad_unrest(self, c: Country, delta: float, focus: Optional[List[str]] = None, bleed: float = 0.35) -> None:
        """Convenience for applying unrest/cheer to focused factions and a milder effect to others."""
        focus = [s.lower() for s in focus] if focus else []
        deltas: Dict[str, float] = {}
        for f in c.domestic_factions:
            if not focus:
                deltas[f.name.lower()] = delta
            elif f.name.lower() in focus:
                deltas[f.name.lower()] = delta
            else:
                deltas[f.name.lower()] = delta * bleed
        self._adjust_faction_approvals(c, deltas, default=delta if not deltas else 0.0)

    def _finalise_war_result(self, attacker_name: str, defender_name: str, winner: str, steal_amt: int, best_res: str, gold_take: int, rounds: int, capitulated: bool, cause: str) -> None:
        wa = self.world.country(winner)
        loser = defender_name if winner == attacker_name else attacker_name
        wl = self.world.country(loser)
        wa.wars_won += 1
        wl.wars_lost += 1
        wa.last_war_turn = self.world.turn
        wl.last_war_turn = self.world.turn
        wa.last_war_result = "WON"
        wa.last_war_against = wl.name
        wl.last_war_result = "LOST"
        wl.last_war_against = wa.name
        self.world.war_log.append({
            "turn": self.world.turn,
            "attacker": attacker_name,
            "defender": defender_name,
            "winner": winner,
            "loser": loser,
            "gold": gold_take,
            "res_taken": {"resource": best_res, "qty": steal_amt},
            "rounds": rounds,
            "capitulation": bool(capitulated),
            "cause": cause,
        })

    def _apply_war_resolution_no_battle(self, winner: str, loser: str, cause: str) -> None:
        wa = self.world.country(winner)
        wl = self.world.country(loser)
        gold_take = max(0, int(wl.gold * 0.35))
        wl.gold -= gold_take
        wa.gold += gold_take
        best_res = max(RESOURCES, key=lambda r: wl.stock.get(r, 0))
        steal_amt = min(wl.stock.get(best_res, 0), 50)
        wl.stock[best_res] = max(0, wl.stock.get(best_res, 0) - steal_amt)
        wa.stock[best_res] = wa.stock.get(best_res, 0) + steal_amt
        self._adjust_faction_approvals(wa, {"military": +9, "nationalists": +9}, default=+4)
        self._adjust_faction_approvals(wl, {"military": -10, "nationalists": -10, "business_elite": -6, "workers": -6}, default=-5)
        for k in RESOURCES:
            wa.production[k] = int(max(0, round(wa.production.get(k, 0) * 1.10)))
        headline = (f"War concluded: {winner} vs {loser} → Winner: {winner} "
                    f"(+{gold_take} gold, took {steal_amt} {best_res}; rounds=0, occupation).")
        self.world.add_news(headline)
        self.world.wars.append((winner, loser))
        self._finalise_war_result(winner, loser, winner, steal_amt, best_res, gold_take, 0, False, cause)

    def _research_income_tick(self, c: Country) -> None:
        econ_gain = 10 * c.research.economic
        ind_lvl = c.research.industrial
        ind_gain = self.random.randint(5 * ind_lvl, 30 * ind_lvl) if ind_lvl > 0 else 0
        total = econ_gain + ind_gain
        if total > 0:
            c.gold += total
            self.world.add_news(
                f"Research income: {c.name} +{econ_gain} gold (economic), +{ind_gain} gold (industrial)."
            )

    async def close(self):
        await self.ollama.close()
        if self.openrouter:
            await self.openrouter.close()

    def _update_memories_from_news(self, turn: int) -> None:
        per_country: Dict[str, List[str]] = {name: [] for name in self.world.countries.keys()}

        for line in self.world.news:
            if line.startswith("Bankruptcy:"):
                for who in per_country.keys():
                    per_country[who].append(f"Turn {turn} – {line}")
                continue
            if line.startswith("Aid accepted:"):
                for who in per_country.keys():
                    per_country[who].append(f"Turn {turn} – {line}")
                continue
            m = BUILD_NEWS_RE.match(line)
            if m:
                who, plus = m.groups()
                if who in per_country:
                    per_country[who].append(f"Turn {turn} – Military: expanded armed forces by +{plus}.")
                continue

            m = RESEARCH_NEWS_RE.match(line)
            if m:
                who, area, inc, spend = m.groups()
                if who in per_country:
                    per_country[who].append(f"Turn {turn} – Research: {area} +{inc} (spent {spend}).")
                continue
            m = FACTION_FORM_RE.search(line)
            if m:
                vis, fname = m.groups()
                for who in per_country.keys():
                    f = self.world.faction_of(who)
                    if f and f.name == fname:
                        per_country[who].append(f"Turn {turn} – Faction: joined/exists [{fname}] ({vis}).")
                continue

            m = FACTION_LEFT_RE.search(line)
            if m:
                who, fname = m.groups()
                if who in per_country:
                    per_country[who].append(f"Turn {turn} – Faction: left [{fname}].")
                continue
            m = TRADE_NEWS_RE.match(line)
            if m:
                seller, buyer, qty, res, ppu, total = m.groups()
                if seller in per_country:
                    per_country[seller].append(f"Turn {turn} – Trade: sold {qty} {res} to {buyer} @ {ppu} (total {total}).")
                if buyer in per_country:
                    per_country[buyer].append(f"Turn {turn} – Trade: bought {qty} {res} from {seller} @ {ppu} (total {total}).")
                continue

            m = LOAN_NEWS_RE.match(line)
            if m:
                lender, borrower, gold, rate = m.groups()
                if lender in per_country:
                    per_country[lender].append(f"Turn {turn} – Loan: lent {gold} gold to {borrower} @{rate}%/turn.")
                if borrower in per_country:
                    per_country[borrower].append(f"Turn {turn} – Loan: borrowed {gold} gold from {lender} @{rate}%/turn.")
                continue
            m = WAR_CONCLUDED_RE.match(line)
            if m:
                a, b, winner, gold_take, steal_amt, steal_res = m.groups()
                gold_take = int(gold_take)
                steal_amt = int(steal_amt)
                loser = b if winner == a else a
                if winner in per_country:
                    per_country[winner].append(
                        f"Turn {turn} – War vs {loser}: WON (+{gold_take} gold, captured {steal_amt} {steal_res})."
                    )
                if loser in per_country:
                    per_country[loser].append(
                        f"Turn {turn} – War vs {winner}: LOST (−{gold_take} gold, −{steal_amt} {steal_res})."
                    )
                continue
            m = WAR_DECLARED_RE.match(line)
            if m:
                atk, defn, why = m.groups()
                if atk in per_country:
                    per_country[atk].append(f"Turn {turn} – Declared war on {defn}. Cause: {why}.")
                if defn in per_country:
                    per_country[defn].append(f"Turn {turn} – Was attacked by {atk}. Cause claimed: {why}.")
                continue
            m = WAR_NEWS_RE.match(line)
            if m:
                a, b, winner, gold_take, steal_amt, steal_res, cause = m.groups()
                gold_take = int(gold_take)
                steal_amt = int(steal_amt)
                loser = b if winner == a else a
                if winner in per_country:
                    per_country[winner].append(
                        f"Turn {turn} – War vs {loser}: WON (+{gold_take} gold, captured {steal_amt} {steal_res}, casualties ≈15%)."
                        + (f" Cause: {cause}." if cause else "")
                    )
                if loser in per_country:
                    per_country[loser].append(
                        f"Turn {turn} – War vs {winner}: LOST (−{gold_take} gold, −{steal_amt} {steal_res}, casualties ≈40%)."
                        + (f" Cause: {cause}." if cause else "")
                    )
                continue

            if line.startswith("Infrastructure:"):
                try:
                    who = line.split(": ", 1)[1].split(" ", 1)[0]
                    if who in per_country:
                        per_country[who].append(f"Turn {turn} – {line.split(': ',1)[1]}")
                except Exception:
                    pass
                continue
            if line.startswith("Puppet: "):
                for who in per_country.keys():
                    per_country[who].append(f"Turn {turn} – {line}")
                continue
            if line.startswith("Puppet released:"):
                for who in per_country.keys():
                    per_country[who].append(f"Turn {turn} – {line}")
                continue
            if line.startswith("Annexation:"):
                for who in per_country.keys():
                    per_country[who].append(f"Turn {turn} – {line}")
                continue
            if line.startswith("Tribute: "):
                for who in per_country.keys():
                    per_country[who].append(f"Turn {turn} – {line}")
                continue
            m = ALLIANCE_NEWS_RE.match(line)
            if m:
                vis, members_str = m.groups()
                members = [s.strip() for s in members_str.split(",")]
                for mbr in members:
                    others = [x for x in members if x != mbr]
                    if mbr in per_country:
                        per_country[mbr].append(f"Turn {turn} – Alliance: formed {vis} alliance with {', '.join(others)}.")
                continue
            m = VOTE_MEMBER_RE.match(line)
            if m:
                voter, decision, applicant, faction, reason = m.groups()
                if applicant in per_country:
                    if decision.lower() == "accepted":
                        per_country[applicant].append(
                            f"Turn {turn} – Faction vote: {voter} ACCEPTED your entry into [{faction}] (reason: {reason})."
                        )
                    else:
                        per_country[applicant].append(
                            f"Turn {turn} – Faction vote: {voter} DECLINED your entry into [{faction}] (reason: {reason})."
                        )
                if voter in per_country:
                    if decision.lower() == "accepted":
                        per_country[voter].append(
                            f"Turn {turn} – Faction vote: You ACCEPTED {applicant} into [{faction}] (reason: {reason})."
                        )
                    else:
                        per_country[voter].append(
                            f"Turn {turn} – Faction vote: You DECLINED {applicant} into [{faction}] (reason: {reason})."
                        )
                continue

            m = VOTE_OUTCOME_DECLINE_RE.match(line)
            if m:
                faction, applicant, veto_by, why = m.groups()
                if applicant in per_country:
                    per_country[applicant].append(
                        f"Turn {turn} – Faction: Application to [{faction}] was REJECTED (veto by {veto_by}: {why})."
                    )
                if veto_by in per_country:
                    per_country[veto_by].append(
                        f"Turn {turn} – Faction: You vetoed {applicant}'s entry into [{faction}] (reason: {why})."
                    )
                continue

            m = VOTE_OUTCOME_ADMIT_RE.match(line)
            if m:
                faction, applicant = m.groups()
                if applicant in per_country:
                    per_country[applicant].append(
                        f"Turn {turn} – Faction: You were ADMITTED to [{faction}] (unanimous)."
                    )
                continue
            if line.startswith("Parser:"):
                for name in per_country.keys():
                    if f"{name} " in line or f"{name} –" in line:
                        per_country[name].append(f"Turn {turn} – Parser note: {line.split('–',1)[-1].strip()}")

        history_dir = self.io.root / "history"
        history_dir.mkdir(exist_ok=True)
        for name, lines in per_country.items():
            if not lines:
                c = self.world.country(name)
                lines = [f"Turn {turn} – War record: {c.wars_won}W/{c.wars_lost}L"
                         + (f"; last vs {c.last_war_against}: {c.last_war_result} (Turn {c.last_war_turn})" if c.last_war_turn >= 0 else "")]
            agent = self.agents[name]
            agent.memory_private.extend(lines)
            if len(agent.memory_private) > self.MAX_PRIVATE_MEM_LINES:
                agent.memory_private[:] = agent.memory_private[-self.MAX_PRIVATE_MEM_LINES:]
            with open(history_dir / f"{name}.txt", "a", encoding="utf-8") as fh:
                for ln in lines:
                    fh.write(ln + "\n")

    async def run(self, turns: int) -> None:
        start = self.world.turn + 1
        end = self.world.turn + turns
        for t in range(start, end + 1):
            self.world.turn = t
            console.rule(f"Turn {t}")
            tribute_totals: Dict[str, Dict[str, int]] = {}
            for c in self.world.countries.values():
                c.recompute_stability()
            for c in self.world.countries.values():
                c.stability_prev = c.stability
                c.tax_rate_prev = c.tax_rate
                econ_mult = stability_econ_multiplier(c.stability)
                for r, q in c.production.items():
                    added = int(max(0, round(q * econ_mult)))
                    c.stock[r] = c.stock.get(r, 0) + added
                    if c.overlord:
                        try:
                            overlord = self.world.country(c.overlord)
                            tribute = added // 2
                            if tribute > 0:
                                c.stock[r] = max(0, c.stock[r] - tribute)
                                overlord.stock[r] = overlord.stock.get(r, 0) + tribute
                                tmap = tribute_totals.setdefault(c.name, {k: 0 for k in RESOURCES})
                                tmap[r] += tribute
                        except KeyError:
                            c.overlord = None
                            c.puppet_since_turn = None
                            pass
            for puppet, resmap in tribute_totals.items():
                if not any(resmap.values()):
                    continue
                ov = self.world.country(self.world.country(puppet).overlord) if self.world.country(puppet).overlord else None
                parts = [f"{k}:{v}" for k, v in resmap.items() if v > 0]
                if ov:
                    self.world.add_news(f"Tribute: {puppet}→{ov.name} transferred " + ", ".join(parts) + ".")

            growth_ok: Dict[str, bool] = {}
            for c in self.world.countries.values():
                growth_ok[c.name] = self._apply_population_upkeep(c)

            for c in self.world.countries.values():
                c.econ_income_pct_active = c.econ_income_pct_next
                c.econ_income_pct_next = 0.0

            for c in self.world.countries.values():
                self._collect_tax_income(c)

            for c in self.world.countries.values():
                self._research_income_tick(c)

            for c in self.world.countries.values():
                self._roll_poverty_unrest(c)

            for c in self.world.countries.values():
                if self.random.random() < 0.08:
                    self._broad_unrest(c, -5)
                    self.world.add_news(f"Faction scandal: {c.name} approvals slipped (~−5 each). Stability now {c.stability}.")

            for c in self.world.countries.values():
                if c.bankrupt and c.gold > 0:
                    c.bankrupt = False
                    c.zero_gold_streak = 0
            for c in self.world.countries.values():
                if c.bankrupt and c.bankruptcy_turn == self.world.turn - 1:
                    self.world.add_news(
                        (
                            "Bankruptcy: %s declared insolvency (gold=0 for 2 consecutive turns). "  # noqa: UP030
                            "Nations may submit **aid bids** this turn as "  # noqa: UP030
                            '{"aid_bid":[{"bankrupt":"%s","gold":INT>0,"ask":{"resource":"food|iron|oil|timber|rare_earths","qty":INT>0}}]} '  # noqa: UP030
                            "— the insolvent nation will accept the **least-bad** offer."
                        ) % (c.name, c.name)
                    )
                    console.rule(f"[bold blue]IMF ROOM[/] Insolvency auction opens for {c.name}")
                    console.print(
                        "  Submit aid with: "
                        '{"aid_bid":[{"bankrupt":"%s","gold":G,"ask":{"resource":"food|iron|oil|timber|rare_earths","qty":Q}}]}' % c.name  # noqa: UP030
                    )

            decisions: Dict[str, Tuple[ModelDecision, str, Optional[str]]] = {}
            market_orders: List[MarketOrder] = []
            aid_offers: Dict[str, List[Tuple[str, AidBid]]] = {}
            war_decls: List[Tuple[str, str, str, Optional[str]]] = []
            alliance_reqs: List[Tuple[str, AllianceOrder]] = []
            loan_reqs: List[LoanOrder] = []

            results: List[Tuple[str, str, Optional[str], Optional[Dict[str, Any]]]] = []
            agent_items = list(self.agents.items())
            ai_items = [(c, a) for (c, a) in agent_items if not is_human_player_model(a.model)]
            human_items = [(c, a) for (c, a) in agent_items if is_human_player_model(a.model)]

            for cname, agent in ai_items + human_items:
                cname, raw, think, parsed = await self._ask_agent(cname, agent)
                results.append((cname, raw, think, parsed))
                if not is_human_player_model(agent.model):
                    await self._unload_model(agent.provider, agent.model)
                await asyncio.sleep(0.1)
                turn_dir = self.io.turn_dir(t)
                self.io.save_text(turn_dir / f"{cname}__raw.txt", raw)

                if think:
                    self.io.save_text(turn_dir / f"{cname}__think.txt", think)
                if parsed is None:
                    console.print(f"[yellow]WARN[/] {cname} didn't return valid JSON. Skipping.")
                    continue
                try:
                    dec = ModelDecision.model_validate(parsed)
                except ValidationError as ve:
                    console.print(f"[yellow]WARN[/] {cname} invalid decision: {ve}")
                    continue
                self.io.save_json(turn_dir / f"{cname}__parsed.json", dec.model_dump())
                decisions[cname] = (dec, raw, think)
                console.print(
                    f"[green]DONE[/] {cname} — "
                    f"trade:{len(dec.trade)} "
                    f"build:{len(dec.build)} "
                    f"research:{len(dec.research)} "
                    f"alliance:{len(dec.alliance)} "
                    f"war:{len(dec.war)} "
                    f"loans:{len(dec.loans)} "
                    f"policy:{len(dec.policy) if hasattr(dec, 'policy') else 0}"
                )
                if dec.public_message:
                    self.world.add_news(f"{cname} public message: {dec.public_message}")
                for tr in dec.trade:
                    if tr.resource not in RESOURCES or tr.direction not in ("buy", "sell"):
                        continue
                    market_orders.append(MarketOrder(cname, tr.direction, tr.resource, max(0, tr.qty), max(0, tr.price_per_unit), tr.counterparty))
                for bo in dec.build:
                    self._apply_build(self.world.country(cname), bo)
                for ro in dec.research:
                    self._apply_research(self.world.country(cname), ro)
                for io in getattr(dec, "infrastructure", []):
                    self._apply_infrastructure(self.world.country(cname), io)
                for ao in dec.alliance:
                    alliance_reqs.append((cname, ao))
                for ww in dec.war:
                    war_decls.append((cname, ww.target, ww.cause, getattr(ww, 'goal', None)))
                for lo in dec.loans:
                    loan_reqs.append(lo)
                for ab in getattr(dec, "aid_bid", []):
                    try:
                        tgt = str(ab.bankrupt)
                        if (tgt in self.world.countries
                            and self.world.country(tgt).bankrupt
                            and cname != tgt):
                            aid_offers.setdefault(tgt, []).append((cname, ab))
                        elif cname == tgt:
                            self.world.add_news(f"Parser: {cname} – aid_bid to self is invalid; ignored.")
                    except Exception:
                        self.world.add_news(f"Parser: {cname} – malformed aid_bid skipped.")
                if dec.tax is not None:
                    self._apply_tax(self.world.country(cname), dec.tax)
                for td in getattr(dec, "trade_decision", []):
                    if td.decision == "accept":
                        market_orders.append(MarketOrder(cname, td.direction, td.resource, max(0, td.qty), max(0, td.price_per_unit), td.counterparty))
                    else:
                        self.world.add_news(f"Trade: {cname} declined {td.counterparty}'s {td.direction} of {td.qty} {td.resource} @ {td.price_per_unit} ({td.reason or 'no reason'}).")
                for pol in dec.policy:
                    if pol.organize_events:
                        cc = self.world.country(cname)
                        if (self.world.turn - cc.last_events_turn) >= EVENTS_COOLDOWN_TURNS and cc.gold >= 250:
                            cc.gold -= 250
                            self._broad_unrest(cc, +10, focus=["workers", "religious", "nationalists"], bleed=0.6)
                            cc.last_events_turn = self.world.turn
                            self.world.add_news(f"Domestic: {cname} organized national festivals (−250 gold, factions cheered).")
                        else:
                            left = EVENTS_COOLDOWN_TURNS - (self.world.turn - cc.last_events_turn)
                            if left > 0:
                                self.world.add_news(f"Parser: {cname} – invalid function → organize_events is on cooldown ({left} turn(s) left) ✨.")

            for overlord_name, (dec, _, _) in self._last_decisions.items():
                for pc in getattr(dec, "puppet_control", []):
                    self._apply_puppet_control(overlord_name, pc)
            for tgt, bids in aid_offers.items():
                valid: List[Tuple[str, AidBid]] = []
                for bidder, bid in bids:
                    try:
                        if bid.gold > 0 and self.world.country(bidder).gold >= int(bid.gold):
                            ask_res = str(bid.ask.get("resource", "")).lower()
                            ask_qty = int(bid.ask.get("qty", 0))
                            if ask_res in RESOURCES and ask_qty > 0:
                                valid.append((bidder, bid))
                    except Exception:
                        pass
                if not valid:
                    self.world.add_news(f"Aid window: No valid bids received for {tgt}.")
                    continue
                choice = await self._ask_bankruptcy_choice(tgt, valid)
                if choice is None:
                    weights = {"rare_earths": 5, "oil": 3, "iron": 2, "timber": 1, "food": 1}
                    bidder, bid = min(
                        valid,
                        key=lambda x: (weights[x[1].ask["resource"]] * x[1].ask["qty"]) / max(1, x[1].gold)
                    )
                else:
                    bidder, bid = choice
                self._execute_aid(bidder, tgt, bid)

            market = Market()
            market.clear(self.world, market_orders)
            for d in market.public_deals:
                self.world.add_news(f"Trade: {d['seller']}→{d['buyer']} {d['qty']} {d['resource']} @ {d['ppu']} (total {d['total']}).")

            self._resolve_loans(loan_reqs)

            await self._resolve_alliances(alliance_reqs)

            for attacker, defender, cause, goal in war_decls:
                if attacker not in self.world.countries or defender not in self.world.countries:
                    continue
                if self.world.country(attacker).army <= 0:
                    self.world.add_news(f"War aborted: {attacker} attempted to attack {defender} with no deployable forces.")
                    continue
                await self._war_room(attacker, defender, cause, goal)

            for c in self.world.countries.values():
                if c.tax_rate - c.tax_rate_prev >= 0.10 - 1e-9:
                    drop = self.random.randint(6, 12)
                    self._adjust_faction_approvals(
                        c,
                        {
                            "workers": -drop * 1.1,
                            "business_elite": -drop,
                            "nationalists": -drop * 0.6,
                            "military": -drop * 0.4,
                        },
                        default=-drop * 0.5,
                    )
                    self.world.add_news(
                        f"Civil Unrest (taxation): {c.name} factions bristled after tax rise "
                        f"from {int(c.tax_rate_prev*100)}% to {int(c.tax_rate*100)}%."
                    )
            for c in self.world.countries.values():
                self._apply_population_growth(c, allow_growth=growth_ok.get(c.name, False))

            for c in self.world.countries.values():
                c.recompute_stability()

            turn_dir = self.io.turn_dir(t)
            snap = self._world_snapshot()
            self.io.save_json(turn_dir / "world.json", snap)
            try:
                self.io.save_json(self.io.root / "rt.json", snap)
            except Exception:
                pass
            self._print_table()
            self._update_memories_from_news(t)
            self._accrue_interest()

            for c in self.world.countries.values():
                if c.gold == 0:
                    c.zero_gold_streak += 1
                else:
                    c.zero_gold_streak = 0
                    if c.bankrupt:
                        c.bankrupt = False
                if (not c.bankrupt) and c.zero_gold_streak >= 2:
                    c.bankrupt = True
                    c.bankruptcy_turn = self.world.turn
                    self.world.add_news(f"Bankruptcy: {c.name} declared insolvency (gold=0 for 2 turns). Aid window opens next turn.")
                    console.rule(f"[bold blue]IMF ROOM[/] Insolvency flagged for {c.name}")
                    console.print("  Aid window will open at the start of next turn.")

            self.world.news = []

        console.rule("Simulation complete")

    async def _ask_agent(self, cname: str, agent: Agent) -> Tuple[str, str, Optional[str], Optional[Dict[str, Any]]]:
        c = self.world.country(cname)
        system_prompt = BASE_SYSTEM
        user_prompt = build_user_prompt(self.world, cname)
        if agent.memory_private:
            recent = agent.memory_private[-self.MEMORY_LINES_IN_PROMPT:]
            user_prompt += "\nYour memory across turns (last {} events):\n{}".format(
                len(recent), "\n".join(f"- {s}" for s in recent)
            )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        options = {"temperature": 0.6, "top_p": 0.95, "seed": 1}

        if is_human_player_model(agent.model):
            turn_dir = self.io.turn_dir(self.world.turn)
            sys_path = turn_dir / f"{cname}__prompt_system.txt"
            usr_path = turn_dir / f"{cname}__prompt_user.txt"
            self.io.save_text(sys_path, system_prompt)
            self.io.save_text(usr_path, user_prompt)

            reply_filename = f"{agent.model}_parsed.json"
            reply_path = turn_dir / reply_filename

            console.rule(f"[bold]HUMAN TURN[/] {cname} ({agent.model})")
            console.print("[dim]Exact prompts were also written to:[/]")
            console.print(f"  system → {sys_path}")
            console.print(f"  user   → {usr_path}\n")
            console.print(
                "Based on the above, create a single JSON object matching the schema and save it here:\n"
                f"  [bold]{reply_path}[/]\n"
                "Then press Enter. I will parse/validate it and continue the turn."
            )

            while True:
                try:
                    input("Press Enter after saving your JSON file...")
                except KeyboardInterrupt:
                    console.print("[red]Interrupted[/] — defaulting to skip this model’s actions.")
                    return cname, "", None, None
                if not reply_path.exists():
                    console.print(f"[yellow]File not found[/]: {reply_path}. Please save it and press Enter again.")
                    continue
                try:
                    raw = reply_path.read_text(encoding="utf-8")
                except Exception as e:
                    console.print(f"[red]Read error[/] {e}. Fix the file and press Enter again.")
                    continue

                think, parsed, json_text, notes = extract_and_coerce_decision(raw)
                for note in notes:
                    self.world.add_news(f"Parser: {cname} – {note}")
                if parsed is None:
                    console.print("[yellow]Invalid/empty JSON[/]. Please correct and press Enter again.")
                    continue
                try:
                    _ = ModelDecision.model_validate(parsed)
                except ValidationError as ve:
                    console.print(f"[yellow]Schema validation failed[/]: {ve}\nFix and press Enter again.")
                    continue

                self.io.save_text(turn_dir / f"{cname}__raw.txt", raw)
                try:
                    self.io.save_json(turn_dir / f"{cname}__parsed.json", parsed)
                except Exception:
                    pass
                self.world.add_news(f"Parser: {cname} – human input accepted.")
                return cname, raw, None, parsed

        try:
            # Linux VRAM workaround: clear any lingering models before loading the next one.
            await self._stop_all_clients()
        except Exception:
            pass

        provider_label = "OpenRouter" if agent.provider == "or" else "Ollama"
        try:
            console.print(
                f"[green]ENGINE:[/] Called model [{agent.alias or friendly_alias(agent.model)}] ({agent.model}) "
                f"via {provider_label} to reply for nation [{cname}]"
            )
            raw = await self._chat_model(agent, messages, options=options, stream=False)
        except httpx.HTTPStatusError as e:
            detail = ""
            if e.response is not None:
                try:
                    detail = e.response.text.strip()
                except Exception:
                    detail = ""
            extra = f"\n{provider_label} response: {detail}" if detail else ""
            raw = f"ERROR contacting {provider_label} for {cname}: {e}{extra}"
        except httpx.HTTPError as e:
            raw = f"ERROR contacting {provider_label} for {cname}: {e}"

        think, parsed, json_text, notes = extract_and_coerce_decision(raw)
        for note in notes:
            self.world.add_news(f"Parser: {cname} – {note}")
        if parsed is None:
            self.world.add_news(f"Parser: {cname} – invalid JSON; regenerating once with identical prompt.")
            try:
                raw2 = await self._chat_model(agent, messages, options=options, stream=False)
            except httpx.HTTPStatusError as e:
                detail = ""
                if e.response is not None:
                    try:
                        detail = e.response.text.strip()
                    except Exception:
                        detail = ""
                extra = f"\n{provider_label} response: {detail}" if detail else ""
                raw2 = f"ERROR contacting {provider_label} for {cname} on retry: {e}{extra}"
            except httpx.HTTPError as e:
                raw2 = f"ERROR contacting {provider_label} for {cname} on retry: {e}"
            think2, parsed2, json_text2, notes2 = extract_and_coerce_decision(raw2)
            for note in notes2:
                self.world.add_news(f"Parser: {cname} – {note}")
            try:
                turn_dir = self.io.turn_dir(self.world.turn)
                self.io.save_text(turn_dir / f"{cname}__raw_attempt1.txt", raw)
                self.io.save_text(turn_dir / f"{cname}__raw_attempt2.txt", raw2)
                diff_txt = "\n".join(
                    difflib.unified_diff(
                        raw.splitlines(),
                        raw2.splitlines(),
                        fromfile=f"{cname}__raw_attempt1.txt",
                        tofile=f"{cname}__raw_attempt2.txt",
                        lineterm=""
                    )
                )
                if diff_txt.strip():
                    self.io.save_text(turn_dir / f"{cname}__regen_diff.patch", diff_txt + "\n")
            except Exception:
                pass
            if parsed2 is not None:
                self.world.add_news(f"Parser: {cname} – regeneration succeeded.")
                return cname, raw2, think2, parsed2
            else:
                self.world.add_news(f"Parser: {cname} – regeneration failed; skipping this model’s actions this turn.")
                return cname, raw, think, None
        return cname, raw, think, parsed

    def _apply_build(self, c: Country, bo: BuildOrder) -> None:
        ok = True
        for r, q in bo.use.items():
            if c.stock.get(r, 0) < q:
                ok = False
                break
        if c.gold < bo.gold_cost:
            ok = False
        if not ok:
            return
        for r, q in bo.use.items():
            c.stock[r] -= q
        c.gold -= bo.gold_cost
        c.army += max(0, bo.unit_power)
        self.world.add_news(f"{c.name} expanded armed forces by +{bo.unit_power}.")
        self._adjust_faction_approvals(c, {"military": 4, "nationalists": 2}, default=0.5)

    def _apply_research(self, c: Country, ro: ResearchOrder) -> None:
        if ro.area not in RESEARCH_AREAS:
            return
        if ro.area == "social":
            if (self.world.turn - c.last_social_turn) < SOCIAL_COOLDOWN_TURNS:
                left = SOCIAL_COOLDOWN_TURNS - (self.world.turn - c.last_social_turn)
                self.world.add_news(f"Parser: {c.name} – invalid function → social research is on cooldown ({left} turn(s) left) 💗.")
                return
            spend = ro.spend_gold
            if spend is None and ro.units is not None:
                spend = max(0, int(ro.units)) * 100
            if not spend or spend < 100:
                return
            spend = min(spend, c.gold)
            if spend < 100:
                return
            c.gold -= spend
            units100 = spend // 100
            c.research.social += units100
            c.social_growth_bonus += 0.001 * units100
            c.last_social_turn = self.world.turn
            pct = round(units100 * 0.1, 1)
            self.world.add_news(f"Social policy: {c.name} invested {spend} gold (+{pct}% pop growth; cooldown {SOCIAL_COOLDOWN_TURNS} turns).")
            self._adjust_faction_approvals(c, {"workers": 3, "religious": 2}, default=0.5)
            return

        spend = ro.spend_gold
        if spend is None and ro.units is not None:
            spend = max(0, int(ro.units)) * RESEARCH_UNIT_COST
        if not spend or spend <= 0:
            return
        spend = min(spend, c.gold)
        if spend == 0:
            return

        if ro.area == "economic":
            X = spend
            if X < 100:
                bonus_pct = (X / 100.0) * 0.5
            else:
                bonus_pct = (X / 100.0) ** 1.25
            c.gold -= spend
            c.econ_income_pct_next += bonus_pct
            self.world.add_news(
                f"Economic policy: {c.name} invested {spend} gold; next-turn income modifier +{bonus_pct:.2f}%."
            )
            self._adjust_faction_approvals(c, {"business_elite": 4}, default=0.5)
            return
        if ro.area == "industrial":
            delta = 2 if spend >= 500 else (1 if spend >= 250 else 0)
            if delta <= 0:
                return
            c.gold -= spend
            choices = ["food", "iron", "oil", "timber"]
            res = random.choice(choices)
            c.production[res] = int(c.production.get(res, 0)) + delta
            self.world.add_news(
                f"Industrial R&D: {c.name} boosted production {res} by +{delta}/turn (spent {spend})."
            )
            self._adjust_faction_approvals(c, {"business_elite": 3, "workers": 2, "greens": -2}, default=0.3)
            return
        c.gold -= spend
        incr = int(math.sqrt(spend))
        c.research.military += incr
        self.world.add_news(f"{c.name} advanced military research (+{incr}, spent {spend}).")
        self._adjust_faction_approvals(c, {"military": 5, "nationalists": 4, "greens": -2}, default=0.5)

    def _apply_infrastructure(self, c: Country, io: InfraOrder) -> None:
        t = io.type
        n = max(1, int(io.count))
        if t not in INFRA_BUILDS:
            self.world.add_news(f"Parser: {c.name} – unknown infrastructure '{t}'.")
            return
        spec = INFRA_BUILDS[t]
        faction_effects = {
            "oil_drill": {"business_elite": 4, "workers": 2, "greens": -4},
            "iron_mine": {"business_elite": 3, "workers": 2, "greens": -2},
            "timber_mine": {"business_elite": 3, "workers": 2, "greens": -3},
            "food_farm": {"workers": 4, "business_elite": 1, "greens": -1},
            "rare_earths_exploration": {"business_elite": 4, "nationalists": 2, "greens": -3},
        }
        for _ in range(n):
            cost = spec["cost"]
            if c.gold < cost:
                self.world.add_news(f"Parser: {c.name} – insufficient gold for {t} (needs {cost}).")
                break
            c.gold -= cost
            if t == "rare_earths_exploration":
                if random.random() < 0.5:
                    c.production["rare_earths"] = int(c.production.get("rare_earths", 0)) + 1
                    self.world.add_news(f"Infrastructure: {c.name} invested 10000 gold into rare_earths discovery. It succeeded. +1 rare_earths/turn.")
                    self._adjust_faction_approvals(c, faction_effects.get(t, {}), default=0.3)
                else:
                    self.world.add_news(f"Infrastructure: {c.name} invested 10000 gold into rare_earths discovery. It failed.")
                    self._adjust_faction_approvals(c, {"business_elite": -2, "greens": -1}, default=-0.2)
            else:
                res = spec["resource"]
                d = spec["delta"]
                c.production[res] = int(c.production.get(res, 0)) + d
                nice = t.replace("_", " ")
                self.world.add_news(f"Infrastructure: {c.name} built {nice} (+{d} {res}/turn, −{cost} gold).")
                self._adjust_faction_approvals(c, faction_effects.get(t, {}), default=0.3)

    def _apply_tax(self, c: Country, to: TaxOrder) -> None:
        target = max(0, min(50, int(to.set_rate))) / 100.0
        c.tax_rate = target

    def _collect_tax_income(self, c: Country) -> None:
        pop_m = max(1, c.population // 1_000_000)
        base_per_million = 8
        econ_boost = 1.0 + (c.research.economic / 200.0)
        stab_mult = stability_econ_multiplier(c.stability)
        income = int(round(pop_m * base_per_million * c.tax_rate * econ_boost * stab_mult))
        if c.econ_income_pct_active > 0.0:
            income = int(round(income * (1.0 + (c.econ_income_pct_active / 100.0))))
        if income > 0:
            c.gold += income
            self.world.add_news(
                f"Tax: {c.name} collected {income} gold (tax {int(c.tax_rate*100)}%, pop {pop_m}M"
                + (f", econ boost +{c.econ_income_pct_active:.2f}%" if c.econ_income_pct_active > 0 else "")  # noqa: PIE804
                + ")."
            )

    async def _ask_bankruptcy_choice(self, bankrupt: str, offers: List[Tuple[str, AidBid]]) -> Optional[Tuple[str, AidBid]]:
        """
        Ask the bankrupt nation's model to choose the least-bad offer.
        Returns (bidder, AidBid) or None if parsing fails.
        """
        agent = self.agents[bankrupt]
        c = self.world.country(bankrupt)
        present = []
        for i, (bidder, bid) in enumerate(offers):
            present.append({
                "index": i,
                "from": bidder,
                "gold": int(bid.gold),
                "ask": {"resource": bid.ask["resource"], "qty": int(bid.ask["qty"])} if isinstance(bid.ask, dict) else {},
            })
        schema = {"aid_choice": {"index": "INT one of the listed indices"}}
        prompt = (
            "INSOLVENCY AID — You are bankrupt (gold=0 for 2 turns). "
            "Pick the **least-bad** offer (minimise strategic harm; prefer keeping scarce/critical resources):\n"
            f"Your stocks: {json.dumps(c.stock, ensure_ascii=False)}\n"
            f"Offers: {json.dumps(present, ensure_ascii=False)}\n"
            "Return ONLY: " + json.dumps(schema, ensure_ascii=False)
        )
        messages = [
            {"role": "system", "content": 'Select the least-bad aid offer. Respond with ONLY JSON matching {"aid_choice":{"index":INT}}.'},
            {"role": "user", "content": prompt},
        ]
        try:
            raw = await self._chat_model(agent, messages, options={"temperature":0.3,"top_p":0.9}, stream=False)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            obj = json.loads(m.group(0)) if m else {}
            idx = int(obj.get("aid_choice", {}).get("index", -1))
            if 0 <= idx < len(offers):
                return offers[idx]
        except Exception:
            pass
        return None

    def _execute_aid(self, bidder: str, target: str, bid: AidBid) -> None:
        if bidder == target:
            self.world.add_news(f"Aid aborted: self-bid by {bidder}.")
            return

        B = self.world.country(bidder)
        T = self.world.country(target)
        gold = int(max(0, bid.gold))
        res = str(bid.ask["resource"]).lower()
        qty = int(max(0, bid.ask["qty"]))
        if gold <= 0 or res not in RESOURCES or qty <= 0:
            self.world.add_news(f"Aid aborted: invalid bid terms from {bidder} to {target}.")
            return
        if B.gold < gold:
            self.world.add_news(f"Aid aborted: {bidder} lacked gold for promised bailout to {target}.")
            return
        B.gold -= gold
        T.gold += gold
        give = min(qty, T.stock.get(res, 0))
        T.stock[res] = max(0, T.stock.get(res, 0) - give)
        B.stock[res] = B.stock.get(res, 0) + give
        if give < qty:
            self.world.add_news(f"Aid note: {target} lacked {res}; transferred {give}/{qty}.")
        if T.gold > 0:
            T.bankrupt = False
            T.zero_gold_streak = 0
        self.world.add_news(
            f"Aid accepted: {target} accepted {bidder}'s bailout (+{gold} gold). "
            f"In return {bidder} receives {give}/{qty} {res}."
        )
        console.rule("[bold blue]IMF ROOM[/] Bailout")
        console.print(f"  {bidder} → {target}: +{gold} gold ; {bidder} takes {give}/{qty} {res}")

    def _apply_population_upkeep(self, c: Country) -> bool:
        pop_m = max(1, int(round(c.population / 1_000_000)))
        need_food = math.ceil(pop_m * POP_CONSUMP_PER_MILLION["food"])
        need_iron = math.ceil(pop_m * POP_CONSUMP_PER_MILLION["iron"])
        need_oil = math.ceil(pop_m * POP_CONSUMP_PER_MILLION["oil"])

        shortages = []
        for res, need in (("food", need_food), ("iron", need_iron), ("oil", need_oil)):
            have = c.stock.get(res, 0)
            take = min(have, need)
            c.stock[res] = have - take
            if take < need:
                shortages.append(res)

        if shortages:
            drop = 0
            if "food" in shortages:
                drop += self.random.randint(6, 12)
            if "iron" in shortages:
                drop += self.random.randint(2, 4)
            if "oil" in shortages:
                drop += self.random.randint(2, 4)
            if drop:
                self._broad_unrest(c, -drop, focus=["workers", "nationalists"], bleed=0.5)
            self.world.add_news(
                f"Upkeep: {c.name} population consumed F:{need_food}/I:{need_iron}/O:{need_oil}. "
                f"Shortages in {', '.join(shortages)} (factions angered)."
            )
        else:
            self.world.add_news(
                f"Upkeep: {c.name} population consumed F:{need_food}/I:{need_iron}/O:{need_oil} (no shortages)."
            )
        return "food" not in shortages

    def _apply_population_growth(self, c: Country, allow_growth: bool) -> None:
        if not allow_growth:
            return
        rate = min(BASE_POP_GROWTH_RATE + c.social_growth_bonus, MAX_POP_GROWTH_RATE)
        if rate <= 0:
            return
        delta = int(c.population * rate)
        if delta <= 0:
            return
        c.population += delta
        self.world.add_news(
            f"Demographics: {c.name} population +{delta} ({rate*100:.2f}%)."
        )

    def _roll_poverty_unrest(self, c: Country) -> None:
        p = 0.0
        if c.gold < 25:
            p = 0.50
        elif c.gold < 50:
            p = 0.35
        if p > 0 and self.random.random() < p:
            drop = self.random.randint(5, 10)
            self._broad_unrest(c, -drop, focus=["workers", "religious"], bleed=0.4)
            self.world.add_news(f"Civil Unrest (poverty): {c.name} factions protested low reserves (gold={c.gold}).")

    async def _resolve_alliances(self, reqs: List[Tuple[str, AllianceOrder]]) -> None:
        for cname, (dec, _, _) in getattr(self, "_last_decisions", {}).items():
            for a in dec.alliance:
                if a.leave:
                    f = self.world.faction_of(cname)
                    if f and cname in f.members:
                        f.members.remove(cname)
                        self.world.add_news(f"Faction: {cname} left [{f.name}].")
                        console.print(f"[yellow]FACTION[/] {cname} left faction [{f.name}].")

        self.world.factions[:] = [f for f in self.world.factions if f.members]

        want: Dict[str, List[AllianceOrder]] = {}
        faction_by_name = {f.name: f for f in self.world.factions}
        for who, (dec, _, _) in getattr(self, "_last_decisions", {}).items():
            for a in dec.alliance:
                if a.leave or not a.target:
                    continue
                want.setdefault(who, []).append(a)

        formed: List[Faction] = []
        mutual_pairs: set[frozenset[str]] = set()

        for a_name, a_list in want.items():
            for prop in a_list:
                b_name = prop.target
                if b_name in faction_by_name:
                    fac = faction_by_name[b_name]
                    sponsor = fac.created_by if fac.created_by in fac.members else (fac.members[0] if fac.members else None)
                    if sponsor:
                        self.world.add_news(
                            f"Parser: {a_name} – alliance target '{b_name}' matched a faction; treating as join request to [{b_name}] via sponsor {sponsor}."
                        )
                        b_name = sponsor
                if b_name in want and any(x.target == a_name for x in want[b_name]):
                    mutual_pairs.add(frozenset({a_name, b_name}))
                if b_name in want and any(x.target == a_name for x in want[b_name]):
                    fa = self.world.faction_of(a_name)
                    fb = self.world.faction_of(b_name)
                    if fa and fb and fa is fb:
                        continue
                    secret = True
                    name = prop.faction_name or f"{a_name}-{b_name} Pact"
                    for x in want[b_name]:
                        if x.target == a_name:
                            name = prop.faction_name or x.faction_name or name
                            secret = prop.secret and x.secret
                            break
                    if fa and not fb:
                        fa.members.append(b_name)
                        fa.secret = fa.secret and secret
                        self.world.add_news(f"Faction formed ({'secret' if fa.secret else 'public'}): [{fa.name}] adds {b_name}.")
                        console.print(f"[green]FACTION UPDATED[/] [{fa.name}] adds {b_name}. Members now: {', '.join(fa.members)}")
                        models = ", ".join(f"{m}({self.agents[m].model})" for m in fa.members if m in self.agents)
                        console.print(f"  Members' models: {models}")
                    elif fb and not fa:
                        fb.members.append(a_name)
                        fb.secret = fb.secret and secret
                        self.world.add_news(f"Faction formed ({'secret' if fb.secret else 'public'}): [{fb.name}] adds {a_name}.")
                        console.print(f"[green]FACTION UPDATED[/] [{fb.name}] adds {a_name}. Members now: {', '.join(fb.members)}")
                        models = ", ".join(f"{m}({self.agents[m].model})" for m in fb.members if m in self.agents)
                        console.print(f"  Members' models: {models}")
                    elif not fa and not fb:
                        newf = Faction(name=name, members=[a_name, b_name], secret=secret, created_by=a_name)
                        if not any(set(f.members) == set(newf.members) for f in self.world.factions):
                            self.world.factions.append(newf)
                            formed.append(newf)
                            console.print(f"[bold green]FACTION FORMED[/] ({'secret' if secret else 'public'}): [{name}] — {a_name}, {b_name}")
                            console.print(f"  Models: {a_name} -> {self.agents[a_name].model}, {b_name} -> {self.agents[b_name].model}")

        for f in formed:
            self.world.add_news(f"Faction formed ({'secret' if f.secret else 'public'}): [{f.name}] — {', '.join(f.members)}.")
            models = ", ".join(f"{m}({self.agents[m].model})" for m in f.members if m in self.agents)
            console.print(f"[green]FACTION LOG[/] [{f.name}] members/models: {models}")

        for a_name, a_list in want.items():
            for prop in a_list:
                if not prop.faction_name:
                    continue
                fa = self.world.faction_of(a_name)
                fb = self.world.faction_of(prop.target)
                if fa or fb:
                    continue
                if prop.faction_name in faction_by_name:
                    continue
                newf = Faction(name=prop.faction_name, members=[a_name], secret=prop.secret, created_by=a_name)
                self.world.factions.append(newf)
                self.world.add_news(f"Faction formed ({'secret' if newf.secret else 'public'}): [{newf.name}] — {a_name}.")
                console.print(f"[bold green]FACTION FORMED[/] ({'secret' if newf.secret else 'public'}): [{newf.name}] — founder {a_name}")
                faction_by_name[newf.name] = newf

        join_requests: List[Tuple[str, str, Faction, str]] = []
        for a_name, a_list in want.items():
            for prop in a_list:
                b_name = prop.target
                if frozenset({a_name, b_name}) in mutual_pairs:
                    continue
                fb = self.world.faction_of(b_name)
                fa = self.world.faction_of(a_name)
                if fb and (not fa or fa is not fb):
                    join_requests.append((a_name, b_name, fb, prop.message or ""))

        for applicant, sponsor, faction, reason in join_requests:
            console.print(f"[cyan]JOIN REQUEST[/] {applicant} applied to join [{faction.name}] (sponsored by {sponsor}). Reason: {reason or '—'}")
            ok = True
            for member in list(faction.members):
                model_to_ask = self.agents[member].model if member in self.agents else "UNKNOWN"
                console.print(f"  Asking voter: {member} (model={model_to_ask}) to vote on admitting {applicant}...")
                vote = await self._ask_alliance_vote(member, applicant, faction, sponsor, reason)
                if vote is None and len(faction.members) == 1 and member == sponsor:
                    vote = AllianceVote(requester=applicant, faction=faction.name, decision="accept", reason="sponsor fallback")

                if vote is None or vote.decision != "accept":
                    why = (vote.reason if (vote and vote.reason) else "no stated reason")
                    self.world.add_news(f"Faction vote: {member} declined {applicant} into [{faction.name}] (reason: {why}).")
                    self.world.add_news(f"Faction vote: [{faction.name}] declined {applicant}'s entry (veto by {member}: {why}).")
                    console.print(f"[red]VOTE[/] {member} vetoed/failed to vote for {applicant} ({why}). Admission denied.")
                    ok = False
                    break
                else:
                    reason_txt = vote.reason or "—"
                    self.world.add_news(f"Faction vote: {member} accepted {applicant} into [{faction.name}] (reason: {reason_txt}).")
                    console.print(f"[green]VOTE[/] {member} accepted {applicant} (reason: {reason_txt}).")

            if ok:
                if applicant not in faction.members:
                    faction.members.append(applicant)
                self.world.add_news(f"Faction vote: [{faction.name}] admitted {applicant} (unanimous).")
                console.print(f"[bold green]ADMITTED[/] {applicant} admitted to [{faction.name}]. Members now: {', '.join(faction.members)}")
                models = ", ".join(f"{m}({self.agents[m].model})" for m in faction.members if m in self.agents)
                console.print(f"  Members' models after admission: {models}")

    def _war_hp(self, army: int) -> int:
        return max(0, int(army * WAR_HP_PER_ARMY))

    def _calc_attack_value(self, c: Country) -> float:
        research_bonus = 2.0 * (c.research.military / 10.0)
        stab_mult = stability_military_multiplier(c.stability)
        return (WAR_BASE_ATTACK + research_bonus) * stab_mult

    def _attack_narrative(self, attacker: str, defender: str) -> str:
        choices = [
            f"{attacker} launched a predawn crossing at a border bridge, catching {defender} off‑guard.",
            f"{attacker} massed armour in the low hills and drove hard at {defender}'s frontier towns.",
            f"{attacker} used diversionary artillery then punched through {defender}'s weakest sector.",
            f"{attacker} executed a river flanking manoeuvre and bypassed {defender}'s forts.",
            f"{attacker} seized key rail junctions before {defender} could mobilise reserves.",
        ]
        return self.random.choice(choices)

    def _apply_damage(self, hp_def: int, dmg: float, defended: bool) -> Tuple[int, int]:
        mult = (1.0 - WAR_DEFEND_BONUS) if defended else 1.0
        dealt = max(1, int(round(dmg * mult)))
        return max(0, hp_def - dealt), dealt

    async def _ask_war_decision(self, actor: str, context: str, allow: List[str]) -> WarDecision:
        agent = self.agents[actor]
        allow_sorted = sorted(set(allow))
        schema = {
            "war_decision": {
                "action": "attack|defend|call_allies|capitulate",
                "reason": "STRING (optional)"
            }
        }
        prompt = (
            "WAR ROOM — Choose one action. Allowed: " + ", ".join(allow_sorted) + "\
\
" +
            context + "\
\
Return ONLY JSON matching: " + json.dumps(schema, ensure_ascii=False)
        )
        messages = [
            {"role":"system","content":"You are a head of state under attack. Reply with ONLY valid JSON under key 'war_decision'."},
            {"role":"user","content": prompt},
        ]
        provider_label = "OpenRouter" if agent.provider == "or" else "Ollama"
        try:
            raw = await self._chat_model(agent, messages, options={"temperature":0.4,"top_p":0.9}, stream=False)
        except httpx.HTTPError as e:
            self.world.add_news(
                f"WAR ROOM: {actor} failed to respond via {provider_label} ({e}). Defaulting to 'defend'."
            )
            return WarDecision(action="defend")

        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            obj = json.loads(m.group(0)) if m else {}
            wd = obj.get("war_decision", obj)
            dec = WarDecision(**{k: wd.get(k) for k in ["action","reason"]})
        except Exception:
            dec = WarDecision(action="defend")

        if dec.action not in allow_sorted:
            dec.action = "defend"
        return dec

    async def _ask_ally_assist(self, ally: str, attacker: str, defender: str) -> bool:
        agent = self.agents[ally]
        context = (f"WAR ROOM — Ally request: {defender} is under attack by {attacker}.\n"
                   f"You may strike {attacker} *once* this turn. Respond true/false in JSON as: {{\"assist\":true|false}}.")
        messages = [
            {"role":"system","content":'You are a cautious ally. Return ONLY JSON like {"assist":true|false}.'},
            {"role":"user","content": context},
        ]
        try:
            raw = await self._chat_model(agent, messages, options={"temperature":0.3,"top_p":0.9}, stream=False)
        except httpx.HTTPError:
            return False
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            obj = json.loads(m.group(0)) if m else {}
            return bool(obj.get("assist", False))
        except Exception:
            return False

    async def _war_room(self, attacker_name: str, defender_name: str, cause: str, goal: Optional[str] = None) -> None:
        a = self.world.country(attacker_name)
        d = self.world.country(defender_name)

        console.rule(f"[bold red]WAR ROOM[/] {attacker_name} attacks {defender_name}")
        self.world.add_news(f"War declared: {attacker_name} → {defender_name}. Cause: {cause}.")
        self._adjust_faction_approvals(
            a,
            {
                "business_elite": -8,
                "workers": -8,
                "greens": -6,
                "religious": -4,
                "nationalists": -2,
                "military": -2,
            },
            default=-5,
        )
        self._adjust_faction_approvals(
            d,
            {
                "business_elite": -10,
                "workers": -10,
                "greens": -8,
                "religious": -6,
                "nationalists": +3,
                "military": +2,
            },
            default=-6,
        )

        hp_a = self._war_hp(a.army)
        hp_d = self._war_hp(d.army)
        if hp_a <= 0:
            self.world.add_news(f"War aborted: {attacker_name} had 0 HP army; attack on {defender_name} cancelled.")
            console.print(f"[red]WAR ABORTED[/] {attacker_name} has 0 HP; no attack possible.")
            self._broad_unrest(a, -5, focus=["military", "nationalists"], bleed=0.4)
            return
        if hp_d <= 0:
            console.print(f"[yellow]OCCUPATION[/] {defender_name} has 0 HP; {attacker_name} takes objectives without battle.")
            winner, loser = attacker_name, defender_name
            self._apply_war_resolution_no_battle(winner, loser, cause)
            return
        def_str = f"HP {hp_d} (army {d.army})"
        atk_str = f"HP {hp_a} (army {a.army})"
        console.print(f"  [magenta]{attacker_name}[/] {atk_str}  vs  [cyan]{defender_name}[/] {def_str}")

        defending_next_a = False
        defending_next_d = False

        narrative = self._attack_narrative(attacker_name, defender_name)
        atk_val = self._calc_attack_value(a)
        jitter = 1.0 + self.random.uniform(-WAR_JITTER, WAR_JITTER)
        dmg = atk_val * 1.0 * jitter
        hp_d, dealt = self._apply_damage(hp_d, dmg, defended=defending_next_d)
        defending_next_d = False
        console.print(f"  {narrative}")
        console.print(f"  → {attacker_name} deals {dealt} dmg (atk≈{atk_val:.1f}). {defender_name} HP now {hp_d}.")
        self.world.add_news(f"War room: {attacker_name} struck first ({dealt} dmg). {defender_name} HP {hp_d}.")

        rounds = 1
        capitulated = None

        while rounds < WAR_MAX_ROUNDS and hp_a > 0 and hp_d > 0 and not capitulated:
            rounds += 1
            context = (f"{attacker_name} attacked: last hit {dealt} damage. Your HP {hp_d}. Opponent HP {hp_a}.\
"
                       f"Actions: ATTACK deals damage now; DEFEND reduces next incoming by {int(WAR_DEFEND_BONUS*100)}%; "
                       f"CALL_ALLIES asks your faction to strike once each; CAPITULATE ends the war.")
            allow = ["attack","defend","call_allies","capitulate"]
            dec_d = await self._ask_war_decision(defender_name, context, allow)
            console.print(f"  [cyan]{defender_name}[/] chose: {dec_d.action}{' — '+dec_d.reason if dec_d.reason else ''}")

            if dec_d.action == "capitulate":
                capitulated = defender_name
                break
            if dec_d.action == "call_allies":
                fac = self.world.faction_of(defender_name)
                if fac:
                    for ally in list(fac.members):
                        if ally in (defender_name, attacker_name):
                            continue
                        if ally not in self.world.countries:
                            continue
                        assist = await self._ask_ally_assist(ally, attacker_name, defender_name)
                        if assist:
                            ca = self.world.country(ally)
                            atk_val2 = self._calc_attack_value(ca)
                            jit2 = 1.0 + self.random.uniform(-WAR_JITTER, WAR_JITTER)
                            hp_a, dealt2 = self._apply_damage(hp_a, atk_val2 * jit2, defended=defending_next_a)
                            defending_next_a = False
                            self.world.add_news(f"Ally strike: {ally} hit {attacker_name} for {dealt2}. {attacker_name} HP {hp_a}.")
                            console.print(f"    Ally {ally} strikes {attacker_name} for {dealt2} (atk≈{atk_val2:.1f}). HP now {hp_a}.")
                if dec_d.reason and "attack" in dec_d.reason.lower():
                    dec_d.action = "attack"
                elif dec_d.reason and "defend" in dec_d.reason.lower():
                    dec_d.action = "defend"
                else:
                    dec_d.action = "defend"

            if dec_d.action == "defend":
                defending_next_d = True
            elif dec_d.action == "attack":
                atk_val_d = self._calc_attack_value(d)
                jitd = 1.0 + self.random.uniform(-WAR_JITTER, WAR_JITTER)
                hp_a, dealtd = self._apply_damage(hp_a, atk_val_d * jitd, defended=defending_next_a)
                defending_next_a = False
                console.print(f"  → {defender_name} counterattacks for {dealtd} (atk≈{atk_val_d:.1f}). {attacker_name} HP now {hp_a}.")
                self.world.add_news(f"War room: {defender_name} countered ({dealtd}). {attacker_name} HP {hp_a}.")

            if hp_a <= 0 or hp_d <= 0:
                break

            context_a = (f"Enemy stance: {dec_d.action}. Your HP {hp_a}. Opponent HP {hp_d}. Choose next move.")
            dec_a = await self._ask_war_decision(attacker_name, context_a, ["attack","defend"])
            console.print(f"  [magenta]{attacker_name}[/] chose: {dec_a.action}{' — '+dec_a.reason if dec_a.reason else ''}")
            if dec_a.action == "defend":
                defending_next_a = True
            else:
                atk_val_a2 = self._calc_attack_value(a)
                jita2 = 1.0 + self.random.uniform(-WAR_JITTER, WAR_JITTER)
                hp_d, dealt_a2 = self._apply_damage(hp_d, atk_val_a2 * jita2, defended=defending_next_d)
                defending_next_d = False
                console.print(f"  → {attacker_name} presses the attack for {dealt_a2} (atk≈{atk_val_a2:.1f}). {defender_name} HP now {hp_d}.")
                self.world.add_news(f"War room: {attacker_name} pressed ({dealt_a2}). {defender_name} HP {hp_d}.")

        if capitulated == defender_name or hp_d <= 0 or (hp_a > hp_d and rounds >= WAR_MAX_ROUNDS):
            winner, loser = attacker_name, defender_name
        elif hp_a <= 0 or (hp_d > hp_a and rounds >= WAR_MAX_ROUNDS):
            winner, loser = defender_name, attacker_name
        else:
            winner, loser = (attacker_name, defender_name) if hp_d <= hp_a else (defender_name, attacker_name)

        wa, wl = self.world.country(winner), self.world.country(loser)

        if winner == attacker_name:
            a.army = max(0, hp_a // WAR_HP_PER_ARMY)
            d.army = max(0, hp_d // WAR_HP_PER_ARMY)
        else:
            a.army = max(0, hp_a // WAR_HP_PER_ARMY)
            d.army = max(0, hp_d // WAR_HP_PER_ARMY)

        gold_take = max(0, int(wl.gold * 0.35))
        wl.gold -= gold_take
        wa.gold += gold_take

        best_res = max(RESOURCES, key=lambda r: wl.stock.get(r, 0))
        steal_amt = min(wl.stock.get(best_res, 0), self.random.randint(25, 100))
        wl.stock[best_res] = max(0, wl.stock.get(best_res, 0) - steal_amt)
        wa.stock[best_res] = wa.stock.get(best_res, 0) + steal_amt

        win_bonus = 10 if capitulated else 7
        lose_penalty = 16 if capitulated else 12
        self._adjust_faction_approvals(
            wa,
            {
                "military": win_bonus,
                "nationalists": win_bonus,
                "business_elite": 4,
                "workers": 3,
                "greens": -2,
            },
            default=win_bonus * 0.3,
        )
        self._adjust_faction_approvals(
            wl,
            {
                "military": -lose_penalty,
                "nationalists": -lose_penalty,
                "workers": -lose_penalty * 0.6,
                "business_elite": -lose_penalty * 0.6,
                "greens": -lose_penalty * 0.4,
            },
            default=-lose_penalty * 0.4,
        )
        for k in RESOURCES:
            wa.production[k] = int(max(0, round(wa.production.get(k, 0) * 1.10)))
        self.world.add_news(f"Expansion incentive: {winner} production increased by +10% after victory.")

        headline = (f"War concluded: {attacker_name} vs {defender_name} → Winner: {winner} "
                     f"(+{gold_take} gold, took {steal_amt} {best_res}; rounds={rounds}{', capitulation' if capitulated else ''}).")
        console.print(f"[bold green]{headline}[/]")
        self.world.add_news(headline)
        self.world.wars.append((winner, loser))

        self._finalise_war_result(
            attacker_name=attacker_name,
            defender_name=defender_name,
            winner=winner,
            steal_amt=steal_amt,
            best_res=best_res,
            gold_take=gold_take,
            rounds=rounds,
            capitulated=bool(capitulated),
            cause=cause,
        )

        self._apply_war_goal(winner, loser, goal)

    def _apply_war_goal(self, winner: str, loser: str, goal: Optional[str]) -> None:
        if not goal:
            return
        goal_l = str(goal).lower()
        if goal_l == "puppet":
            wl = self.world.country(loser)
            wl.overlord = winner
            wl.puppet_since_turn = self.world.turn
            f = self.world.faction_of(loser)
            if f and loser in f.members:
                f.members.remove(loser)
            self.world.add_news(f"Puppet: {winner} installed a puppet regime in {loser} (Turn {self.world.turn}). Tribute: 50% of production flows to {winner} each turn.")
            console.print(f"[bold magenta]PUPPET[/] {winner} installed a puppet regime in {loser} (Turn {self.world.turn}).")
        elif goal_l == "annex":
            self._annex_country(winner, loser)

    def _apply_puppet_control(self, overlord_name: str, pc: PuppetControl) -> None:
        act = str(pc.action).lower()
        pup = str(pc.puppet)
        if pup not in self.world.countries:
            self.world.add_news(f"Parser: {overlord_name} – puppet_control refers to unknown country '{pup}'.")
            return
        c = self.world.country(pup)
        if c.overlord != overlord_name:
            self.world.add_news(f"Parser: {overlord_name} – '{pup}' is not your puppet; ignoring '{act}'.")
            return
        if act == "release":
            c.overlord = None
            was = c.puppet_since_turn
            c.puppet_since_turn = None
            self.world.add_news(f"Puppet released: {overlord_name} released {pup} (puppet since Turn {was}).")
            console.print(f"[bold yellow]PUPPET RELEASE[/] {overlord_name} released {pup} (was puppet since Turn {was}).")
        elif act == "annex":
            self._annex_country(overlord_name, pup)
        else:
            self.world.add_news(f"Parser: {overlord_name} – unknown puppet_control action '{act}'.")

    def _annex_country(self, winner: str, loser: str) -> None:
        if loser not in self.world.countries:
            return
        wa = self.world.country(winner)
        wl = self.world.country(loser)
        wa.surface_km2 += wl.surface_km2
        wa.population += wl.population
        for r in RESOURCES:
            wa.production[r] = int(wa.production.get(r, 0)) + int(wl.production.get(r, 0))
            wa.stock[r] = int(wa.stock.get(r, 0)) + int(wl.stock.get(r, 0))
        wa.gold += wl.gold
        for c in self.world.countries.values():
            c.loans_in.pop(loser, None)
            c.loans_out.pop(loser, None)
        f = self.world.faction_of(loser)
        if f and loser in f.members:
            f.members.remove(loser)
        for name, ctry in self.world.countries.items():
            if ctry.overlord == loser:
                ctry.overlord = None
                ctry.puppet_since_turn = None
        self.world.add_news(f"Annexation: {winner} annexed {loser}. Territory, population and industry absorbed. {loser} ceases to exist.")
        console.print(f"[bold red]ANNEXATION[/] {winner} annexed {loser} — territory, population and industry absorbed.")
        self.agents.pop(loser, None)
        self.world.countries.pop(loser, None)

    async def _ask_alliance_vote(self, voter: str, applicant: str, faction: Faction, sponsor: str, reason: str) -> Optional[AllianceVote]:
        voter_agent = self.agents[voter]
        mem = self.agents.get(applicant).memory_private if self.agents.get(applicant) else []
        app_hist = "\
".join(mem[-12:]) if mem else "No prior events."
        mem2 = self.agents.get(voter).memory_private if self.agents.get(voter) else []
        voter_hist = "\
".join(mem2[-12:]) if mem2 else "No prior events."
        c_app = self.world.country(applicant)
        app_summary = {
            "name": c_app.name,
            "production": c_app.production,
            "population": c_app.population,
            "tax_rate": round(c_app.tax_rate,3),
            "research": c_app.research.as_dict(),
        }
        user = (
            f"VOTE REQUEST — Faction: [{faction.name}]\
"
            f"Sponsor: {sponsor}\
"
            f"Applicant: {applicant}\
"
            f"Applicant summary: {json.dumps(app_summary, ensure_ascii=False)}\
"
            f"Applicant recent history:\
{app_hist}\
\
"
            f"Your recent history:\
{voter_hist}\
\
"
            f"Sponsor’s stated reason: {reason or '—'}\
\
"
            "Return ONLY JSON:\
"
            "{\
"
            '  "alliance_vote": [{"requester":"'+applicant+'","faction":"'+faction.name+'","decision":"accept|decline","reason":STRING}]\
'
            "}\
"
        )
        messages = [
            {"role":"system","content":"You are a cautious head of state. Return only valid JSON with 'alliance_vote' as specified."},
            {"role":"user","content": user},
        ]

        provider_label = "OpenRouter" if voter_agent.provider == "or" else "Ollama"
        console.print(
            f"[magenta]VOTE CALL[/] contacting {voter} using model: {voter_agent.model} ({provider_label}) ..."
        )
        try:
            raw = await self._chat_model(voter_agent, messages, options={"temperature":0.4,"top_p":0.9}, stream=False)
        except httpx.HTTPError as e:
            console.print(f"[red]ERROR[/] during alliance vote call for {voter}: {e}")
            return None

        try:
            await self._unload_model(voter_agent.provider, voter_agent.model)
            console.print(f"[dim]UNLOAD[/] requested for model {voter_agent.model} (post-vote).")
        except Exception:
            console.print(f"[yellow]WARN[/] failed to unload model {voter_agent.model} after vote (best-effort).")

        _, parsed, _, _ = extract_and_coerce_decision(raw)
        if not parsed:
            console.print(f"[yellow]WARN[/] {voter} returned no/parsing-failed JSON for alliance vote.")
            return None
        try:
            dec = ModelDecision.model_validate(parsed)
        except ValidationError:
            console.print(f"[yellow]WARN[/] {voter} returned invalid vote JSON.")
            return None
        return dec.alliance_vote[0] if dec.alliance_vote else None

    def _resolve_loans(self, loans: List[LoanOrder]) -> None:
        banner_shown = False
        for lo in loans:
            if lo.action == "offer":
                lender = None
                for cname, (dec, _, _) in getattr(self, "_last_decisions", {}).items():
                    if lo in dec.loans:
                        lender = cname
                        break
                if lender is None:
                    continue
                if lo.counterparty not in self.world.countries:
                    continue
                L = self.world.country(lender)
                B = self.world.country(lo.counterparty)
                if L.gold >= lo.gold and lo.gold > 0 and 0.0 <= lo.interest_rate <= 1.0:
                    L.gold -= lo.gold
                    B.gold += lo.gold
                    L.loans_out[B.name] = (L.loans_out.get(B.name, (0, lo.interest_rate))[0] + lo.gold, lo.interest_rate)
                    B.loans_in[L.name] = (B.loans_in.get(L.name, (0, lo.interest_rate))[0] + lo.gold, lo.interest_rate)
                    self.world.add_news(f"Loan: {lender} → {B.name} {lo.gold} gold @ {int(lo.interest_rate*100)}%/turn.")
                    if not banner_shown:
                        console.rule("[bold blue]IMF ROOM[/] Loan Desk")
                        banner_shown = True
                    console.print(f"  {lender} → {B.name}: +{lo.gold} gold @ {int(lo.interest_rate*100)}%/turn")
            elif lo.action == "request":
                pass

    def _accrue_interest(self) -> None:
        for c in self.world.countries.values():
            for lender, (amt, rate) in list(c.loans_in.items()):
                due = int(amt * rate)
                if c.gold >= due:
                    c.gold -= due
                    self.world.country(lender).gold += due
                else:
                    self._adjust_faction_approvals(c, {"business_elite": -3}, default=-1)

    def _world_snapshot(self) -> Dict[str, Any]:
        return {
            "turn": self.world.turn,
            "countries": {
                k: {
                    **dataclasses.asdict(v),
                    "model": self.agents[k].model,
                    "model_provider": self.agents[k].provider,
                    "model_alias": self.agents[k].alias or friendly_alias(self.agents[k].model),
                    "flag": (Path(self.agents[k].flag).name if self.agents[k].flag else None),
                    "flag_url": (f"flags/{Path(self.agents[k].flag).name}" if self.agents[k].flag else None),
                }
                for k, v in self.world.countries.items()
            },
            "alliances": [dataclasses.asdict(a) for a in self.world.alliances],
            "factions": [
                {"name": f.name, "members": list(f.members), "secret": f.secret, "created_by": f.created_by}
                for f in self.world.factions
            ],
            "wars": list(self.world.wars),
            "news": list(self.world.news),
            "war_log": list(self.world.war_log),
        }

    def _print_table(self) -> None:
        tab = Table(title=f"Turn {self.world.turn} — World Overview", box=box.SIMPLE_HEAVY)
        tab.add_column("Country")
        tab.add_column("Gold", justify="right")
        tab.add_column("Army", justify="right")
        tab.add_column("Stability", justify="right")
        tab.add_column("Tension", justify="right")
        tab.add_column("Faction", justify="left")
        tab.add_column("Tax%", justify="right")
        tab.add_column("Pop(M)", justify="right")
        tab.add_column("Research", justify="left")
        tab.add_column("Key stocks")
        for c in self.world.countries.values():
            stocks = ", ".join(f"{k}:{v}" for k, v in c.stock.items() if v > 0)[:50]
            d_stab = c.stability - getattr(c, "stability_prev", c.stability)
            stab_cell = f"{c.stability} ({'+' if d_stab>=0 else ''}{d_stab})" if d_stab != 0 else f"{c.stability}"
            fobj = self.world.faction_of(c.name)
            fac_cell = f"{fobj.name} ({'secret' if fobj and fobj.secret else 'public'})" if fobj else "—"
            tab.add_row(
                c.name,
                str(c.gold),
                str(c.army),
                stab_cell,
                f"{c.faction_tension:.1f}",
                fac_cell,
                str(int(c.tax_rate*100)),
                str(max(1, c.population // 1_000_000)),
                f"{c.research.economic}/{c.research.industrial}/{c.research.military}",
                stocks
            )
        console.print(tab)


def seed_world(names: List[str]) -> World:
    rnd = random.Random(1234)
    countries: Dict[str, Country] = {}

    def sample_tax_rate() -> float:
        band = rnd.choices(
            population=[0,1,2,3,4],
            weights=[2, 6, 70, 18, 4],
            k=1
        )[0]
        if band == 0: lo, hi = 0, 5
        elif band == 1: lo, hi = 5, 15
        elif band == 2: lo, hi = 15, 30
        elif band == 3: lo, hi = 30, 40
        else: lo, hi = 40, 50
        return rnd.randint(lo, hi) / 100.0
    for i, n in enumerate(names):
        base_gold = rnd.randint(700, 3500)
        base_area = rnd.randint(40_000, 2_000_000)
        prod = {
            "food": rnd.randint(3, 20),
            "iron": rnd.randint(0, 15),
            "oil": rnd.randint(0, 10),
            "timber": rnd.randint(0, 20),
            "rare_earths": rnd.randint(0, 6),
        }
        if sum(1 for v in prod.values() if v == 0) < 1:
            zero_key = rnd.choice(list(prod.keys()))
            prod[zero_key] = 0
        keys = list(prod.keys())
        z1, z2 = rnd.sample(keys, 2)
        prod[z1] = 0
        prod[z2] = 0
        stock = {k: rnd.randint(20, 120) for k in RESOURCES}
        pop = rnd.randint(2_000_000, 90_000_000)
        tr = sample_tax_rate()
        countries[n] = Country(
            name=n,
            surface_km2=base_area,
            production=prod,
            stock=stock,
            gold=base_gold,
            army=rnd.randint(0, 40),
            stability=rnd.randint(50, 80),
            population=pop,
            tax_rate=tr,
            tax_rate_prev=tr,
            domestic_factions=build_domestic_factions(rnd, anchor=rnd.randint(50, 75)),
        )
        countries[n].recompute_stability()
    return World(countries=countries)


def attach_agents(world: World, models: List[str]) -> List[Agent]:
    pairs = list(zip(list(world.countries.keys()), models))
    agents: List[Agent] = []
    for cname, model in pairs:
        provider, resolved_model = resolve_model_and_provider(model)
        p = preset_for_model(resolved_model)
        flag = p.get("flag") if p else None
        agents.append(
            Agent(
                country=cname,
                model=resolved_model,
                alias=friendly_alias(resolved_model),
                flag=flag,
                provider=provider,
            )
        )
    return agents
