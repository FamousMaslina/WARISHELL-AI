from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import ValidationError

from config import RESEARCH_AREAS, RESEARCH_UNIT_COST, RESOURCES
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
)

# --------- Parser helpers ----------------------------------------------------
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
FACTION_FORM_RE = re.compile(r"Faction formed \((secret|public)\): \[(.+?)\]")

def _is_response_truncated(raw: str) -> bool:
    """
    Check if the raw response appears to be truncated/cut off.
    Returns True if the response looks incomplete.
    """
    if not raw:
        return True
    
    # Check for common truncation patterns at the end
    stripped = raw.strip()
    
    # If it ends with incomplete JSON patterns, it's likely truncated
    if stripped.endswith('{') or stripped.endswith('['):
        return True
    if stripped.endswith(':'):
        return True
    if stripped.endswith(','):
        return True
    if stripped.endswith('"'):
        # Could be valid, but check if it's an unclosed string
        # Count quotes - odd number means unclosed string
        if stripped.count('"') % 2 == 1:
            return True
    
    # Check for unclosed braces/brackets
    brace_count = 0
    bracket_count = 0
    in_string = False
    escape_next = False
    
    for char in stripped:
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        elif char == '[':
            bracket_count += 1
        elif char == ']':
            bracket_count -= 1
    
    # If braces or brackets are unbalanced, response is truncated
    if brace_count != 0 or bracket_count != 0:
        return True
    
    return False
FACTION_LEFT_RE = re.compile(r"Faction:\s+([\w\-]+)\s+left\s+\[(.+?)\]\.")
VOTE_MEMBER_RE = re.compile(
    r"Faction vote:\s+([\w\-]+)\s+(accepted|declined)\s+([\w\-]+)\s+(?:to|into)\s+\[([^\]]+)\]\s+\(reason:\s*(.*?)\)\.",
    re.IGNORECASE,
)
VOTE_OUTCOME_DECLINE_RE = re.compile(
    r"Faction vote:\s+\[([^\]]+)\]\s+declined\s+([\w\-]+)'s entry\s+\(veto by\s+([\w\-]+):\s*(.*?)\)\.",
    re.IGNORECASE,
)
VOTE_OUTCOME_ADMIT_RE = re.compile(
    r"Faction vote:\s+\[([^\]]+)\]\s+admitted\s+([\w\-]+)\s+\(unanimous\)\.",
    re.IGNORECASE,
)

WAR_NEWS_RE = re.compile(
    r"War:\s+([\w\-]+)\s+vs\s+([\w\-]+)\s+→\s+Winner:\s+([\w\-]+)\s+\(\+(\d+)\s+gold,\s+stole\s+(\d+)\s+(\w+)\)\.(?:\s+Cause cited:\s*(.*))?"
)
WAR_CONCLUDED_RE = re.compile(
    r"War concluded:\s+([\w\-]+)\s+vs\s+([\w\-]+)\s+→\s+Winner:\s+([\w\-]+)\s+\(\+(\d+)\s+gold,\s+(?:took|stole)\s+(\d+)\s+(\w+);.*?\)\.",
    re.IGNORECASE,
)
WAR_DECLARED_RE = re.compile(
    r"War declared:\s+([\w\-]+)\s+→\s+([\w\-]+)\.\s+Cause:\s*(.*?)\.",
    re.IGNORECASE,
)
TRADE_NEWS_RE = re.compile(r"Trade:\s+([\w\-]+)→([\w\-]+)\s+(\d+)\s+(\w+)\s+@\s+(\d+)\s+\(total\s+(\d+)\)\.")
LOAN_NEWS_RE = re.compile(r"Loan:\s+([\w\-]+)\s+→\s+([\w\-]+)\s+(\d+)\s+gold\s+@\s+(\d+)%/turn\.")
BUILD_NEWS_RE = re.compile(r"([\w\-]+)\s+expanded armed forces by \+(\d+)\.")
RESEARCH_NEWS_RE = re.compile(r"([\w\-]+)\s+advanced\s+(economic|industrial|military)\s+research\s+\(\+(\d+),\s+spent\s+(\d+)\)\.")
ALLIANCE_NEWS_RE = re.compile(r"Alliance formed \((secret|public)\):\s+(.+?)\.")


def extract_and_coerce_decision(raw: str) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str], List[str]]:
    """Return (think, decision_dict_in_ModelDecision_shape, json_text, parser_notes)."""
    think = None
    m = THINK_RE.search(raw)
    if m:
        think = m.group(1).strip()

    text = THINK_RE.sub("", raw)
    decoder = json.JSONDecoder()
    i = 0
    objs: List[Any] = []
    while i < len(text):
        nxt = re.search(r"[\{\[]", text[i:])
        if not nxt:
            break
        start = i + nxt.start()
        try:
            val, end = decoder.raw_decode(text, idx=start)
            objs.append(val)
            i = end
        except json.JSONDecodeError:
            i = start + 1

    notes: List[str] = []
    
    # Check if response appears to be truncated
    if _is_response_truncated(raw):
        notes.append("WARNING: Response appears to be truncated/cut off at the end")
    
    for candidate in reversed(objs):
        try:
            dec = ModelDecision.model_validate(candidate)
            return think, dec.model_dump(), json.dumps(dec.model_dump(), ensure_ascii=False), notes
        except ValidationError:
            pass

    dec, coercion_notes = _coerce_to_model_decision(objs)
    notes.extend(coercion_notes)
    if dec:
        d = dec.model_dump()
        return think, d, json.dumps(d, ensure_ascii=False), notes

    return think, None, None, notes


def _coerce_to_model_decision(objs: List[Any]) -> Tuple[Optional[ModelDecision], List[str]]:
    notes: List[str] = []
    out = ModelDecision()

    def set_tax(rate_like: Any, src: str):
        try:
            if isinstance(rate_like, float) and rate_like <= 1.0:
                pct = int(round(rate_like * 100))
            else:
                pct = int(rate_like)
            pct = max(0, min(50, pct))
            out.tax = TaxOrder(set_rate=pct)
        except Exception:
            notes.append(f"Skipped tax (bad rate) ({src}).")

    def add_research(area: Optional[str], spend: Optional[int], units: Optional[int], src: str):
        if not area:
            notes.append(f"Skipped research with no area ({src}).")
            return
        area = str(area).lower()
        if area in ("econ", "economy"):
            area = "economic"
        if area in ("ind", "industry"):
            area = "industrial"
        if area not in RESEARCH_AREAS:
            notes.append(f"Skipped research unknown area '{area}' ({src}).")
            return
        ro = ResearchOrder(area=area, spend_gold=spend, units=units)
        out.research.append(ro)

    def add_trade(direction: str, resource: str, qty: Optional[int], price: Optional[int], counterparty: Optional[str], src: str):
        direction = (direction or "").lower()
        resource = (resource or "").lower()
        if direction not in ("buy", "sell") or resource not in RESOURCES:
            notes.append(f"Skipped trade (bad dir/resource) ({src}).")
            return
        try:
            qty = int(qty)
        except Exception:
            qty = None
        try:
            ppu = int(price) if price is not None else None
        except Exception:
            ppu = None
        if not qty or qty <= 0 or not ppu or ppu <= 0:
            notes.append(f"Skipped trade missing qty/price ({src}).")
            return
        out.trade.append(TradeOffer(direction=direction, resource=resource, qty=qty, price_per_unit=ppu, counterparty=counterparty))

    def add_build(unit_power: Optional[int], use: Optional[Dict[str, int]], gold_cost: Optional[int], src: str):
        try:
            power = int(unit_power) if unit_power is not None else None
            cost = int(gold_cost) if gold_cost is not None else None
        except Exception:
            power, cost = None, None
        if not power or power <= 0 or not cost or cost <= 0:
            notes.append(f"Skipped build (bad power/cost) ({src}).")
            return
        use = {k: int(v) for k, v in (use or {}).items() if k in RESOURCES and isinstance(v, (int, float)) and v > 0}
        out.build.append(BuildOrder(unit_power=power, use=use, gold_cost=cost))

    flat: List[Any] = []
    for obj in objs:
        flat.extend(obj if isinstance(obj, list) else [obj])

    for obj in flat:
        if not isinstance(obj, dict):
            continue

        if isinstance(obj.get("research"), dict):
            for k, v in obj["research"].items():
                try:
                    units = int(v)
                except Exception:
                    units = None
                add_research(k, None, units, "research-dict")

        if isinstance(obj.get("research"), list):
            for it in obj["research"]:
                if not isinstance(it, dict):
                    continue
                area = it.get("area") or it.get("field") or it.get("priority")
                spend = it.get("spend_gold") or it.get("gold") or it.get("investment") or it.get("budget") or it.get("spend")
                units = it.get("units") or it.get("points")
                try:
                    spend = int(spend) if spend is not None else None
                except Exception:
                    spend = None
                try:
                    units = int(units) if units is not None else None
                except Exception:
                    units = None
                add_research(area, spend, units, "research-list")

        if isinstance(obj.get("build"), list):
            for it in obj["build"]:
                if not isinstance(it, dict):
                    continue
                unit_power = it.get("unit_power") or it.get("power") or it.get("army") or it.get("army_power")
                gold_cost = it.get("gold_cost") or it.get("cost") or it.get("price") or it.get("budget")
                use = it.get("use") or it.get("resources") or {}
                add_build(unit_power, use, gold_cost, "build-list")

        if isinstance(obj.get("trade"), list):
            for it in obj["trade"]:
                if not isinstance(it, dict):
                    continue
                add_trade(
                    it.get("direction"),
                    it.get("resource"),
                    it.get("qty") or it.get("quantity") or it.get("amount"),
                    it.get("price_per_unit") or it.get("ppu") or it.get("price"),
                    it.get("counterparty"),
                    "trade-list",
                )

        if isinstance(obj.get("trade_decision"), list) or isinstance(obj.get("trade_response"), list):
            key = "trade_decision" if isinstance(obj.get("trade_decision"), list) else "trade_response"
            for it in obj.get(key, []):
                if not isinstance(it, dict):
                    continue
                try:
                    td = TradeDecision(
                        counterparty=str(it.get("counterparty")),
                        resource=str(it.get("resource")).lower(),
                        direction=str(it.get("direction")).lower(),
                        qty=int(it.get("qty")),
                        price_per_unit=int(it.get("price_per_unit") or it.get("ppu") or it.get("price")),
                        decision=str(it.get("decision")).lower(),
                        reason=(it.get("reason") if isinstance(it.get("reason"), str) else None),
                    )
                    out.trade_decision.append(td)
                except Exception:
                    notes.append("Skipped malformed trade_decision.")

        if isinstance(obj.get("alliance_vote"), list):
            for it in obj["alliance_vote"]:
                if not isinstance(it, dict):
                    continue
                dec = str(it.get("decision", "")).lower()
                if dec not in ("accept", "decline"):
                    notes.append("Skipped alliance_vote with bad decision.")
                    continue
                try:
                    av = AllianceVote(
                        requester=str(it.get("requester")),
                        faction=str(it.get("faction")),
                        decision=dec,
                        reason=(it.get("reason") if isinstance(it.get("reason"), str) else None),
                    )
                    out.alliance_vote.append(av)
                except Exception:
                    notes.append("Skipped malformed alliance_vote.")

        if isinstance(obj.get("public_message"), str):
            out.public_message = obj["public_message"]

        if "tax" in obj and out.tax is None:
            tv = obj["tax"]
            if isinstance(tv, dict):
                set_tax(tv.get("set_rate") or tv.get("rate") or tv.get("tax_rate"), "tax-dict")
            else:
                set_tax(tv, "tax-scalar")
        if "tax_rate" in obj and out.tax is None:
            set_tax(obj.get("tax_rate"), "tax_rate-top")

        action = str(obj.get("action", "")).lower()
        if action:
            if action in ("research", "r&d"):
                area = obj.get("area") or obj.get("field") or obj.get("priority")
                spend = obj.get("spend_gold") or obj.get("gold") or obj.get("investment") or obj.get("budget") or obj.get("spend")
                units = obj.get("units") or obj.get("points")
                try:
                    spend = int(spend) if spend is not None else None
                except Exception:
                    spend = None
                try:
                    units = int(units) if units is not None else None
                except Exception:
                    units = None
                add_research(area, spend, units, "action-research")

            elif action in ("build", "build_army", "military"):
                unit_power = obj.get("unit_power") or obj.get("power") or obj.get("army") or obj.get("army_power")
                gold_cost = obj.get("gold_cost") or obj.get("cost") or obj.get("price") or obj.get("budget")
                use = obj.get("use") or obj.get("resources") or {k: obj[k] for k in RESOURCES if isinstance(obj.get(k), (int, float))}
                add_build(unit_power, use, gold_cost, "action-build")

            elif action == "trade":
                add_trade(
                    obj.get("direction"),
                    obj.get("resource"),
                    obj.get("qty") or obj.get("quantity") or obj.get("amount"),
                    obj.get("price_per_unit") or obj.get("ppu") or obj.get("price"),
                    obj.get("counterparty"),
                    "action-trade",
                )
            elif action in ("tax", "set_tax", "taxation"):
                set_tax(obj.get("set_rate") or obj.get("rate") or obj.get("tax_rate"), "action-tax")
            elif action in ("festival", "stability", "organize_events", "rally"):
                out.policy.append(PolicyOrder(organize_events=True))
        if obj.get("organize_events") is True:
            out.policy.append(PolicyOrder(organize_events=True))
        if obj.get("leave_alliance") or obj.get("leave_faction"):
            out.alliance.append(AllianceOrder(leave=True))
        if "faction_name" in obj and isinstance(obj.get("target"), str):
            out.alliance.append(AllianceOrder(
                target=obj["target"],
                faction_name=str(obj["faction_name"]),
                secret=bool(obj.get("secret", True))
            ))
    if not any([out.trade, out.build, out.research, out.alliance, out.war, out.loans, out.public_message]):
        return None, notes
    return out, notes
