from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import RESOURCES
from console_utils import console
from models import World


@dataclass
class MarketOrder:
    country: str
    direction: str
    resource: str
    qty: int
    ppu: int
    counterparty: Optional[str] = None


class Market:
    def __init__(self) -> None:
        self.public_deals: List[Dict[str, Any]] = []

    def clear(self, world: World, orders: List[MarketOrder]) -> None:
        targeted = [o for o in orders if o.counterparty]
        used: set[int] = set()
        for i, o in enumerate(targeted):
            if i in used:
                continue
            for j, p in enumerate(targeted):
                if j in used or j == i:
                    continue
                if (o.country == p.counterparty and o.counterparty == p.country
                    and o.resource == p.resource and o.direction != p.direction):
                    if o.direction == "sell":
                        seller, buyer, ask = o.country, o.counterparty, o.ppu
                        seller_order, buyer_order = o, p
                    else:
                        seller, buyer, ask = p.country, p.counterparty, p.ppu
                        seller_order, buyer_order = p, o
                    cs = world.country(seller)
                    cb = world.country(buyer)
                    max_afford = cb.gold // max(1, ask)
                    exec_qty = min(seller_order.qty, buyer_order.qty,
                                   cs.stock.get(o.resource, 0), max_afford)
                    if exec_qty <= 0:
                        continue
                    if self._execute(world, seller, buyer, o.resource, exec_qty, ask):
                        seller_order.qty -= exec_qty
                        buyer_order.qty -= exec_qty
                        if seller_order.qty <= 0:
                            used.add(i if seller_order is o else j)
                        if buyer_order.qty <= 0:
                            used.add(i if buyer_order is o else j)

        from collections import defaultdict
        buys_by_res = defaultdict(list)
        sells_by_res = defaultdict(list)
        for k, o in enumerate(orders):
            if o.counterparty or k in used:
                continue
            (buys_by_res if o.direction == "buy" else sells_by_res)[o.resource].append(o)

        for res in RESOURCES:
            buys = sorted(buys_by_res.get(res, []), key=lambda x: (-x.ppu))
            sells = sorted(sells_by_res.get(res, []), key=lambda x: (x.ppu))
            i = j = 0
            while i < len(buys) and j < len(sells):
                b, s = buys[i], sells[j]
                if b.ppu < s.ppu:
                    break
                seller, buyer, ask = s.country, b.country, s.ppu
                cs, cb = world.country(seller), world.country(buyer)
                if cs.stock.get(res, 0) <= 0:
                    j += 1
                    continue
                if cb.gold < ask:
                    i += 1
                    continue
                exec_qty = min(b.qty, s.qty, cs.stock.get(res, 0), cb.gold // ask)
                if exec_qty <= 0:
                    i += 1
                    continue
                if self._execute(world, seller, buyer, res, exec_qty, ask):
                    b.qty -= exec_qty
                    s.qty -= exec_qty
                    if b.qty <= 0:
                        i += 1
                    if s.qty <= 0:
                        j += 1

    def _execute(self, world: World, seller: str, buyer: str, res: str, qty: int, ppu: int) -> bool:
        if qty <= 0 or ppu <= 0:
            return False
        cs, cb = world.country(seller), world.country(buyer)
        if cs.stock.get(res, 0) < qty:
            return False
        total = qty * ppu
        if cb.gold < total:
            return False
        cs.stock[res] -= qty
        cb.stock[res] = cb.stock.get(res, 0) + qty
        cs.gold += total
        cb.gold -= total
        self.public_deals.append({
            "seller": seller, "buyer": buyer, "resource": res,
            "qty": qty, "ppu": ppu, "total": total,
        })
        console.print(f"[yellow]WTO[/]: {buyer} buys {qty} {res} from {seller} @ {ppu} (total {total}).")
        return True
