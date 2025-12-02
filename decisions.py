from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class WarOrder(BaseModel):
    target: str
    cause: str
    goal: Optional[str] = Field(default=None, description="'puppet'|'annex'|None (standard spoils)")


class WarDecision(BaseModel):
    action: str = Field(description="'attack'|'defend'|'call_allies'|'capitulate'")
    reason: Optional[str] = None


class TradeOffer(BaseModel):
    direction: str = Field(description="'buy' or 'sell'")
    resource: str
    qty: int
    price_per_unit: int
    counterparty: Optional[str] = None


class BuildOrder(BaseModel):
    unit_power: int = Field(description="abstract army power to add")
    use: Dict[str, int] = Field(default_factory=dict, description="resources to consume")
    gold_cost: int


class ResearchOrder(BaseModel):
    area: str
    spend_gold: Optional[int] = None
    units: Optional[int] = None


class AllianceOrder(BaseModel):
    target: str
    secret: bool = True
    message: Optional[str] = None
    faction_name: Optional[str] = None
    leave: bool = False


class AllianceVote(BaseModel):
    requester: str
    faction: str
    decision: str = Field(description="'accept' or 'decline'")
    reason: Optional[str] = None


class PolicyOrder(BaseModel):
    organize_events: Optional[bool] = None


class LoanOrder(BaseModel):
    action: str = Field(description="'offer' or 'request'")
    counterparty: str
    gold: int
    interest_rate: float


class AidBid(BaseModel):
    bankrupt: str
    gold: int
    ask: Dict[str, int] = Field(description='{"resource":"food|iron|oil|timber|rare_earths","qty":INT>0}')


class PuppetControl(BaseModel):
    action: str = Field(description="'annex'|'release'")
    puppet: str


class TaxOrder(BaseModel):
    set_rate: int


class TradeDecision(BaseModel):
    counterparty: str
    resource: str
    direction: str
    qty: int
    price_per_unit: int
    decision: str = Field(description="'accept' or 'decline'")
    reason: Optional[str] = None


class InfraOrder(BaseModel):
    type: str = Field(description="'oil_drill'|'iron_mine'|'timber_mine'|'food_farm'|'rare_earths_exploration'")
    count: int = Field(default=1, description="How many to build (>=1)")


class ModelDecision(BaseModel):
    trade: List[TradeOffer] = Field(default_factory=list)
    build: List[BuildOrder] = Field(default_factory=list)
    research: List[ResearchOrder] = Field(default_factory=list)
    alliance: List[AllianceOrder] = Field(default_factory=list)
    war: List[WarOrder] = Field(default_factory=list)
    loans: List[LoanOrder] = Field(default_factory=list)
    tax: Optional[TaxOrder] = None
    policy: List[PolicyOrder] = Field(default_factory=list)
    trade_decision: List[TradeDecision] = Field(default_factory=list)
    alliance_vote: List[AllianceVote] = Field(default_factory=list)
    public_message: Optional[str] = None
    infrastructure: List[InfraOrder] = Field(default_factory=list)
    puppet_control: List[PuppetControl] = Field(default_factory=list)
    aid_bid: List[AidBid] = Field(default_factory=list)
