"""策略档案：冠军 / 挑战者 / 退役，以及"冻结时点"这条纪律。

这是整个自训练闭环里最重要的一块，也是它和"跑个回测挑个最好看的"之间的分界线：

  1. 进化出来的最好参数只是**挑战者**，不会自动上岗（GA 设计文档 §5.5 原意）；
  2. 人工晋升的那一刻记为 `frozen_at` —— **只有这之后的净值数据才算样本外**；
  3. 挑战者入库前必须过三重闸门：参数平原、Deflated Sharpe、walk-forward；
     任何一关不过就记录原因并拒绝入库，而不是"分数高就先试试"。

为什么这么较真：模拟盘本身不产生证据。你在同一段历史上进化一百次，
第一百次的成绩依然全是样本内 —— 只是换了个更贵的过拟合方式。
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import date as _date
from pathlib import Path
from typing import Dict, List, Optional

from config import DATA_DIR
from paper.store import JsonStore

CHALLENGER = "challenger"
CHAMPION = "champion"
RETIRED = "retired"
REJECTED = "rejected"


@dataclass
class Gates:
    """三重闸门的检查结果。缺数据一律按"未通过"处理，不做乐观假设。"""
    plateau_ok: bool = False
    plateau_detail: Dict[str, object] = field(default_factory=dict)
    dsr_ok: bool = False
    dsr_detail: Dict[str, object] = field(default_factory=dict)
    walk_forward_ok: bool = False
    walk_forward_detail: Dict[str, object] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return bool(self.plateau_ok and self.dsr_ok and self.walk_forward_ok)

    def failures(self) -> List[str]:
        out = []
        if not self.plateau_ok:
            out.append("参数平原未通过：邻域扰动后分数明显塌陷，解可能落在针尖上")
        if not self.dsr_ok:
            out.append("Deflated Sharpe 未达 95%：按尝试次数校正后不能拒绝真实期望为 0")
        if not self.walk_forward_ok:
            out.append("walk-forward 未通过：样本外 Rank IC 接近随机")
        return out

    def to_dict(self) -> dict:
        return {"plateau_ok": self.plateau_ok, "dsr_ok": self.dsr_ok,
                "walk_forward_ok": self.walk_forward_ok, "passed": self.passed,
                "failures": self.failures(),
                "plateau": self.plateau_detail, "deflated_sharpe": self.dsr_detail,
                "walk_forward": self.walk_forward_detail}

    @classmethod
    def from_dict(cls, data: dict) -> "Gates":
        data = data or {}
        return cls(plateau_ok=bool(data.get("plateau_ok")), plateau_detail=data.get("plateau", {}) or {},
                   dsr_ok=bool(data.get("dsr_ok")), dsr_detail=data.get("deflated_sharpe", {}) or {},
                   walk_forward_ok=bool(data.get("walk_forward_ok")),
                   walk_forward_detail=data.get("walk_forward", {}) or {})


@dataclass
class StrategyRecord:
    id: str
    created_at: str
    genes: Dict[str, object]
    score: float
    status: str = CHALLENGER
    breakdown: Dict[str, object] = field(default_factory=dict)
    gates: Gates = field(default_factory=Gates)
    n_trials: int = 0
    window: List[str] = field(default_factory=list)
    frozen_at: Optional[str] = None
    promoted_at: Optional[str] = None
    retired_at: Optional[str] = None
    notes: str = ""
    review: Dict[str, object] = field(default_factory=dict)

    @staticmethod
    def new_id() -> str:
        return "st-" + uuid.uuid4().hex[:8]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gates"] = self.gates.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "StrategyRecord":
        data = dict(data or {})
        data["gates"] = Gates.from_dict(data.get("gates", {}))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


class StrategyArchive(JsonStore):
    """data/strategy_archive.json"""

    def __init__(self, path: Optional[Path] = None):
        super().__init__(path or (DATA_DIR / "strategy_archive.json"))

    # ---------- 读 ----------
    def _raw(self) -> dict:
        return self.read({"records": [], "champion_id": None}) or {"records": [], "champion_id": None}

    def records(self, status: Optional[str] = None) -> List[StrategyRecord]:
        items = [StrategyRecord.from_dict(d) for d in self._raw().get("records", [])]
        if status:
            items = [r for r in items if r.status == status]
        return sorted(items, key=lambda r: r.created_at, reverse=True)

    def get(self, record_id: str) -> Optional[StrategyRecord]:
        for r in self.records():
            if r.id == record_id:
                return r
        return None

    def champion(self) -> Optional[StrategyRecord]:
        cid = self._raw().get("champion_id")
        return self.get(cid) if cid else None

    # ---------- 写 ----------
    def add(self, record: StrategyRecord) -> StrategyRecord:
        raw = self._raw()
        records = raw.get("records", [])
        if not record.gates.passed and record.status == CHALLENGER:
            record.status = REJECTED
            record.notes = (record.notes + " " if record.notes else "") + \
                           "；".join(record.gates.failures()) + "（未入库为挑战者）"
        records.append(record.to_dict())
        raw["records"] = records
        self.write(raw)
        return record

    def promote(self, record_id: str, on_date: Optional[str] = None,
                note: str = "") -> Dict[str, object]:
        """
        人工晋升：把挑战者设为冠军，并**在此刻**记录 frozen_at。

        返回 ok=False 的情况：记录不存在、闸门没过、或已经样本外验证过（避免重复冻结）。
        """
        today = (on_date or _date.today().isoformat())[:10]
        raw = self._raw()
        records = raw.get("records", [])
        target = None
        for d in records:
            if d.get("id") != record_id:
                continue
            if not (d.get("gates") or {}).get("passed"):
                return {"ok": False, "reason": "该策略未通过三重闸门，不能晋升",
                        "failures": Gates.from_dict(d.get("gates", {})).failures()}
            if d.get("status") == CHAMPION and d.get("frozen_at"):
                return {"ok": False, "reason": f"该策略已在 {d['frozen_at']} 冻结晋升过"}
            d["status"] = CHAMPION
            d["frozen_at"] = today
            d["promoted_at"] = today
            d["notes"] = (d.get("notes", "") + " " + note).strip()
            target = d
        if target is None:
            return {"ok": False, "reason": f"找不到策略 {record_id}"}

        for d in records:
            if d.get("status") == CHAMPION and d.get("id") != record_id:
                d["status"] = RETIRED
                d["retired_at"] = today
        raw["champion_id"] = record_id
        raw["records"] = records
        self.write(raw)
        return {"ok": True, "champion": target, "frozen_at": today,
                "meaning": f"策略在 {today} 冻结。只有 {today} 之后公布的净值才算样本外证据，"
                           f"此前的任何成绩都是样本内。"}

    def update_review(self, record_id: str, review: Dict[str, object]) -> bool:
        raw = self._raw()
        changed = False
        for d in raw.get("records", []):
            if d.get("id") == record_id:
                d["review"] = review
                changed = True
        if changed:
            self.write(raw)
        return changed

    def summary(self) -> Dict[str, object]:
        recs = self.records()
        champ = self.champion()
        return {
            "total": len(recs),
            "champions": len([r for r in recs if r.status == CHAMPION]),
            "challengers": len([r for r in recs if r.status == CHALLENGER]),
            "rejected": len([r for r in recs if r.status == REJECTED]),
            "retired": len([r for r in recs if r.status == RETIRED]),
            "champion": champ.to_dict() if champ else None,
        }
