"""模拟盘与策略档案的 JSON 落盘（原子写）。

用文件而不是数据库：这个项目定位是"单机、免费、可离线"，
一个 JSON 文件足够，而且用户能直接打开看、能手工改坏再跑测试。
写入一律"先写临时文件再 replace"，避免进程被杀时留下半个文件。
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

from config import DATA_DIR
from paper.account import PaperAccount

logger = logging.getLogger("fund-agent")


class JsonStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def read(self, default):
        if not self.path.exists():
            return default
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[存储] {self.path.name} 读取失败，按空处理: {e}")
            return default

    def write(self, payload) -> bool:
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                os.replace(tmp, self.path)
                return True
            except Exception as e:
                logger.error(f"[存储] {self.path.name} 写入失败: {e}")
                try:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
                except OSError:
                    pass
                return False


class PaperStore(JsonStore):
    """模拟盘账户仓库（data/paper_accounts.json）。"""

    def __init__(self, path: Optional[Path] = None):
        super().__init__(path or (DATA_DIR / "paper_accounts.json"))

    def _raw(self) -> dict:
        return self.read({"accounts": []}) or {"accounts": []}

    def list_accounts(self) -> List[PaperAccount]:
        return [PaperAccount.from_dict(d) for d in self._raw().get("accounts", [])]

    def get(self, account_id: str) -> Optional[PaperAccount]:
        for d in self._raw().get("accounts", []):
            if d.get("id") == account_id:
                return PaperAccount.from_dict(d)
        return None

    def save(self, account: PaperAccount) -> bool:
        raw = self._raw()
        accounts = raw.get("accounts", [])
        payload = account.to_dict()
        for i, d in enumerate(accounts):
            if d.get("id") == account.id:
                accounts[i] = payload
                break
        else:
            accounts.append(payload)
        raw["accounts"] = accounts
        return self.write(raw)

    def remove(self, account_id: str) -> bool:
        raw = self._raw()
        raw["accounts"] = [a for a in raw.get("accounts", []) if a.get("id") != account_id]
        return self.write(raw)
