"""모범 라이브러리 CRUD.

exemplars.json 형식:
{
  "exemplars": [
    {"id": ..., "ticker": ..., "name": ..., "start_date": "YYYY-MM-DD",
     "end_date": "YYYY-MM-DD", "active": true, "created_at": "ISO8601",
     "profile_path": "..."}
  ]
}
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path


@dataclass
class Exemplar:
    id: str
    ticker: str
    name: str
    start_date: date
    end_date: date
    active: bool
    created_at: str
    profile_path: str


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def make_exemplar_id(ticker: str, start: date, name: str | None) -> str:
    """티커 + 시작연도 + 이름 슬러그로 ID 생성."""
    slug_source = (name or ticker).lower()
    slug = _SLUG_RE.sub("-", slug_source).strip("-") or "exemplar"
    return f"{ticker.lower()}-{start.year}-{slug}"


def _exemplar_to_dict(ex: Exemplar) -> dict:
    d = asdict(ex)
    d["start_date"] = ex.start_date.isoformat()
    d["end_date"] = ex.end_date.isoformat()
    return d


def _exemplar_from_dict(d: dict) -> Exemplar:
    return Exemplar(
        id=d["id"],
        ticker=d["ticker"],
        name=d["name"],
        start_date=date.fromisoformat(d["start_date"]),
        end_date=date.fromisoformat(d["end_date"]),
        active=d["active"],
        created_at=d["created_at"],
        profile_path=d["profile_path"],
    )


class ExemplarLibrary:
    """exemplars.json 기반 라이브러리."""

    def __init__(self, json_path: Path):
        self.json_path = Path(json_path)
        self._items: dict[str, Exemplar] = {}
        self._load()

    def _load(self) -> None:
        if not self.json_path.exists():
            return
        with self.json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data.get("exemplars", []):
            ex = _exemplar_from_dict(d)
            self._items[ex.id] = ex

    def _save(self) -> None:
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"exemplars": [_exemplar_to_dict(e) for e in self._items.values()]}
        with self.json_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(self, exemplar: Exemplar) -> None:
        if exemplar.id in self._items:
            raise ValueError(f"exemplar id {exemplar.id!r} already exists")
        self._items[exemplar.id] = exemplar
        self._save()

    def get(self, exemplar_id: str) -> Exemplar | None:
        return self._items.get(exemplar_id)

    def delete(self, exemplar_id: str) -> None:
        if exemplar_id not in self._items:
            raise KeyError(exemplar_id)
        del self._items[exemplar_id]
        self._save()

    def toggle_active(self, exemplar_id: str, active: bool) -> None:
        if exemplar_id not in self._items:
            raise KeyError(exemplar_id)
        ex = self._items[exemplar_id]
        self._items[exemplar_id] = Exemplar(**{**asdict(ex), "active": active,
                                                "start_date": ex.start_date,
                                                "end_date": ex.end_date})
        self._save()

    def list_all(self, active_only: bool = False) -> list[Exemplar]:
        items = list(self._items.values())
        if active_only:
            items = [e for e in items if e.active]
        return items
