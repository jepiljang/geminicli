"""library.py 단위 테스트."""

from datetime import date

import pytest

from recommendation.exemplar.library import (
    Exemplar,
    ExemplarLibrary,
    make_exemplar_id,
)


def test_make_exemplar_id_format():
    eid = make_exemplar_id("NVDA", date(2023, 1, 1), "AI Rally")
    assert eid.startswith("nvda-2023-")
    assert eid == eid.lower()
    assert " " not in eid


def test_make_exemplar_id_uses_ticker_when_no_name():
    eid = make_exemplar_id("AAPL", date(2020, 4, 1), None)
    assert eid == "aapl-2020-aapl" or eid.startswith("aapl-2020-")


def test_library_add_creates_entry(tmp_path):
    lib = ExemplarLibrary(tmp_path / "exemplars.json")
    ex = Exemplar(
        id="nvda-2023-rally",
        ticker="NVDA",
        name="NVDA 2023 AI 랠리",
        start_date=date(2023, 1, 1),
        end_date=date(2023, 7, 1),
        active=True,
        created_at="2026-05-17T14:30:00",
        profile_path="data/exemplars/profiles/nvda-2023-rally.parquet",
    )
    lib.add(ex)
    assert lib.get("nvda-2023-rally") == ex


def test_library_persists_to_json(tmp_path):
    json_path = tmp_path / "exemplars.json"
    lib1 = ExemplarLibrary(json_path)
    ex = Exemplar(
        id="aapl-2020", ticker="AAPL", name="test",
        start_date=date(2020, 4, 1), end_date=date(2020, 9, 1),
        active=True, created_at="2026-05-17T00:00:00",
        profile_path="x.parquet",
    )
    lib1.add(ex)

    lib2 = ExemplarLibrary(json_path)
    assert lib2.get("aapl-2020") == ex


def test_library_toggle_active(tmp_path):
    lib = ExemplarLibrary(tmp_path / "exemplars.json")
    ex = Exemplar(
        id="x", ticker="X", name="x", start_date=date(2020, 1, 1),
        end_date=date(2020, 6, 1), active=True, created_at="t",
        profile_path="p",
    )
    lib.add(ex)
    lib.toggle_active("x", False)
    assert lib.get("x").active is False
    lib.toggle_active("x", True)
    assert lib.get("x").active is True


def test_library_delete(tmp_path):
    lib = ExemplarLibrary(tmp_path / "exemplars.json")
    ex = Exemplar(
        id="x", ticker="X", name="x", start_date=date(2020, 1, 1),
        end_date=date(2020, 6, 1), active=True, created_at="t",
        profile_path="p",
    )
    lib.add(ex)
    lib.delete("x")
    assert lib.get("x") is None


def test_library_list_active_only(tmp_path):
    lib = ExemplarLibrary(tmp_path / "exemplars.json")
    a = Exemplar(id="a", ticker="A", name="a", start_date=date(2020, 1, 1),
                 end_date=date(2020, 6, 1), active=True, created_at="t",
                 profile_path="p1")
    b = Exemplar(id="b", ticker="B", name="b", start_date=date(2020, 1, 1),
                 end_date=date(2020, 6, 1), active=False, created_at="t",
                 profile_path="p2")
    lib.add(a)
    lib.add(b)
    actives = lib.list_all(active_only=True)
    assert [e.id for e in actives] == ["a"]
    everything = lib.list_all()
    assert {e.id for e in everything} == {"a", "b"}


def test_library_duplicate_id_raises(tmp_path):
    lib = ExemplarLibrary(tmp_path / "exemplars.json")
    ex = Exemplar(id="x", ticker="X", name="x", start_date=date(2020, 1, 1),
                  end_date=date(2020, 6, 1), active=True, created_at="t",
                  profile_path="p")
    lib.add(ex)
    with pytest.raises(ValueError, match="already exists"):
        lib.add(ex)


def test_library_get_missing_returns_none(tmp_path):
    lib = ExemplarLibrary(tmp_path / "exemplars.json")
    assert lib.get("nope") is None


def test_library_delete_missing_raises(tmp_path):
    lib = ExemplarLibrary(tmp_path / "exemplars.json")
    with pytest.raises(KeyError):
        lib.delete("nope")
