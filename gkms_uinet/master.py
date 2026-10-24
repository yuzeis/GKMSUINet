"""The master tables the features read, from a directory of .jsonl or .yaml files."""
from __future__ import annotations

import bisect
import json
import re
from pathlib import Path

TABLES = ("ProduceCard", "ProduceCardCustomize", "ProduceCardGrowEffect", "ProduceCardSearch", "ProduceExamEffect",
          "ProduceExamTrigger", "ProduceExamStatusEnchant", "ProduceDrink", "ProduceDrinkEffect")
_ENUM_VALUES = json.loads((Path(__file__).parent / "data" / "enums.json").read_text(encoding="utf-8"))["values"]
_ENUM_KEYS = sorted(_ENUM_VALUES)
_ENUM_RE = re.compile(r"([A-Z][A-Za-z0-9]*)_([A-Za-z0-9_]+)")
_DEFAULTS = ("", 0, False, None)


def _norm(s):
    return s.replace("_", "").lower()


def _is_enum_type(prefix):
    p = _norm(prefix)
    i = bisect.bisect_left(_ENUM_KEYS, p)
    return i < len(_ENUM_KEYS) and _ENUM_KEYS[i].startswith(p)


def _enum(s):
    """An enum name as its number; -1 for a value of a known enum this package does not know; else the string."""
    m = _ENUM_RE.fullmatch(s)
    if not m:
        return s
    if _norm(s) in _ENUM_VALUES:
        return _ENUM_VALUES[_norm(s)]
    if m.group(2) == "Unknown":
        return 0
    return -1 if _is_enum_type(m.group(1)) else s


def _plain(v):
    """A row written with every field (enum names, zeros, empty strings) as the protobuf form, where a field at its
    default is absent and an enum is its number."""
    if isinstance(v, dict):
        if set(v) == {"name", "number"}:
            return v
        out = {}
        for k, x in v.items():
            x = _plain(x)
            if isinstance(x, (list, dict)) and not x:
                continue
            if not isinstance(x, (list, dict)) and x in _DEFAULTS:
                continue
            out[k] = x
        return out
    if isinstance(v, list):
        return [_plain(x) for x in v]
    if isinstance(v, str):
        return _enum(v)
    return v


def _rows(directory, table):
    for suffix in (".jsonl", ".json", ".yaml", ".yml"):
        path = directory / (table + suffix)
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"{table}.jsonl / .json / .yaml not in {directory}")
    if suffix == ".jsonl":
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        items = json.loads(path.read_text(encoding="utf-8"))
    else:
        import yaml
        with path.open(encoding="utf-8") as fh:
            items = yaml.load(fh, Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader)) or []
    for item in items:
        if "decoded" in item:                         # {"decode_status", "decoded", "outer"} rows
            if item.get("decode_status", "decoded") != "decoded":
                continue
            row = dict(item["decoded"])
            outer = item.get("outer") or {}
            if "UpgradeCount" in outer:
                row.setdefault("upgradeCount", outer["UpgradeCount"])
            yield row
        else:
            yield _plain(item)


class Master:
    def __init__(self, directory):
        d = Path(directory)
        rows = {t: list(_rows(d, t)) for t in TABLES}
        self.cards = {}
        for r in rows["ProduceCard"]:
            up = int(r.pop("upgradeCount", 0) or 0)
            self.cards[(r["id"], up)] = r
        self.customizes = {(r["id"], int(r.get("customizeCount") or 0)): r for r in rows["ProduceCardCustomize"]}
        self.grow_effects = {r["id"]: r for r in rows["ProduceCardGrowEffect"]}
        self.searches = {r["id"]: r for r in rows["ProduceCardSearch"]}
        self.effects = {r["id"]: r for r in rows["ProduceExamEffect"]}
        self.triggers = {r["id"]: r for r in rows["ProduceExamTrigger"]}
        self.enchants = {r["id"]: r for r in rows["ProduceExamStatusEnchant"]}
        self.drinks = {r["id"]: r for r in rows["ProduceDrink"]}
        self.drink_effects = {r["id"]: r for r in rows["ProduceDrinkEffect"]}
