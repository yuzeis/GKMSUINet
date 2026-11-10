"""Version 2 features: every field of a card, an effect, a trigger and a card search the exam reads, and the card
search rule (which cards a search matches) for the relations between the things in view."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .feat import Featurizer, Vocab, card_origin, num_of, slog

SLOTS = 12
SLOT_CAT = ("etype", "phase", "fstat", "spos", "prange", "mpos",
            "fstat2", "tetype", "tspos", "egroup", "tegroup", "grow", "egroup2", "tegroup2")
SLOT_NUM = ("present", "v1", "v2", "count", "turn", "enchant", "target", "once", "nested", "cond", "fsval",
            "phval", "pickmax", "chain",
            # the condition (trigger)
            "fsval2", "fneg", "l_vo", "l_da", "l_vi", "l_sp",
            "t_search", "t_ids", "t_act", "t_men", "t_tro", "t_plan", "t_unique", "t_starter", "t_ngrp",
            "t_low", "t_zone", "t_limit",
            "t_r1", "t_r2", "t_r3", "t_r4", "t_r100", "t_stlo", "t_sthi", "t_u0", "t_u1", "t_u2", "t_u3",
            # the effect's own card search
            "s_ids", "s_act", "s_men", "s_tro", "s_plan", "s_unique", "s_starter", "s_ngrp", "s_limit",
            "s_random", "s_first", "s_pool", "s_self",
            "s_r1", "s_r2", "s_r3", "s_r4", "s_r100", "s_stlo", "s_sthi", "s_u0", "s_u1", "s_u2", "s_u3",
            # picks and grows
            "pickmin", "pickref", "shortage", "ngrow", "grow_v", "grow_play", "tupg")
THING_CAT = ("category", "rarity", "plan", "ctype", "mpos", "cfstat", "cphase", "origin")
THING_NUM_V1 = ("cost", "stamina", "force", "upgrade", "nodup", "initial", "ncust", "c_lesson", "c_cost_red",
                "c_cost_add", "c_count", "c_fpp", "c_pen", "c_turn", "ngrow", "senchant", "cfsval", "nfx",
                "phval", "fsval2")
VOCAB_PATH = Path(__file__).parent / "data" / "vocab.v2.json"
SLOT_FIELD = {"etype": "etype", "phase": "phase", "fstat": "fstat", "spos": "spos", "prange": "prange", "mpos": "mpos",
              "fstat2": "fstat", "tetype": "etype", "tspos": "spos", "egroup": "egroup", "tegroup": "egroup",
              "grow": "grow", "egroup2": "egroup", "tegroup2": "egroup"}
THING_FIELD = {"category": "category", "rarity": "rarity", "plan": "plan", "ctype": "ctype", "mpos": "mpos",
               "cfstat": "fstat", "cphase": "phase", "origin": "origin"}
LESSONS = (5, 6, 7, 3)                        # PRODUCE_STEP_LESSON_TYPE vocal / dance / visual / SP
CATEGORIES = (1, 2, 3)                        # active / mental / trouble
RARITIES = (1, 2, 3, 4, 100)                  # the master's card rarities (100: the event / special tier)
PLAY_PHASES = {2, 3, 11, 23, 25, 44}          # card play, after play, search play, play-count intervals
MOVE_PHASES = {29, 30, 31}                    # a card moved to the hand / grave / lost


def load_vocab2(path=VOCAB_PATH):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return Vocab(doc["fields"]), list(doc["card_groups"])


# ----------------------------------------------------------------------------- the card search rule
class Matcher:
    """Whether a card (id, upgrade) matches a ProduceCardSearch.  ANY_GROUP: a trigger's search takes its effect
    groups as alternatives (one is enough); a zone's needs all of them."""

    def __init__(self, master):
        self.m = master
        self._row = {}
        self._hit = {}

    def row(self, cid, up):
        key = (cid, up)
        r = self._row.get(key)
        if r is None:
            r = self.m.cards.get(key)
            if r is None:
                tiers = sorted(u for (c, u) in self.m.cards if c == cid)
                below = [t for t in tiers if t <= up]
                r = self.m.cards.get((cid, below[-1] if below else tiers[0])) if tiers else {}
            self._row[key] = r = r or {}
        return r

    def match(self, search_id, cid, up=0, any_group=False, upgrade=None, extra=()):
        """EXTRA: effect groups the card has beyond its master row (from its customizes and in-exam grows)."""
        key = (search_id, cid, up, any_group, upgrade, extra)
        hit = self._hit.get(key)
        if hit is None:
            hit = self._hit[key] = self._match(search_id, cid, up, any_group, upgrade, extra)
        return hit

    def _match(self, search_id, cid, up, any_group, upgrade, extra=()):
        s = self.m.searches.get(search_id)
        if s is None:
            return False
        row = self.row(cid, up)
        if not row:
            return False
        ids = s.get("produceCardIds") or ()
        if ids and cid not in ids:
            return False
        cats = [num_of(x) for x in s.get("cardCategories") or ()]
        if cats and num_of(row.get("category")) not in cats:
            return False
        rars = [num_of(x) for x in s.get("cardRarities") or ()]
        if rars and num_of(row.get("rarity")) not in rars:
            return False
        if s.get("planType") is not None and num_of(s.get("planType")) != num_of(row.get("planType")):
            return False
        groups = s.get("effectGroupIds") or ()
        if groups:
            own = set(row.get("effectGroupIds") or ()) | set(extra or ())
            hits = sum(1 for g in groups if g in own)
            if (hits == 0) if any_group else (hits < len(groups)):
                return False
        if s.get("cardSearchTag") and s.get("cardSearchTag") != row.get("searchTag"):
            return False
        upg = s.get("upgradeCounts") or ()
        if upgrade is not None and upgrade >= 0 and upg and int(upgrade) not in [int(x) for x in upg]:
            return False
        lo, hi = s.get("staminaMin"), s.get("staminaMax")
        if lo is not None or hi is not None:
            stamina = row.get("forceStamina") or row.get("stamina") or 0
            if lo is not None and stamina < int(lo):
                return False
            if hi is not None and stamina > int(hi):
                return False
        return True


def _freeze(x):
    return tuple(_freeze(y) for y in x) if isinstance(x, (list, tuple)) else x


def card_key(c):
    """A hashable key of a card in view: (id, upgrade[, customizes, grows])."""
    u = int(c.get("u", 0) or 0)
    if not c.get("c") and not c.get("g"):
        return (c["id"], u)
    return (c["id"], u, _freeze(c.get("c")), _freeze(c.get("g")))


# ----------------------------------------------------------------------------- slots
def _search_nums(s, prefix):
    """The filter numbers of a card search (zeros without one)."""
    if not s:
        return {}
    cats = {num_of(x) for x in s.get("cardCategories") or ()}
    rars = {num_of(x) for x in s.get("cardRarities") or ()}
    upg = {int(x) for x in s.get("upgradeCounts") or ()}
    tag = s.get("cardSearchTag")
    out = {prefix + "ids": min(3, len(s.get("produceCardIds") or ())) / 3.0,
           prefix + "act": float(1 in cats), prefix + "men": float(2 in cats), prefix + "tro": float(3 in cats),
           prefix + "plan": float(num_of(s.get("planType")) or 0) / 4.0,
           prefix + "unique": float(tag == "idol-unique"), prefix + "starter": float(tag == "starter"),
           prefix + "ngrp": len(s.get("effectGroupIds") or ()) / 2.0, prefix + "limit": slog(s.get("limitCount")),
           prefix + "stlo": slog(s.get("staminaMin")), prefix + "sthi": slog(s.get("staminaMax"))}
    for r in RARITIES:
        out[f"{prefix}r{r}"] = float(r in rars)
    for u in range(4):
        out[f"{prefix}u{u}"] = float(u in upg)
    if prefix == "s_":
        order = num_of(s.get("orderType"))
        out.update(s_random=float(order == 3), s_first=float(order == 1),
                   s_pool=float(bool(s.get("produceCardPoolId") or s.get("produceCardRandomPoolId"))),
                   s_self=float(bool(s.get("isSelf"))))
    return out


class Featurizer2(Featurizer):
    """Version 2 arrays of cards, enchants, drinks and reserved effects."""

    def __init__(self, master, vocab=None, card_groups=None):
        if vocab is None:
            vocab, card_groups = load_vocab2()
        super().__init__(master, vocab)
        self.groups = list(card_groups)
        self.group_ix = {g: i for i, g in enumerate(self.groups)}
        self.thing_num = THING_NUM_V1 + tuple(f"g{i}" for i in range(len(self.groups))) + (
            "e_l_vo", "e_l_da", "e_l_vi", "e_l_sp", "e_fneg", "e_filter", "e_play", "e_move")
        self.matcher = Matcher(self.m)
        self._xg = {}
        self._fx = {}
        self._arch = None

    # -- the condition of a slot
    def _cond(self, trigger_id):
        """(categoricals, numbers) of a trigger, as the slot's condition."""
        t = self.m.triggers.get(trigger_id) if trigger_id else None
        cat = {"phase": 0, "fstat": 0, "fstat2": 0, "tetype": 0, "tspos": 0, "tegroup": 0, "tegroup2": 0}
        nums = {}
        if not t:
            return cat, nums, 0.0, 0.0
        ph = [num_of(x) for x in t.get("phaseTypes") or ()]
        fs = [num_of(x) for x in t.get("fieldStatusTypes") or ()]
        fsv = list(t.get("fieldStatusValues") or ())
        phv = list(t.get("phaseValues") or ())
        et = [num_of(x) for x in t.get("effectTypes") or ()]
        cat.update(phase=self.v("phase", ph[0]) if ph else 0, fstat=self.v("fstat", fs[0]) if fs else 0,
                   fstat2=self.v("fstat", fs[1]) if len(fs) > 1 else 0, tetype=self.v("etype", et[0]) if et else 0)
        lesson = num_of(t.get("lessonType"))
        nums.update(fsval2=slog(fsv[1]) if len(fsv) > 1 else 0.0,
                    fneg=float(bool(t.get("fieldStatusCheckTypes"))),
                    l_vo=float(lesson == 5), l_da=float(lesson == 6), l_vi=float(lesson == 7), l_sp=float(lesson == 3),
                    t_low=slog(t.get("lowerSearchCount")))
        sid = t.get("produceCardSearchId")
        zone = list(t.get("fieldStatusProduceCardSearchIds") or ())
        s = self.m.searches.get(sid) if sid else (self.m.searches.get(zone[0]) if zone else None)
        if s:
            nums["t_search"] = 1.0
            nums["t_zone"] = float(not sid)
            nums.update(_search_nums(s, "t_"))
            cat["tspos"] = self.v("spos", num_of(s.get("cardPositionType")))
            g = s.get("effectGroupIds") or ()
            cat["tegroup"] = self.v("egroup", g[0]) if g else 0
            cat["tegroup2"] = self.v("egroup", g[1]) if len(g) > 1 else 0
        return cat, nums, (slog(fsv[0]) if fsv else 0.0), (slog(phv[0]) if phv else 0.0)

    def _slot(self, effect_id, out, nested=0, cond_trigger=None, once=False, value_adds=None, turn_adds=None):
        if len(out) >= SLOTS:
            return
        e = self.m.effects.get(effect_id)
        if e is None:
            c = [0] * len(SLOT_CAT)
            c[0] = 1
            n = [0.0] * len(SLOT_NUM)
            n[0] = 1.0
            out.append((c, n))
            return
        et = num_of(e.get("effectType"))
        v1 = e.get("effectValue1")
        if v1 is None and e.get("targetUpgradeCount") is not None:
            v1 = e.get("targetUpgradeCount")
        if value_adds and v1 is not None:
            v1 = int(v1) + int(value_adds.get(et, value_adds.get(str(et), 0)) or 0)
        turn = e.get("effectTurn")
        if turn_adds and turn is not None:
            turn = int(turn) + int(turn_adds.get(et, turn_adds.get(str(et), 0)) or 0)
        ccat, cnum, fsv, phv = self._cond(cond_trigger)
        search = self.m.searches.get(e.get("produceCardSearchId")) if e.get("produceCardSearchId") else None
        grows = [self.m.grow_effects.get(g) or {} for g in e.get("produceCardGrowEffectIds") or ()]
        sg = (search or {}).get("effectGroupIds") or ()
        cat = {"etype": self.v("etype", et),
               "spos": self.v("spos", num_of(search.get("cardPositionType"))) if search else 0,
               "prange": self.v("prange", num_of(e.get("pickRangeType"))) if e.get("pickRangeType") is not None else 0,
               "mpos": self.v("mpos", num_of(e.get("movePositionType"))) if e.get("movePositionType") is not None else 0,
               "egroup": self.v("egroup", sg[0]) if sg else 0,
               "egroup2": self.v("egroup", sg[1]) if len(sg) > 1 else 0,
               "grow": self.v("grow", num_of(grows[0].get("effectType"))) if grows else 0}
        cat.update(ccat)
        chain = bool(e.get("chainProduceExamEffectId") or e.get("chainProduceExamEffectIds"))
        num = {"present": 1.0, "v1": slog(v1), "v2": slog(e.get("effectValue2")), "count": slog(e.get("effectCount")),
               "turn": slog(turn), "enchant": float(bool(e.get("produceExamStatusEnchantId"))),
               "target": float(bool(e.get("targetProduceCardId"))), "once": float(bool(once)), "nested": nested / 3.0,
               "cond": float(cond_trigger is not None), "fsval": fsv, "phval": phv,
               "pickmax": min(5.0, float(e.get("pickCountMax") or 0)) / 5.0, "chain": float(chain),
               "pickmin": min(5.0, float(e.get("pickCountMin") or 0)) / 5.0,
               "pickref": float(bool(e.get("pickCountReferenceProduceCardSearchId"))),
               "shortage": float(num_of(e.get("pickCountType")) == 2),
               "ngrow": len(grows) / 3.0, "grow_v": slog(grows[0].get("value")) if grows else 0.0,
               "grow_play": float(any(g.get("playProduceExamEffectId") for g in grows)),
               "tupg": slog(e.get("targetUpgradeCount"))}
        num.update(cnum)
        num.update(_search_nums(search, "s_"))
        out.append(([int(cat.get(f, 0)) for f in SLOT_CAT], [float(num.get(f, 0.0)) for f in SLOT_NUM]))
        if nested < 2:
            enchant = e.get("produceExamStatusEnchantId")
            if enchant:
                self._enchant_slots(enchant, out, nested + 1)
            for chained in ([e["chainProduceExamEffectId"]] if e.get("chainProduceExamEffectId") else []) + list(
                    e.get("chainProduceExamEffectIds") or ()):
                self._slot(chained, out, nested + 1)
            for g in grows[1:]:                        # the other grows, one nested slot each
                if len(out) >= SLOTS:
                    break
                c = [0] * len(SLOT_CAT)
                c[SLOT_CAT.index("grow")] = self.v("grow", num_of(g.get("effectType")))
                n = [0.0] * len(SLOT_NUM)
                n[0], n[SLOT_NUM.index("nested")] = 1.0, (nested + 1) / 3.0
                n[SLOT_NUM.index("grow_v")] = slog(g.get("value"))
                out.append((c, n))
            for g in grows:                            # a grow that plays an effect: that effect, nested
                if g.get("playProduceExamEffectId"):
                    self._slot(g["playProduceExamEffectId"], out, nested + 1)

    def _pack(self, cat, numv, slots):
        sc = np.zeros((SLOTS, len(SLOT_CAT)), np.int16)
        sn = np.zeros((SLOTS, len(SLOT_NUM)), np.float32)
        for i, (c, n) in enumerate(slots[:SLOTS]):
            sc[i] = c
            sn[i] = n
        return (np.asarray(cat, np.int16), np.asarray(numv, np.float32), sc, sn)

    def _widen(self, numv, groups=(), trig=None):
        """The card-level numbers + the card's effect groups + the enchant trigger's fields."""
        g = [0.0] * len(self.groups)
        for x in groups or ():
            i = self.group_ix.get(x)
            if i is not None:
                g[i] = 1.0
        t = self.m.triggers.get(trig) if trig else None
        e = [0.0] * 8
        if t:
            lesson = num_of(t.get("lessonType"))
            ph = {num_of(x) for x in t.get("phaseTypes") or ()}
            has = bool(t.get("produceCardSearchId") or t.get("fieldStatusProduceCardSearchIds"))
            e = [float(lesson == 5), float(lesson == 6), float(lesson == 7), float(lesson == 3),
                 float(bool(t.get("fieldStatusCheckTypes"))), float(has),
                 float(bool(t.get("produceCardSearchId")) and bool(ph & PLAY_PHASES)),
                 float(bool(t.get("produceCardSearchId")) and bool(ph & MOVE_PHASES))]
        return list(numv) + g + e

    # -- cards
    def _card(self, spec):
        rc = self._resolved(spec)
        if rc is None:
            cat = [0] * len(THING_CAT)
            cat[THING_CAT.index("origin")] = self.v("origin", card_origin(spec["id"]))
            n = [0.0] * len(SLOT_NUM)
            n[0] = 1.0
            return self._pack(cat, [0.0] * len(self.thing_num), [([1] + [0] * (len(SLOT_CAT) - 1), n)])
        tc, tn, _sc, _sn = super()._card(spec)          # the card-level fields; the slots are rebuilt below
        slots = []
        value_adds = rc.get("customize_value_adds") or {}
        turn_adds = rc.get("customize_turn_adds") or {}
        lesson_add = int(rc.get("customize_lesson_add") or 0)
        if lesson_add:
            value_adds = dict(value_adds)
            for t in (1, 56, 23):
                value_adds[t] = int(value_adds.get(t, 0) or 0) + lesson_add
        for link in rc.get("playEffects") or ():
            self._slot(link.get("produceExamEffectId"), slots, 0, link.get("produceExamTriggerId"),
                       link.get("isOncePlayEffect"), value_adds, turn_adds)
        senchant = rc.get("produceCardStatusEnchantId")
        if senchant:
            self._enchant_slots(senchant, slots, 1)
        groups = list(rc.get("effectGroupIds") or ())
        for gid in spec.get("g") or ():                 # in-exam grows bring their effect groups
            groups += list((self.m.grow_effects.get(gid) or {}).get("effectGroupIds") or ())
        return self._pack(tc, self._widen(tn, groups, rc.get("playProduceExamTriggerId")), slots)

    # -- enchants
    def _enchant(self, enchant_id):
        tc, tn, _sc, _sn = super()._enchant(enchant_id)
        en = self.m.enchants.get(enchant_id) or {}
        trig = en.get("produceExamTriggerId")
        slots = []
        for eid in en.get("produceExamEffectIds") or ():
            self._slot(eid, slots, 0, cond_trigger=trig)
        return self._pack(tc, self._widen(tn, (), trig), slots)

    # -- drinks
    def _drink(self, drink_id):
        tc, tn, _sc, _sn = super()._drink(drink_id)
        d = self.m.drinks.get(drink_id) or {}
        slots = []
        for deid in d.get("produceDrinkEffectIds") or ():
            de = self.m.drink_effects.get(deid) or {}
            self._slot(de.get("produceExamEffectId"), slots, 0)
        return self._pack(tc, self._widen(tn, d.get("effectGroupIds") or ()), slots)

    # -- a reserved effect ("timers")
    def effect(self, effect_id):
        key = "fx:" + str(effect_id)
        hit = self._ench.get(key)
        if hit is None:
            slots = []
            self._slot(effect_id, slots, 0)
            cat = [0] * len(THING_CAT)
            if effect_id not in self.m.effects:
                cat[0] = 1
            hit = self._ench[key] = self._pack(cat, [0.0] * len(self.thing_num), slots)
        return hit

    # -- relations
    def trigger_search(self, enchant_id):
        """(search id, phase kind) of an enchant's card-watching trigger: kind "play" or "move"; (None, None)."""
        en = self.m.enchants.get(enchant_id) or {}
        t = self.m.triggers.get(en.get("produceExamTriggerId")) or {}
        sid = t.get("produceCardSearchId")
        if not sid:
            return None, None
        ph = {num_of(x) for x in t.get("phaseTypes") or ()}
        return sid, "play" if ph & PLAY_PHASES else "move" if ph & MOVE_PHASES else None

    def extra_groups(self, c):
        """The effect groups a card in view has beyond its master row: those its customizes and in-exam grows add."""
        if not c.get("c") and not c.get("g"):
            return ()
        key = card_key(c)
        hit = self._xg.get(key)
        if hit is None:
            spec = {k: c[k] for k in ("id", "u", "c", "g") if k in c}
            rc = self._resolved(spec) or {}
            row = self.matcher.row(c["id"], int(c.get("u", 0) or 0))
            own = set(row.get("effectGroupIds") or ())
            groups = set(rc.get("effectGroupIds") or ())
            for gid in c.get("g") or ():
                groups |= set((self.m.grow_effects.get(gid) or {}).get("effectGroupIds") or ())
            hit = self._xg[key] = tuple(sorted(groups - own))
        return hit

    def _effect_searches(self, effect_id, out, depth=0):
        e = self.m.effects.get(effect_id) or {}
        if e.get("produceCardSearchId"):
            out.add(e["produceCardSearchId"])
        if e.get("pickCountReferenceProduceCardSearchId"):
            out.add(e["pickCountReferenceProduceCardSearchId"])
        if depth >= 2:
            return
        en = self.m.enchants.get(e.get("produceExamStatusEnchantId")) or {}
        for x in en.get("produceExamEffectIds") or ():
            self._effect_searches(x, out, depth + 1)
        for x in ([e["chainProduceExamEffectId"]] if e.get("chainProduceExamEffectId") else []) + list(
                e.get("chainProduceExamEffectIds") or ()):
            self._effect_searches(x, out, depth + 1)

    def effect_searches(self, kind, key):
        """The card searches a thing's effects act on (the cards they move, upgrade, change, count): KIND "card"
        (KEY a card in view), "enchant" (an enchant id) or "effect" (a reserved effect id).  For relations: which
        cards in view an effect of this thing reaches."""
        if kind == "card":
            ck = card_key(key)
        else:
            ck = (kind, key)
        hit = self._fx.get(ck)
        if hit is None:
            out = set()
            if kind == "card":
                rc = self._resolved({k: key[k] for k in ("id", "u", "c", "g") if k in key}) or {}
                for link in rc.get("playEffects") or ():
                    self._effect_searches(link.get("produceExamEffectId"), out)
                en = self.m.enchants.get(rc.get("produceCardStatusEnchantId")) or {}
                for x in en.get("produceExamEffectIds") or ():
                    self._effect_searches(x, out, 1)
            elif kind == "enchant":
                for x in (self.m.enchants.get(key) or {}).get("produceExamEffectIds") or ():
                    self._effect_searches(x, out)
            else:
                self._effect_searches(key, out)
            hit = self._fx[ck] = tuple(sorted(out))
        return hit

    def condition_searches(self, c):
        """The zone searches a card's play condition counts (the fieldStatusProduceCardSearchIds of its play trigger
        and of its play effects' triggers): which cards in view decide whether it can be played or an effect fires."""
        ck = ("cond",) + card_key(c)
        hit = self._fx.get(ck)
        if hit is None:
            rc = self._resolved({k: c[k] for k in ("id", "u", "c", "g") if k in c}) or {}
            out = set()
            for tid in [rc.get("playProduceExamTriggerId")] + [l.get("produceExamTriggerId")
                                                               for l in rc.get("playEffects") or ()]:
                t = self.m.triggers.get(tid) or {}
                out.update(t.get("fieldStatusProduceCardSearchIds") or ())
            hit = self._fx[ck] = tuple(sorted(out))
        return hit

    # -- the idol card's archetype
    def arch(self, idol):
        """'PLAN1/EXAM_LESSON_BUFF' and the like for an IdolCard id, None when unknown."""
        if self._arch is None:
            self._arch = json.loads((Path(__file__).parent / "data" / "arch.json").read_text(encoding="utf-8"))
        return self._arch.get(idol) if idol else None
