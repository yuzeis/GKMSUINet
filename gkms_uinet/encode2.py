"""A screen observation (UIObs v2) as the version 2 network's input arrays, a batch of one.

Beside the version 1 arrays (wider), ``t_rel`` says which thing in view acts on which card:
  bit 0  a lasting effect's card-play trigger watches the card
  bit 1  a lasting effect's card-move trigger watches the card
  bit 2  an effect of a lasting effect reaches the card through its card search
  bit 3  an effect of a card reaches the card
  bit 4  a card's play condition counts the card
  bit 5  a reserved effect reaches the card
"""
from __future__ import annotations

import numpy as np

from .encode import QUANTILES, VALUE_SHIFT, ZONES, _LONG_WHEN_UNLIMITED, _left, global_features  # noqa: F401
from .encode import KIND_DRINK, KIND_ENCH, KIND_HAND, KIND_PLAYED
from .feat import slog
from .feat2 import SLOT_CAT, SLOT_NUM, SLOTS, THING_CAT, card_key
from .obs import INF

KIND_TIMER = 9
T_EXT = 18
G_NUM = 52
INPUTS = ("g_num", "g_cat", "st_kind", "st_num", "t_kind", "t_cat", "t_num", "s_cat", "s_num", "t_ext", "a_tok",
          "t_rel")


def _card(c):
    return {k: c[k] for k in ("id", "u", "c", "g") if k in c}


def encode2(o, f):
    """The input arrays of observation O; F is a Featurizer2."""
    v, mt = f.v, f.matcher
    stance = (o.get("stance") or [None, 0])[0]
    g_cat = [v("plan", o["exam"].get("plan")), v("step", o["exam"].get("step")),
             v("produce", o["exam"].get("produce")), v("stance", stance), v("arch", f.arch(o["exam"].get("idol")))]
    st = o.get("status") or []
    st_kind = np.zeros((1, max(1, len(st))), np.int64)
    st_num = np.zeros((1, max(1, len(st)), 3), np.float32)
    for i, s in enumerate(st):
        turns = 20 if s[2] == INF and s[0] in _LONG_WHEN_UNLIMITED else s[2]
        left, inf = _left(turns)
        st_kind[0, i] = v("status", s[0])
        st_num[0, i] = (slog(s[1]), left, inf)

    # -- the things in view
    kinds, arrays, ext, cards = [], [], [], []            # cards: (token, card) of every card in view
    for i, c in enumerate(o["hand"]):
        kinds.append(KIND_HAND)
        arrays.append(f.card(_card(c)))
        ext.append([min(i, 9) / 9.0, float(c.get("p", 0)), 0, 0, 0, 0] + [0.0] * 12)
        cards.append((len(kinds) - 1, c))
    for d in o.get("drinks") or ():
        kinds.append(KIND_DRINK)
        arrays.append(f.drink(d))
        ext.append([0, 1.0] + [0.0] * (T_EXT - 2))
    for zone, kind in ZONES:
        for c in o.get(zone) or ():
            kinds.append(kind)
            arrays.append(f.card(_card(c)))
            ext.append([0.0] * T_EXT)
            cards.append((len(kinds) - 1, c))
    ench_tok = []
    for e in o.get("ench") or ():
        ul, ui = _left(e[1])
        tl, ti = _left(e[2])
        kinds.append(KIND_ENCH)
        arrays.append(f.enchant(e[0]))
        ext.append([0, 0, ul, ui, tl, ti] + [0.0] * 12)
        ench_tok.append((len(kinds) - 1, e))
    timer_tok = []
    soonest = None
    for turns, effect_id, at_end in o.get("timers") or ():
        kinds.append(KIND_TIMER)
        arrays.append(f.effect(effect_id))
        tl, ti = _left(turns)
        ext.append([0, 0, 0, 0, tl, ti] + [0.0] * 11 + [float(at_end)])
        timer_tok.append((len(kinds) - 1, effect_id))
        soonest = turns if soonest is None else min(soonest, turns)

    # -- the card search rule over the cards in view, once per distinct card and search
    distinct = {}
    for t, c in cards:
        k = card_key(c)
        if k not in distinct:
            distinct[k] = (c, f.extra_groups(c), [])
        distinct[k][2].append(t)
    memo = {}

    def matched(sid, any_group, zone):
        key = (sid, any_group, zone)
        hit = memo.get(key)
        if hit is None:
            hit = []
            for c, xg, toks in distinct.values():
                up = int(c.get("u", 0) or 0)
                if mt.match(sid, c["id"], up, any_group, up if zone else None, xg):
                    hit.extend(toks)
            memo[key] = hit
        return hit

    hand_toks = {t for t, _c in cards[:len(o["hand"])]}
    not_played = {t for t, _c in cards} - {t for t, k in enumerate(kinds) if k == KIND_PLAYED}
    edges = set()
    ptrig, mtrig = {}, {}
    any_watched = 0.0
    for t, e in ench_tok:
        sid, kind = f.trigger_search(e[0])
        if sid and kind:
            hit = matched(sid, True, False)
            count = ptrig if kind == "play" else mtrig
            for j in hit:
                edges.add((t, j, 0 if kind == "play" else 1))
                count[j] = count.get(j, 0) + 1
            nh = sum(1 for j in hit if j in hand_toks)
            nv = sum(1 for j in hit if j in not_played)
            ext[t][15], ext[t][16] = min(5, nh) / 5.0, min(10, nv) / 10.0
            any_watched = max(any_watched, float(nh > 0))
        for sid in f.effect_searches("enchant", e[0]):
            for j in matched(sid, False, True):
                edges.add((t, j, 2))
    for t, c in cards:
        for sid in f.effect_searches("card", c):
            for j in matched(sid, False, True):
                edges.add((t, j, 3))
        for sid in f.condition_searches(c):
            for j in matched(sid, False, False):
                edges.add((t, j, 4))
    for t, eid in timer_tok:
        for sid in f.effect_searches("effect", eid):
            for j in matched(sid, False, True):
                edges.add((t, j, 5))

    # -- cost changes and play-count buffs on each card, with their lifetimes
    costs = [tuple(x) for x in o.get("costs") or ()]
    pcount = [tuple(x) for x in o.get("pcount") or ()]
    cost_hit = [None if sid is None else set(matched(sid, False, True)) for sid, _v, _l in costs]
    rep_hit = [set(matched(sid, False, False)) for sid, _x, _u, _t in pcount]
    for t, _c in cards:
        row = ext[t]
        row[6], row[7] = min(3, ptrig.get(t, 0)) / 3.0, min(3, mtrig.get(t, 0)) / 3.0
        for (_sid, value, left), hit in zip(costs, cost_hit):
            if hit is None or t in hit:
                row[8], row[9], row[10] = 1.0, slog(value), min(20, int(left)) / 10.0
                break
        best = None
        for buff, hit in zip(pcount, rep_hit):
            if t in hit and (best is None or int(buff[1]) > int(best[1])):
                best = buff
        if best is not None:
            tl, ti = _left(best[3])
            row[11], row[12], row[13], row[14] = min(2, int(best[1])) / 2.0, min(20, int(best[2])) / 10.0, tl, ti

    T = max(1, len(kinds))
    t_kind = np.zeros((1, T), np.int64)
    t_cat = np.zeros((1, T, len(THING_CAT)), np.int64)
    t_num = np.zeros((1, T, len(f.thing_num)), np.float32)
    s_cat = np.zeros((1, T, SLOTS, len(SLOT_CAT)), np.int64)
    s_num = np.zeros((1, T, SLOTS, len(SLOT_NUM)), np.float32)
    t_ext = np.zeros((1, T, T_EXT), np.float32)
    for i, (kind, (tc, tn, sc, sn)) in enumerate(zip(kinds, arrays)):
        t_kind[0, i], t_cat[0, i], t_num[0, i], s_cat[0, i] = kind, tc, tn, sc
        s_num[0, i] = sn.astype(np.float16)               # the precision the network was trained on
        t_ext[0, i] = ext[i]
    t_rel = np.zeros((1, T, T), np.int64)
    for i, j, bit in edges:
        t_rel[0, i, j] |= 1 << bit
    n_hand = len(o["hand"])
    a = [int(x[1]) if x[0] == "play" else n_hand + int(x[1]) if x[0] == "drink" else -1 for x in o["legal"]]
    a_tok = np.full((1, max(1, len(a))), -2, np.int64)
    a_tok[0, :len(a)] = a
    g = list(global_features(o)) + [len(o.get("timers") or ()) / 4.0, len(costs) / 2.0, len(pcount) / 2.0,
                                    0.0 if soonest is None else 1.0 / (1.0 + soonest), any_watched]
    return {"g_num": np.asarray(g[:G_NUM], np.float32)[None], "g_cat": np.asarray([g_cat], np.int64),
            "st_kind": st_kind, "st_num": st_num, "t_kind": t_kind, "t_cat": t_cat, "t_num": t_num, "s_cat": s_cat,
            "s_num": s_num, "t_ext": t_ext, "a_tok": a_tok, "t_rel": t_rel}
