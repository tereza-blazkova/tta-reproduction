#!/usr/bin/env python
"""Stage 35 — do students differ in how often they are right, and do different students get
different codes?

The within-student designs (stages 15, 18) find that 31 pooled code associations fall to three
once a student is compared with themselves, so the pooled associations are carried by
differences between students. This stage shows those differences directly, in two steps.

**A. Students differ.** A student's correctness is the share of their post-session answers that
are correct (`--outcome correct`, the default). Its spread across the students with two or more
sessions is compared with the spread after their sessions are dealt to students at random, each
student keeping their number of sessions. Shuffling whole sessions keeps a session's responses
together, so a good day is not counted as a good student; the excess variance is the variance of
a stable student trait. A split-half correlation (correctness on half a student's sessions
against the other half) states the same thing in a second form.

The alternative outcome (`--outcome residual`, and the `other_outcome` arm of every run) scores
each response against its question: the outcome minus the question's correctness rate, computed
without the response's own session and shrunk toward its objective's rate (and an objective's
toward the corpus rate) when the question is rare. It asks whether a student is strong or only
answered easy questions.

**B. Different students, different codes.** Each student's sessions are split at random into two
halves: attainment is measured on one half and the code distribution on the other, so no session
contributes both to the outcome that ranks its student and to the codes compared. The top and
bottom thirds of students by attainment are compared by the Jensen-Shannon divergence of their mean
code distributions, for the 41 tutor moves and the 16 student signals separately, and per code by
the ratio of mean shares. Twenty random splits, each used in both directions, are averaged.

Two permutation nulls, with the same permutation applied to every split:
  * students: attainment is permuted across students, keeping each student's sessions and codes
    intact. Finite groups give a positive divergence even with no difference, and this null
    measures that floor;
  * within tutor: attainment is permuted only among students who share a main tutor, so a code
    difference that comes from which tutor a strong student drew is part of the null.

Shares here are raw (unshrunk) shares of a role's labelled turns from the 122B's labels. The stage
first checks that shrinking its counts reproduces the model's feature table.

Two scripts carry the number 35: this one and `35_between_within.py` (`results/between-within`).
Where a later stage or `ttapub.attainment` says "stage 35", it means this one.

    python scripts/35_student_heterogeneity.py --run student-heterogeneity
"""
from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from scipy import sparse

from ttapub import agg, config, data, runlog, within
from ttapub import taxonomy_v3 as T

#: Pseudo-count, in responses, that shrinks a rare question's rate toward its objective's and an
#: objective's toward the corpus. Frozen; used only by the question-adjusted outcome.
M_SHRINK = 10.0
ROLES = ("tutor", "student")
#: session column summed for each outcome; the other one is the robustness arm
OUTCOME = {"correct": "y_sum", "residual": "res_sum"}


def expected_correct(tr: pd.DataFrame, m: float = M_SHRINK) -> np.ndarray:
    """Each response's question correctness rate, leaving out the response's own session."""
    y = tr.is_correct.astype(float)
    g = y.mean()

    def loso(key: str) -> tuple[np.ndarray, np.ndarray]:
        tot = y.groupby(tr[key])
        own = y.groupby([tr[key], tr.session_id])
        s = tot.transform("sum") - own.transform("sum")
        n = tot.transform("size") - own.transform("size")
        return s.to_numpy(), n.to_numpy()

    s_lo, n_lo = loso("learning_objective_id")
    p_lo = (s_lo + m * g) / (n_lo + m)
    s_q, n_q = loso("question_id")
    return (s_q + m * p_lo) / (n_q + m)


def label_counts(role: str, sessions: pd.Index) -> pd.DataFrame:
    files = sorted(config.LABELS_V3.glob(f"v3_full_train_{role}_shard*.csv"))
    lab = pd.concat([pd.read_csv(f, dtype={"session_id": str}, usecols=["session_id", "turn", "code"])
                     for f in files], ignore_index=True).drop_duplicates(["session_id", "turn"])
    cnt = pd.crosstab(lab.session_id, lab.code)
    bad = sorted(set(cnt.columns) - set(T.codes(role)))
    if bad:
        raise SystemExit(f"{role} labels outside the pool: {bad[:5]}")
    return cnt.reindex(index=sessions, columns=T.codes(role), fill_value=0)


def check_against_features(counts: dict[str, pd.DataFrame]) -> dict:
    """Shrinking these counts must give the model's own columns, or they are not the same data."""
    feat = pd.read_csv(config.FEATURES_TRAIN, dtype={"session_id": str}).set_index("session_id")
    prior = agg.load_prior()
    out = {}
    for role, cnt in counts.items():
        c = cnt.reindex(feat.index, fill_value=0)
        n = c.sum(axis=1).to_numpy()
        pri = np.array([prior[k] for k in c.columns])
        mine = (c.to_numpy() + agg.SHRINK_KAPPA * pri) / (n[:, None] + agg.SHRINK_KAPPA)
        theirs = feat[[agg.column(k, role) for k in c.columns]].to_numpy()
        # the feature table leaves a session with no turns of this role empty; it must be
        # exactly the sessions with no labels here, and every other session must match
        empty = np.isnan(theirs).any(axis=1)
        gap = np.abs(mine - theirs)[~empty].max(axis=1)
        out[role] = {"max_abs_gap": float(gap.max()), "sessions_compared": int((~empty).sum()),
                     "sessions_empty_in_both": int(empty.sum())}
        print(f"  {role}: shrunk label counts vs feature table, max gap {gap.max():.2e} over "
              f"{(~empty).sum():,} sessions; {empty.sum()} sessions without turns", flush=True)
        if gap.max() > 1e-6 or (n[empty] > 0).any() or not np.isfinite(gap).all():
            raise SystemExit(f"{role} label counts do not reproduce {config.FEATURES_TRAIN.name}")
    return out


def jsd_terms(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Per-code Jensen-Shannon terms in bits; each is >= 0 and they sum to the divergence."""
    P = P / P.sum(-1, keepdims=True)
    Q = Q / Q.sum(-1, keepdims=True)
    M = (P + Q) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(P > 0, P * np.log2(P / M), 0.0)
        b = np.where(Q > 0, Q * np.log2(Q / M), 0.0)
    return 0.5 * a + 0.5 * b


class Splits:
    """Random halves of each student's sessions, and what each half measures."""

    def __init__(self, sess: pd.DataFrame, shares: dict[str, np.ndarray], n_st: int,
                 reps: int, seed: int, col: str, alt: str):
        self.n_st, self.col, self.alt = n_st, col, alt
        rng = np.random.default_rng(seed)
        st = sess.st.to_numpy()
        size = np.bincount(st, minlength=n_st)
        first = np.r_[0, np.cumsum(size)[:-1]]
        self.reps = []
        for _ in range(reps):
            # sessions ordered by student, at random within student; the first floor(n/2) of
            # each student's sessions form one half
            o = np.lexsort((rng.random(len(st)), st))
            rank = np.empty(len(st), int)
            rank[o] = np.arange(len(st)) - first[st[o]]
            half = rank < size[st] // 2
            for h in (True, False):
                self.reps.append(self._one(sess, shares, half == h, half != h))

    def _one(self, sess, shares, a_mask, b_mask) -> dict:
        st = sess.st.to_numpy()

        def mean(col, m):
            return (np.bincount(st[m], sess[col].to_numpy()[m], self.n_st)
                    / np.bincount(st[m], sess.n_resp.to_numpy()[m], self.n_st))

        out = {"attainment": mean(self.col, a_mask), "attainment_alt": mean(self.alt, a_mask),
               "other": mean(self.col, b_mask), "other_expected": mean("exp_sum", b_mask)}
        for role, S in shares.items():
            ok = b_mask & ~np.isnan(S[:, 0])
            W = sparse.csr_matrix((np.ones(ok.sum()), (st[ok], np.flatnonzero(ok))),
                                  shape=(self.n_st, len(st)))
            k = np.asarray(W.sum(1)).ravel()
            C = np.asarray(W @ np.nan_to_num(S)) / np.where(k > 0, k, 1)[:, None]
            out[role] = (C, (k > 0).astype(float))
        return out


def thirds(x: np.ndarray, keep: np.ndarray, rng) -> np.ndarray:
    """+1 top third, -1 bottom third, 0 middle or excluded; ties broken at random."""
    g = np.zeros(len(x), int)
    idx = np.flatnonzero(keep)
    r = np.lexsort((rng.random(len(idx)), x[idx]))
    k = len(idx) // 3
    g[idx[r[:k]]] = -1
    g[idx[r[-k:]]] = 1
    return g


def contrast(groups: list[np.ndarray], reps: list[dict], role: str, W: np.ndarray) -> tuple:
    """Average over splits of (JSD, per-code log2 ratio, per-code terms, shares) under weights.

    `W` is (D, n_students): a row of student weights per draw (permuted group labels are passed
    in as `groups`, a list over splits of (D, n_students) label matrices; bootstrap draws pass
    multiplicities in W with fixed labels)."""
    jsd = lr = terms = top = bot = 0.0
    for g, rep in zip(groups, reps):
        C, v = rep[role]
        wt, wb = W * (g == 1), W * (g == -1)
        Pt = (wt @ C) / (wt @ v)[:, None]
        Pb = (wb @ C) / (wb @ v)[:, None]
        t = jsd_terms(Pt, Pb)
        jsd = jsd + t.sum(1)
        terms = terms + t
        lr = lr + np.log2(Pt / Pb)
        top, bot = top + Pt, bot + Pb
    n = len(reps)
    return jsd / n, lr / n, terms / n, top / n, bot / n


def stratified_perm(strata: np.ndarray, rng) -> np.ndarray:
    base = np.argsort(strata, kind="stable")
    shuf = np.lexsort((rng.random(len(strata)), strata))
    p = np.empty(len(strata), int)
    p[base] = shuf
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="student-heterogeneity")
    ap.add_argument("--splits", type=int, default=20)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outcome", choices=list(OUTCOME), default="correct",
                    help="correct = share of answers correct; residual = minus the question's rate")
    a = ap.parse_args()
    shards = sorted(config.LABELS_V3.glob("v3_full_train_*_shard*.csv"))
    d = runlog.start(a.run, vars(a), [config.TRAIN_LABELS, config.METADATA, config.FEATURES_TRAIN,
                                       config.CODE_PRIOR, *shards])
    rng = np.random.default_rng(a.seed)

    meta = data.load_metadata()
    tr = (data.load_train()
          .merge(meta[["response_id", "student_id", "tutor_id", "question_id"]], on="response_id")
          .reset_index(drop=True))
    tr["expected"] = expected_correct(tr)
    tr["res"] = tr.is_correct - tr.expected

    counts = {r: label_counts(r, pd.Index(tr.session_id.unique())) for r in ROLES}
    gate = check_against_features(counts)

    sess = tr.groupby("session_id").agg(student_id=("student_id", "first"),
                                        tutor_id=("tutor_id", "first"),
                                        res_sum=("res", "sum"), y_sum=("is_correct", "sum"),
                                        exp_sum=("expected", "sum"), n_resp=("res", "size"))
    n_sess = sess.groupby("student_id").size()
    multi = n_sess.index[n_sess >= 2]
    sess = sess[sess.student_id.isin(multi)].reset_index()
    students = pd.Index(sorted(multi))
    sess["st"] = students.get_indexer(sess.student_id)
    n_st = len(students)
    shares = {}
    for r in ROLES:
        c = counts[r].reindex(sess.session_id).to_numpy(float)
        n = c.sum(1, keepdims=True)
        shares[r] = np.where(n > 0, c / np.where(n > 0, n, 1), np.nan)
    print(f"{n_st:,} students with two or more sessions, {len(sess):,} sessions, "
          f"{int(sess.n_resp.sum()):,} responses", flush=True)

    # ── A. spread of student means against sessions dealt to students at random ──
    st = sess.st.to_numpy()
    n_r = np.bincount(st, sess.n_resp.to_numpy(), n_st)

    def spread(col: str, B: int) -> dict:
        v = sess[col].to_numpy()
        m_obs = np.bincount(st, v, n_st) / n_r
        null_var, null_hist = [], []
        bins = np.arange(0.0, 1.0001, 0.1) if col == "y_sum" else np.arange(-1.0, 1.0001, 0.05)
        for _ in range(B):
            p = rng.permutation(st)
            m = np.bincount(p, v, n_st) / np.bincount(p, sess.n_resp.to_numpy(), n_st)
            null_var.append(m.var(ddof=1))
            null_hist.append(np.histogram(m, bins)[0])
        null_var = np.array(null_var)
        tau2 = m_obs.var(ddof=1) - null_var.mean()
        hist = pd.DataFrame({"bin_lo": bins[:-1], "bin_hi": bins[1:],
                             "observed": np.histogram(m_obs, bins)[0]})
        q = np.percentile(np.array(null_hist), [2.5, 50, 97.5], axis=0)
        hist["null_lo"], hist["null_mid"], hist["null_hi"] = q
        return {"sd_observed_pp": 100 * float(m_obs.std(ddof=1)),
                "sd_null_pp": 100 * float(np.sqrt(null_var.mean())),
                "variance_ratio": float(m_obs.var(ddof=1) / null_var.mean()),
                "sd_stable_trait_pp": 100 * float(np.sqrt(max(tau2, 0.0))),
                "p_perm": float((1 + (null_var >= m_obs.var(ddof=1)).sum()) / (B + 1)),
                "mean_pp": 100 * float(m_obs.mean())}, hist

    col = OUTCOME[a.outcome]
    alt = next(v for v in OUTCOME.values() if v != col)
    A, hist = spread(col, a.perm)
    A_alt, hist_alt = spread(alt, a.perm)
    print(f"A  {a.outcome}: mean {A['mean_pp']:.1f}, SD {A['sd_observed_pp']:.1f} pp vs "
          f"{A['sd_null_pp']:.1f} by chance, stable trait {A['sd_stable_trait_pp']:.1f} pp "
          f"(p {A['p_perm']:.4f}); other outcome: SD {A_alt['sd_observed_pp']:.1f} vs "
          f"{A_alt['sd_null_pp']:.1f}, trait {A_alt['sd_stable_trait_pp']:.1f} pp", flush=True)

    # ── splits, shared by the split-half check and by B ──
    reps = Splits(sess, shares, n_st, a.splits, a.seed + 1, col, alt).reps
    tutor_main = (sess.groupby("st").tutor_id.agg(lambda s: s.value_counts().index[0])
                  .reindex(range(n_st)).fillna("none").to_numpy())
    strata = pd.factorize(tutor_main)[0]
    n_sess_st = np.bincount(st, minlength=n_st)

    # split-half: the outcome on one half against the other, binned on the first half. A share of
    # correct answers from a few responses is coarse (over a third of students are right on every
    # answer of a half), so it is binned by value with 100% on its own; the residual by fifths.
    r_half = np.mean([np.corrcoef(rep["attainment"], rep["other"])[0, 1] for rep in reps])
    perms = [rng.permutation(n_st) for _ in range(a.perm)]
    r_null = np.array([np.mean([np.corrcoef(rep["attainment"][p], rep["other"])[0, 1]
                                for rep in reps[:4]]) for p in perms[:500]])
    edges = [0.0, 0.25, 0.5, 0.75, 1.0]
    labels = ["0-25%", "25-50%", "50-75%", "75-99%", "100%"]
    bins = []
    boot_w = rng.multinomial(n_st, np.full(n_st, 1 / n_st), size=a.boot).astype(float)
    for rep in reps:
        if col == "y_sum":
            q = np.where(rep["attainment"] >= 1.0, 4, np.digitize(rep["attainment"], edges[1:-1]))
        else:
            q = pd.qcut(pd.Series(rep["attainment"]).rank(method="first"), 5, labels=False).to_numpy()
        Q = np.stack([(q == k) for k in range(5)]).astype(float)       # (5, n_st)
        mean_b = (Q @ rep["other"]) / Q.sum(1)
        mean_a = (Q @ rep["attainment"]) / Q.sum(1)
        boot_b = (boot_w @ (Q * rep["other"]).T) / (boot_w @ Q.T)        # (boot, 5)
        bins.append((mean_a, mean_b, boot_b, Q.sum(1)))
    fifth = pd.DataFrame({"bin": labels if col == "y_sum" else [f"fifth {k}" for k in range(1, 6)],
                          "students": np.mean([f[3] for f in bins], 0),
                          "attainment_half_pp": 100 * np.mean([f[0] for f in bins], 0),
                          "other_half_pp": 100 * np.mean([f[1] for f in bins], 0)})
    bmean = np.mean([f[2] for f in bins], 0)
    fifth["lo_pp"], fifth["hi_pp"] = 100 * np.percentile(bmean, [2.5, 97.5], 0)
    split_half = {"r": float(r_half), "spearman_brown": float(2 * r_half / (1 + r_half)),
                  "null_r_95": [float(x) for x in np.percentile(r_null, [2.5, 97.5])]}
    print(f"A  split-half r {r_half:.3f} (Spearman-Brown {split_half['spearman_brown']:.3f}); "
          f"null 95% {split_half['null_r_95']}", flush=True)

    # ── B. code distributions of the top against the bottom third ──
    def run_b(attainment_key: str, keep: np.ndarray, arm: str, null_strata=None) -> tuple:
        g_obs = [thirds(rep[attainment_key], keep, rng) for rep in reps]
        res = {}
        for role in ROLES:
            one = np.ones((1, n_st))
            jsd, lr, terms, top, bot = contrast(g_obs, reps, role, one)
            res[role] = {"jsd": jsd[0], "lr": lr[0], "terms": terms[0], "top": top[0],
                         "bot": bot[0], "null_jsd": [], "null_lr": [], "boot_jsd": [],
                         "boot_lr": []}
        for lo in range(0, a.perm, 250):
            ps = [stratified_perm(null_strata, rng) if null_strata is not None else perms[i]
                  for i in range(lo, min(lo + 250, a.perm))]
            G = [np.stack([g[p] for p in ps]) for g in g_obs]
            for role in ROLES:
                jsd, lr, *_ = contrast(G, reps, role, np.ones((len(ps), n_st)))
                res[role]["null_jsd"].append(jsd)
                res[role]["null_lr"].append(lr)
        for lo in range(0, a.boot, 250):
            Wb = boot_w[lo:lo + 250]
            G = [np.broadcast_to(g, Wb.shape) for g in g_obs]
            for role in ROLES:
                jsd, lr, *_ = contrast(G, reps, role, Wb)
                res[role]["boot_jsd"].append(jsd)
                res[role]["boot_lr"].append(lr)
        for role in ROLES:
            r = res[role]
            for k in ("null_jsd", "null_lr", "boot_jsd", "boot_lr"):
                r[k] = np.concatenate(r[k])
            print(f"B  {arm:14s} {role:7s} JSD {1e3 * r['jsd']:.3f} mbit vs null "
                  f"{1e3 * r['null_jsd'].mean():.3f} (95th {1e3 * np.percentile(r['null_jsd'], 95):.3f}), "
                  f"p {(1 + (r['null_jsd'] >= r['jsd']).sum()) / (len(r['null_jsd']) + 1):.4f}",
                  flush=True)
        return res

    everyone = np.ones(n_st, bool)
    four = n_sess_st >= 4
    # the 4+ arm permutes only among the students it keeps, so the null groups are the same people
    arms = {"students": run_b("attainment", everyone, "students"),
            "within_tutor": run_b("attainment", everyone, "within tutor", strata),
            "other_outcome": run_b("attainment_alt", everyone, "other outcome"),
            "four_sessions": run_b("attainment", four, "4+ sessions", four.astype(int))}

    jrows, null_rows = [], []
    for arm, res in arms.items():
        for role in ROLES:
            r = res[role]
            nj = r["null_jsd"]
            jrows.append({"arm": arm, "pool": role, "jsd_mbit": 1e3 * r["jsd"],
                          "ci_lo_mbit": 1e3 * np.percentile(r["boot_jsd"], 2.5),
                          "ci_hi_mbit": 1e3 * np.percentile(r["boot_jsd"], 97.5),
                          "null_mean_mbit": 1e3 * nj.mean(),
                          "null_p95_mbit": 1e3 * np.percentile(nj, 95),
                          "excess_mbit": 1e3 * (r["jsd"] - nj.mean()),
                          "ratio_to_null": r["jsd"] / nj.mean(),
                          "p_perm": (1 + (nj >= r["jsd"]).sum()) / (len(nj) + 1)})
            if arm in ("students", "within_tutor"):
                null_rows.append(pd.DataFrame({"arm": arm, "pool": role, "jsd_mbit": 1e3 * nj}))
    jsd_tab = pd.DataFrame(jrows)
    pd.concat(null_rows).to_csv(d / "jsd_null.csv", index=False)
    jsd_tab.to_csv(d / "jsd.csv", index=False)

    rows = []
    for role in ROLES:
        r, rt, r4 = arms["students"][role], arms["within_tutor"][role], arms["four_sessions"][role]
        prev = np.nanmean(shares[role], 0)
        for k, code in enumerate(T.codes(role)):
            nl, tl, n4 = r["null_lr"][:, k], rt["null_lr"][:, k], r4["null_lr"][:, k]
            rows.append({"code": code, "role": role, "share_pct": 100 * prev[k],
                         "top_pct": 100 * r["top"][k], "bottom_pct": 100 * r["bot"][k],
                         "diff_pp": 100 * (r["top"][k] - r["bot"][k]),
                         "log2_ratio": r["lr"][k],
                         "ci_lo": np.percentile(r["boot_lr"][:, k], 2.5),
                         "ci_hi": np.percentile(r["boot_lr"][:, k], 97.5),
                         "p_perm": (1 + (np.abs(nl) >= abs(r["lr"][k])).sum()) / (len(nl) + 1),
                         "p_perm_within_tutor": (1 + (np.abs(tl - tl.mean()) >= abs(r["lr"][k] - tl.mean())).sum())
                         / (len(tl) + 1),
                         "jsd_term_mbit": 1e3 * r["terms"][k],
                         "jsd_share": r["terms"][k] / r["jsd"],
                         # attainment measured on two or more sessions: less noise in the ranking
                         "log2_ratio_4plus": r4["lr"][k],
                         "ci_lo_4plus": np.percentile(r4["boot_lr"][:, k], 2.5),
                         "ci_hi_4plus": np.percentile(r4["boot_lr"][:, k], 97.5),
                         "p_perm_4plus": (1 + (np.abs(n4) >= abs(r4["lr"][k])).sum()) / (len(n4) + 1)})
    per_code = pd.DataFrame(rows)
    per_code["bh_sig"] = within.benjamini_hochberg(per_code.p_perm.to_numpy(), 0.05)
    per_code["bh_sig_within_tutor"] = within.benjamini_hochberg(
        per_code.p_perm_within_tutor.to_numpy(), 0.05)
    per_code.to_csv(d / "per_code.csv", index=False)
    hist.to_csv(d / "spread_hist.csv", index=False)
    hist_alt.to_csv(d / "spread_hist_other_outcome.csv", index=False)
    fifth.to_csv(d / "split_half_bins.csv", index=False)

    # what else differs between the two groups, so the code contrast can be read against it
    # (all sessions of the students in each third of the first split)
    g0 = thirds(reps[0]["attainment"], everyone, np.random.default_rng(a.seed + 2))
    turns = {r: counts[r].reindex(sess.session_id).sum(1).to_numpy() for r in ROLES}
    desc = {}
    for lab, gv in (("top", 1), ("bottom", -1)):
        m = np.isin(st, np.flatnonzero(g0 == gv))
        desc[lab] = {"students": int((g0 == gv).sum()),
                     "sessions_per_student": float(n_sess_st[g0 == gv].mean()),
                     "tutor_turns_per_session": float(turns["tutor"][m].mean()),
                     "student_turns_per_session": float(turns["student"][m].mean()),
                     "expected_correct_pct": 100 * float(sess.exp_sum[m].sum() / sess.n_resp[m].sum()),
                     "observed_correct_pct": 100 * float(sess.y_sum[m].sum() / sess.n_resp[m].sum())}
    # do strong students get easier questions in their other half? (would confound B with topic)
    r_diff = np.mean([np.corrcoef(rep["attainment"], rep["other_expected"])[0, 1] for rep in reps])

    s = per_code
    runlog.finish(d, {
        "run": a.run, "students_multi_session": int(n_st), "sessions": int(len(sess)),
        "responses": int(sess.n_resp.sum()), "shrink_pseudocount": M_SHRINK,
        "label_check": gate,
        "outcome": a.outcome, "spread": A, "spread_other_outcome": A_alt,
        "split_half": split_half,
        "jsd": jsd_tab.to_dict("records"),
        "codes_bh_sig": {r: int(s[(s.role == r)].bh_sig.sum()) for r in ROLES},
        "codes_bh_sig_within_tutor": {r: int(s[(s.role == r)].bh_sig_within_tutor.sum())
                                      for r in ROLES},
        "groups_split0": desc,
        "r_attainment_vs_question_ease": float(r_diff),
        "main_tutor_strata": int(strata.max() + 1),
    })


if __name__ == "__main__":
    main()
