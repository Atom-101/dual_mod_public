"""On-the-fly generators for the state-tracking ladder (plans/
state_tracking_ladder.md). No corpus, no tokenizer, no LLM: each task emits
(tokens[B,T], mask[B,T]) batches forever from ~30-60 lines of numpy.
Capability shown on these tasks is algorithmic by construction — fresh
batch every step, memorization impossible.

Tasks
  group   running product in Z10 / S4 / S5 (S5 word problem is NC1-complete
          via Barrington — the TC0-breaking headline; Z10 is the solvable
          control). [BOS, g1, s1, g2, s2, ...], loss at s-positions.
  khop    pointer chasing on a planted Hamiltonian cycle; [key ARROW succ]
          pairs shuffled, then [QUERY start k answer] blocks.
  parity  running XOR (empirical-difficulty control; formally in TC0).
  match3  any triple in trailing window W summing to 0 mod M (TC0 control;
          M,W tuned so base rate ~= 0.5 — report it).

Conventions (pinned; do not mix across arms)
  vocab: 0 PAD, 1 BOS, 2 ARROW, 3 QUERY, 4-7 reserved; task tokens from 8.
  Targets always live in a DISJOINT token range from inputs (states vs
  group elements, answers reuse symbol range but sit after QUERY blocks).
  Loss mask multiplies the LOSS (harness indexes masked positions); the
  model attends to everything.
  Group convention: perms are tuples p with p[i] = p(i); composition
  (a@b)(x) = a(b(x)) i.e. APPLY b FIRST; MUL[a,b] = index(a@b); the
  running state after op g_i is s_i = g_i @ s_{i-1} = MUL[g_i, s_{i-1}].
  Verified by unit test below (associativity, identity, known product,
  and naive-application cross-check) — run `python formal_language/harness/statetrack_gen.py`.

Determinism: rng per (task, seed, split, step) via np.random.default_rng
SeedSequence lists — byte-identical stream per seed across arms.
"""
import itertools

import numpy as np
import sys

PAD, BOS, ARROW, QUERY = 0, 1, 2, 3
XFROM = 4                                 # cross-register op marker (keyed task)
BASE = 8                                  # first task token


# --------------------------------------------------------------- groups ----
def _parity(p):
    inv = sum(1 for i in range(len(p)) for j in range(i + 1, len(p))
              if p[i] > p[j])
    return inv % 2


def cayley(group):
    """(MUL[n,n] int16, n). Elements indexed 0..n-1; identity is index 0.
    a5 = even permutations of 5 (smallest NON-SOLVABLE group, |A5|=60) —
    the headline complexity task; s3/s4 are solvable (below the NC1 bar,
    debugging rungs only)."""
    if group.startswith("z"):
        m = int(group[1:])
        a = np.arange(m)
        return ((a[:, None] + a[None, :]) % m).astype(np.int16), m
    if group == "a4xz5":
        # Illusion-of-State (Merrill et al. 2024) TC0 control: A4 x Z5 — SOLVABLE but
        # non-abelian, |G| = 12*5 = 60 = |A5| = |Z60| (state-count-matched trio).
        a4 = sorted(p for p in itertools.permutations(range(4)) if _parity(p) == 0)
        idx4 = {p: i for i, p in enumerate(a4)}
        elems = [(i, z) for i in range(len(a4)) for z in range(5)]      # (a4 index, z5)
        n = len(elems)
        mul = np.zeros((n, n), dtype=np.int16)
        for i, (pa, za) in enumerate(elems):
            for j, (pb, zb) in enumerate(elems):
                pc = idx4[tuple(a4[pa][a4[pb][x]] for x in range(4))]  # a(b(x))
                mul[i, j] = pc * 5 + (za + zb) % 5
        return mul, n
    if group == "a5":
        perms = sorted(p for p in itertools.permutations(range(5))
                       if _parity(p) == 0)
    else:
        k = {"s3": 3, "s4": 4, "s5": 5}[group]
        perms = sorted(itertools.permutations(range(k)))
    idx = {p: i for i, p in enumerate(perms)}
    n = len(perms)
    kk = len(perms[0])
    mul = np.zeros((n, n), dtype=np.int16)
    for i, a in enumerate(perms):
        for j, b in enumerate(perms):
            mul[i, j] = idx[tuple(a[b[x]] for x in range(kk))]   # a(b(x))
    return mul, n


def make_group_batch(rng, B, n_ops, mul, n):
    """[BOS, g1, s1, ..., gN, sN]; states offset by n (disjoint range)."""
    g = rng.integers(0, n, size=(B, n_ops))
    s = np.zeros((B, n_ops), dtype=np.int64)
    prev = np.zeros(B, dtype=np.int64)                 # identity = 0
    for t in range(n_ops):
        prev = mul[g[:, t], prev]
        s[:, t] = prev
    T = 1 + 2 * n_ops
    tok = np.zeros((B, T), dtype=np.int64)
    msk = np.zeros((B, T), dtype=bool)
    tok[:, 0] = BOS
    tok[:, 1::2] = BASE + g
    tok[:, 2::2] = BASE + n + s
    msk[:, 2::2] = True
    return tok, msk


def make_group_tagged_batch(rng, B, n_ops, mul, n, op_pool=None):
    """v3 TAGGING format (Merrill-Petty-Sabharwal; the paper-grade task):
    input stream is ops ONLY [BOS, g1..gN]; the loss at the position of
    g_i targets the running product s_i. States appear as TARGETS ONLY —
    never in the input — so supervision is dense (every position) with
    zero leak: scoring above chance at position 30 requires maintaining
    state internally. Returns (tok[B,T], tgt[B,T], msk[B,T]): tgt holds
    state-token ids (BASE+n offset, same output vocab as before); loss is
    CE(logits[pos], tgt[pos]) AT the position (not next-token)."""
    if op_pool is None:
        g = rng.integers(0, n, size=(B, n_ops))
    else:
        g = np.asarray(op_pool)[rng.integers(0, len(op_pool),
                                             size=(B, n_ops))]
    s = np.zeros((B, n_ops), dtype=np.int64)
    prev = np.zeros(B, dtype=np.int64)
    for t in range(n_ops):
        prev = mul[g[:, t], prev]
        s[:, t] = prev
    T = 1 + n_ops
    tok = np.zeros((B, T), dtype=np.int64)
    tgt = np.zeros((B, T), dtype=np.int64)
    msk = np.zeros((B, T), dtype=bool)
    tok[:, 0] = BOS
    tok[:, 1:] = BASE + g
    tgt[:, 1:] = BASE + n + s
    msk[:, 1:] = True
    return tok, tgt, msk


def make_group_final_batch(rng, B, T, mul, n, k_max, k_min=1):
    """THEOREM-GRADE format (v2): ops-only -> final state, no intermediate
    states anywhere in context. The interleaved format above leaks every
    prefix state, reducing each target to one lookup + one Cayley product —
    depth-1, solvable by attention (observed: vanilla 0.97-0.999 at 4x on
    S4/S5). NC1-hardness only applies when the composition happens inside
    the model: [g1..gk, QUERY, s_k].

    Packs independent episodes [g x k, QUERY, s_k, SEP-implicit] into rows
    of length T; k ~ uniform(1, k_max) (short episodes teach the table,
    long ones force composition). Loss at the s_k position of each episode.
    Returns (tok, msk, diff) with diff = k per masked position (sidecar)."""
    tok = np.zeros((B, T), dtype=np.int64)
    msk = np.zeros((B, T), dtype=bool)
    diff = np.zeros((B, T), dtype=np.int64)
    tok[:, 0] = BOS
    for b in range(B):
        p = 1
        while p + 3 <= T:                       # room for g, QUERY, s
            k = int(rng.integers(k_min, k_max + 1))
            if p + k + 2 > T:
                k = T - p - 2
                if k < 1:
                    break
            g = rng.integers(0, n, size=k)
            s = 0
            for gi in g:
                s = mul[gi, s]
            tok[b, p:p + k] = BASE + g
            tok[b, p + k] = QUERY
            tok[b, p + k + 1] = BASE + n + s
            msk[b, p + k + 1] = True
            diff[b, p + k + 1] = k
            p += k + 2
    return tok, msk, diff


def make_keyed_group_batch(rng, B, mul, n, K, D, rho=0.0, K_max=None):
    """Keyed group accumulation — the (G,K,D,rho) capstone rung (plans/
    keyed_group_accumulation_spec.md). K registers, each a running product in
    group (mul,n); D ops per register, interleaved; a rho-fraction are cross-
    register XOPs s_k <- s_k*s_j; then every register is queried once. Dense
    leak-free TAGGING: at each op's LAST token the target is the updated s_k;
    at each QUERY the target is the final s_k (states are targets only, never
    input). Capacity axis = K (Theta(K) independent state); circuit axis = D.

    Vocab (K_max fixes the layout so it is STABLE across a K-sweep):
      KEY   = BASE           .. BASE+K_max-1
      ELEM  = BASE+K_max     .. +n-1
      STATE = BASE+K_max+n   .. +n-1        (targets; disjoint output range)
    Tokens/op: normal 2 [KEY_k, ELEM_g]; XOP 3 [KEY_k, XFROM, KEY_j]; query 2
    [QUERY, KEY_k]. Returns (tok, tgt, msk, depth); depth = per-register op-count
    at that position (by-depth sidecar; query positions carry depth=D). Rows that
    fall back cross->normal early (need >=2 live regs) pad to T with PAD (masked)."""
    if K_max is None:
        K_max = K
    KEYB, ELEMB, STATEB = BASE, BASE + K_max, BASE + K_max + n
    n_ops = K * D
    n_cross = int(round(rho * n_ops))
    # ONE schedule shared across the batch (register order, cross ops, cross
    # sources) — rows differ only in the group ELEMENTS g, so the whole state
    # update vectorizes across B (the Python loop is O(n_ops), not O(B*n_ops)).
    # Fresh schedule every call ⇒ full structural variety over training.
    sched = np.repeat(np.arange(K), D)
    rng.shuffle(sched)
    is_cross = np.zeros(n_ops, dtype=bool)
    jsrc = np.full(n_ops, -1, dtype=np.int64)
    live = set()
    eligible = []
    for t in range(n_ops):                       # cheap O(n_ops) planning pass
        if len(live - {int(sched[t])}) >= 1:
            eligible.append(t)
        live.add(int(sched[t]))
    if n_cross > 0 and eligible:
        cidx = rng.choice(eligible, size=min(n_cross, len(eligible)),
                          replace=False)
        is_cross[np.atleast_1d(cidx)] = True
    live = set()
    for t in range(n_ops):
        if is_cross[t]:
            jsrc[t] = int(rng.choice(sorted(live - {int(sched[t])})))
        live.add(int(sched[t]))
    # token positions from the (shared) op-type sequence
    pos = np.zeros(n_ops, dtype=np.int64)
    last = np.zeros(n_ops, dtype=np.int64)
    p = 1
    for t in range(n_ops):
        pos[t] = p
        last[t] = p + (2 if is_cross[t] else 1)
        p += 3 if is_cross[t] else 2
    qstart = p
    T = p + 2 * K
    tok = np.zeros((B, T), dtype=np.int64)
    tgt = np.zeros((B, T), dtype=np.int64)
    msk = np.zeros((B, T), dtype=bool)
    depth = np.zeros((B, T), dtype=np.int64)
    tok[:, 0] = BOS
    g_vals = rng.integers(0, n, size=(B, n_ops))
    s = np.zeros((B, K), dtype=np.int64)          # all identity
    seen = np.zeros(K, dtype=np.int64)
    for t in range(n_ops):
        k, p0, lp = int(sched[t]), int(pos[t]), int(last[t])
        if is_cross[t]:
            j = int(jsrc[t])
            s[:, k] = mul[s[:, k], s[:, j]]       # s_k <- s_k * s_j  (vec over B)
            tok[:, p0], tok[:, p0 + 1], tok[:, p0 + 2] = KEYB + k, XFROM, KEYB + j
        else:
            g = g_vals[:, t]
            s[:, k] = mul[g, s[:, k]]             # s_k <- g * s_k     (vec over B)
            tok[:, p0], tok[:, p0 + 1] = KEYB + k, ELEMB + g
        tgt[:, lp] = STATEB + s[:, k]
        msk[:, lp] = True
        seen[k] += 1
        depth[:, lp] = int(seen[k])
    for i, k in enumerate(rng.permutation(K)):
        p0 = qstart + 2 * i
        tok[:, p0], tok[:, p0 + 1] = QUERY, KEYB + int(k)
        tgt[:, p0 + 1] = STATEB + s[:, int(k)]
        msk[:, p0 + 1] = True
        depth[:, p0 + 1] = D
    return tok, tgt, msk, depth


def make_mixed_tagged_batch(rng, B, n_ops, mul_a, n_a, mul_b, n_b,
                            frac_a=0.5, force=None):
    """Mixture training (the decisive formation-window cell): each row is
    EITHER group A (z10) or group B (a5), tagged format, disjoint token
    ranges. Tractable mass (A) funds the carry; the B column reads whether
    hard composition can then use it. force='a'/'b' builds pure eval sets.
    Layout: A ops BASE.., A states BASE+n_a.., B ops BASE+2n_a..,
    B states BASE+2n_a+n_b.. Returns (tok, tgt, msk, task[B])."""
    T = 1 + n_ops
    tok = np.zeros((B, T), dtype=np.int64)
    tgt = np.zeros((B, T), dtype=np.int64)
    msk = np.zeros((B, T), dtype=bool)
    task = np.zeros(B, dtype=np.int64)          # 0=A, 1=B
    tok[:, 0] = BOS
    offB = BASE + 2 * n_a
    for b in range(B):
        is_b = (force == "b") if force else (rng.random() >= frac_a)
        mul, n, off = (mul_b, n_b, offB) if is_b else (mul_a, n_a, BASE)
        task[b] = int(is_b)
        g = rng.integers(0, n, size=n_ops)
        s = 0
        for t in range(n_ops):
            s = mul[g[t], s]
            tok[b, 1 + t] = off + g[t]
            tgt[b, 1 + t] = off + n + s
        msk[b, 1:] = True
    return tok, tgt, msk, task


# ----------------------------------------------------------------- k-hop ----
def make_khop_batch(rng, B, n_sym, n_q, k_max, sorted_pairs=False,
                    ablate_pairs=False):
    """Planted Hamiltonian cycle over n_sym symbols; pairs [key ARROW succ]
    (shuffled unless sorted_pairs — the leakage control arm); queries
    [QUERY start k answer]. ablate_pairs replaces the pair region with
    uniform noise (chance-level audit: acc must be exactly 1/n_sym).
    k tokens sit at the FIXED offset BASE (before symbols) so vocab layout
    is identical across train/eval lengths (n_sym varies, k_max doesn't)."""
    ktok = BASE                                         # k tokens, k=1..k_max
    sym = BASE + k_max                                  # symbols follow
    T = 1 + 3 * n_sym + 4 * n_q
    tok = np.zeros((B, T), dtype=np.int64)
    msk = np.zeros((B, T), dtype=bool)
    diff = np.zeros((B, n_q), dtype=np.int64)           # difficulty sidecar
    tok[:, 0] = BOS
    for b in range(B):
        cyc = rng.permutation(n_sym)                    # cyc[i] -> cyc[i+1]
        succ = np.empty(n_sym, dtype=np.int64)
        succ[cyc] = np.roll(cyc, -1)
        order = np.arange(n_sym) if sorted_pairs else rng.permutation(n_sym)
        keys = cyc[order] if sorted_pairs else rng.permutation(n_sym)
        p = 1
        for key in keys:
            tok[b, p:p + 3] = (sym + key, ARROW, sym + succ[key])
            p += 3
        if ablate_pairs:
            tok[b, 1:p] = rng.integers(sym, sym + n_sym, size=p - 1)
        for qi in range(n_q):
            start = rng.integers(0, n_sym)
            k = rng.integers(1, k_max + 1)
            a = start
            for _ in range(k):
                a = succ[a]
            tok[b, p:p + 4] = (QUERY, sym + start, ktok + k - 1, sym + a)
            msk[b, p + 3] = True
            diff[b, qi] = k
            p += 4
    return tok, msk, diff


def make_niah_batch(rng, B, n_pairs, v_len, n_q, key_pool=512, val_pool=64,
                    pairs_min=None):
    """From-scratch respell-MQAR (is the respell tax structural or
    data-pressure?): [BOS, k1, ARROW, v11..v1L, SEP...] pairs shuffled,
    then queries [QUERY, k, v1..vL] — the model must EMIT the full
    multi-token high-entropy value (the flooding-sensitive shape). Loss on
    all value tokens of queries; diff sidecar = within-value position
    (1..v_len) -> the head-tax curve, trained-on. Keys and value tokens
    from disjoint pools."""
    SEP = 4
    key0 = BASE
    val0 = BASE + key_pool
    T = 1 + n_pairs * (v_len + 3) + n_q * (v_len + 2)
    tok = np.zeros((B, T), dtype=np.int64)
    msk = np.zeros((B, T), dtype=bool)
    diff = np.zeros((B, T), dtype=np.int64)
    tok[:, 0] = BOS
    for b in range(B):
        np_b = n_pairs if pairs_min is None else int(
            rng.integers(pairs_min, n_pairs + 1))
        keys = rng.choice(key_pool, size=np_b, replace=False)
        vals = rng.integers(0, val_pool, size=(n_pairs, v_len))
        p = 1
        for i in rng.permutation(np_b):
            tok[b, p] = key0 + keys[i]
            tok[b, p + 1] = ARROW
            tok[b, p + 2:p + 2 + v_len] = val0 + vals[i]
            tok[b, p + 2 + v_len] = SEP
            p += v_len + 3
        for qi in rng.choice(np_b, size=min(n_q, np_b), replace=False):
            tok[b, p] = QUERY
            tok[b, p + 1] = key0 + keys[qi]
            tok[b, p + 2:p + 2 + v_len] = val0 + vals[qi]
            msk[b, p + 2:p + 2 + v_len] = True
            diff[b, p + 2:p + 2 + v_len] = np.arange(1, v_len + 1)
            p += v_len + 2
    return tok, msk, diff


# ---------------------------------------------------------------- parity ----
def make_parity_batch(rng, B, n_ops):
    bits = rng.integers(0, 2, size=(B, n_ops))
    s = np.bitwise_xor.accumulate(bits, axis=1)
    T = 1 + 2 * n_ops
    tok = np.zeros((B, T), dtype=np.int64)
    msk = np.zeros((B, T), dtype=bool)
    tok[:, 0] = BOS
    tok[:, 1::2] = BASE + bits                          # bits: 8,9
    tok[:, 2::2] = BASE + 2 + s                         # states: 10,11
    msk[:, 2::2] = True
    return tok, msk


# ---------------------------------------------------------------- CA (P-rung) ----
# Cellular-automata prediction — the P-complete rung (plans/p_rung_spec.md).
# Rule 110 is P-complete (nonlinear fold over the row; no small-algebra summary, so
# the associative-scan escape hatch is structurally unavailable). Rule 90 is its
# LINEAR GF(2) twin (next = left XOR right — scannable). The 110/90 pair isolates
# exactly nonlinearity-of-the-fold. Tokens: cells = BASE, BASE+1; rule tag = BASE+2
# (+0 R90 / +1 R110); time tag = BASE+4 + t. Periodic boundary. Tagged (at-position)
# targets, leak-free (SILENT never sees intermediate rows).
CA_RULE = {90:  np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int64),   # l XOR r (linear)
           110: np.array([0, 1, 1, 1, 0, 1, 1, 0], dtype=np.int64)}   # P-complete
CA_RULE_TOK = BASE + 2            # +0 = R90, +1 = R110
CA_TIME_TOK = BASE + 4            # + t


def _ca_step(row, table):
    """One CA step, vectorized over batch, periodic boundary. row [B,w] in {0,1}."""
    l = np.roll(row, 1, axis=1)
    r = np.roll(row, -1, axis=1)
    return table[4 * l + 2 * row + r]


def ca_rollout(rng, B, w, t, rule):
    """[B, t+1, w] space-time diagram from a random row 0."""
    table = CA_RULE[rule]
    rows = np.empty((B, t + 1, w), dtype=np.int64)
    rows[:, 0] = rng.integers(0, 2, size=(B, w))
    for k in range(t):
        rows[:, k + 1] = _ca_step(rows[:, k], table)
    return rows


def ca_certify(rng, w, t, rule, B=4096):
    """Per-(rule,w,t) non-degeneracy certificate. Returns (marginal, short_cycle_frac):
      marginal = P(answer bit == 1) at depth t — want ~0.5 (a biased target lets a
                 constant predictor beat chance).
      short_cycle_frac = fraction of orbits whose state at t already occurred at some
                 step < t (fixed point / cycle shorter than t) — a "deep-t" query on
                 such an orbit is SECRETLY SHALLOW (row_t = row_{t mod p}), so the depth
                 axis is corrupted even though the marginal stays 0.5. This is the BINDING
                 degeneracy for Rule 90 (linear, period O(w)); per-width checks miss it.
    A cell is non-degenerate iff 0.4<=marginal<=0.6 AND short_cycle_frac < 0.02."""
    rows = ca_rollout(rng, B, w, t, rule)                 # [B, t+1, w]
    marg = float(rows[:, t].mean())
    rt = rows[:, t][:, None, :]
    short = float((rows[:, :t, :] == rt).all(-1).any(1).mean()) if t > 0 else 0.0
    return marg, short


def ca_cell_ok(marg, short):
    return (0.4 <= marg <= 0.6) and (short < 0.02)


def make_ca_batch(rng, B, w, t, rule, fmt="silent", pad=0, gap=1):
    """P-rung CA task (returns tok[B,T], tgt[B,T], msk[B,T], tagged convention).
      silent = [BOS, RULE, TIME_t, row0(w), THINK×pad, ANS×w] -> ANS positions target
               row_t. Separate answer region (uniform across archs); `pad` THINK tokens
               give `pad` genuine post-query recurrent steps — the COMPUTE-STARVATION
               control (p_rung §5.1/RO-3): pad=0 starves token-recurrences (~1 step for
               t iterations); pad=t feeds them. DM's sweeps supply iteration regardless of
               pad, so pad rescuing the LSTM (not DM) is the registered sweeps-vs-steps fork.
      trace  = [BOS, RULE, row0 .. row_{t-1}] -> predict next row per cell (depth-1 sanity).
      leaky  = FILL-IN curriculum (format B): full diagram rows 0..t; rows at steps that are
               multiples of `gap` (plus 0 and t) are GIVEN as input, the rest are silent
               targets to bridge. Gap-DOUBLING schedule (gap=1,2,4,..,t then SILENT) makes
               each stage a silent bridge of depth `gap`; endpoint (row-0-only) == SILENT."""
    rows = ca_rollout(rng, B, w, t, rule)
    rtok = CA_RULE_TOK + (0 if rule == 90 else 1)
    if fmt == "silent":
        T = 3 + w + pad + w
        tok = np.zeros((B, T), dtype=np.int64)
        tgt = np.zeros((B, T), dtype=np.int64)
        msk = np.zeros((B, T), dtype=bool)
        tok[:, 0] = BOS
        tok[:, 1] = rtok
        tok[:, 2] = CA_TIME_TOK + t
        tok[:, 3:3 + w] = BASE + rows[:, 0]           # input row0
        tok[:, 3 + w:3 + w + pad] = PAD               # THINK tokens (pad recurrent steps)
        a = 3 + w + pad                                # answer region start
        tok[:, a:a + w] = QUERY                        # answer placeholders
        tgt[:, a:a + w] = BASE + rows[:, t]            # target = row_t
        msk[:, a:a + w] = True
    elif fmt == "trace":
        T = 2 + t * w
        tok = np.zeros((B, T), dtype=np.int64)
        tgt = np.zeros((B, T), dtype=np.int64)
        msk = np.zeros((B, T), dtype=bool)
        tok[:, 0] = BOS
        tok[:, 1] = rtok
        tok[:, 2:] = (BASE + rows[:, :t]).reshape(B, t * w)
        tgt[:, 2:] = (BASE + rows[:, 1:t + 1]).reshape(B, t * w)
        msk[:, 2:] = True
    elif fmt == "leaky":
        # rows given (input) vs predicted (silent targets). gap==t -> only {0,t} given.
        given = set(range(0, t + 1, max(1, gap))) | {0, t}
        T = 2 + (t + 1) * w
        tok = np.zeros((B, T), dtype=np.int64)
        tgt = np.zeros((B, T), dtype=np.int64)
        msk = np.zeros((B, T), dtype=bool)
        tok[:, 0] = BOS
        tok[:, 1] = rtok
        for k in range(t + 1):
            c = 2 + k * w
            if k in given:
                tok[:, c:c + w] = BASE + rows[:, k]    # given row (input)
            else:
                tok[:, c:c + w] = QUERY                # placeholder
                tgt[:, c:c + w] = BASE + rows[:, k]    # silent target (bridge)
                msk[:, c:c + w] = True
    else:
        raise ValueError(f"ca fmt {fmt}")
    return tok, tgt, msk


# ---------------------------------------------------------------- match3 ----
def make_match3_batch(rng, B, n_ops, M=151, W=16, mask_from=None):
    """Target_t = does any triple within the trailing window W of x[..t]
    sum to 0 mod M. Rolling pair-sum counts -> O(T*W).

    v2 (post grid-1 collapse): grid-1 defaults (M=720, W=32) were
    unlearnable — all three archs converged to constant-'no' (accs exactly
    equal to per-set majority rates). Retuned M=151/W=16 for a learnable
    pair-sum structure, and mask_from (default W) scores only full-window
    positions so the base rate is stationary in sequence length (the old
    window-fill gradient made majority-rate vary 0.33-0.44 with length)."""
    x = rng.integers(0, M, size=(B, n_ops))
    y = np.zeros((B, n_ops), dtype=np.int64)
    for b in range(B):
        cnt = np.zeros(M, dtype=np.int32)               # pair sums in window
        win = []
        for t in range(n_ops):
            v = x[b, t]
            if len(win) >= 2 and cnt[(-v) % M] > 0:
                y[b, t] = 1
            for u in win:                               # add pairs with v
                cnt[(u + v) % M] += 1
            win.append(v)
            if len(win) > W - 1:                        # evict oldest
                old = win.pop(0)
                for u in win:
                    cnt[(old + u) % M] -= 1
    T = 1 + 2 * n_ops
    tok = np.zeros((B, T), dtype=np.int64)
    msk = np.zeros((B, T), dtype=bool)
    tok[:, 0] = BOS
    tok[:, 1::2] = BASE + x                             # values: 8..8+M-1
    tok[:, 2::2] = BASE + M + y                         # no/yes: 8+M, 8+M+1
    msk[:, 2::2] = True
    if mask_from is None:
        mask_from = W
    msk[:, :1 + 2 * mask_from] = False                  # full-window positions only
    return tok, msk


def match3_base_rate(M=151, W=16, n=200_000, seed=0):
    rng = np.random.default_rng(seed)
    _, msk = None, None
    tok, msk = make_match3_batch(rng, 8, n // 8, M=M, W=W)
    return float((tok[msk] - (BASE + M)).mean())


def step_rng(task_id, seed, split, step):
    return np.random.default_rng([task_id, seed, split, step])


# ------------------------------------------------------------- unit tests ---
if __name__ == "__main__":
    # Trap 1: pin the Cayley convention (review by eye before any arm runs)
    mul, n = cayley("s5")
    assert n == 120
    perms = sorted(itertools.permutations(range(5)))
    assert perms[0] == (0, 1, 2, 3, 4) and mul[0].tolist() == list(range(n))
    assert mul[:, 0].tolist() == list(range(n)), "identity right-mult"
    rng = np.random.default_rng(0)
    for _ in range(200):                                # associativity
        a, b, c = rng.integers(0, n, 3)
        assert mul[mul[a, b], c] == mul[a, mul[b, c]]
    swap01 = perms.index((1, 0, 2, 3, 4))
    swap12 = perms.index((0, 2, 1, 3, 4))
    # (swap01 @ swap12)(x) = swap01(swap12(x)): 0->1, 1->2, 2->0
    assert perms[mul[swap01, swap12]] == (1, 2, 0, 3, 4), "known product"
    # running product matches naive function application
    g = rng.integers(0, n, 12)
    s = 0
    seq_fn = list(range(5))                             # apply ops to a list
    for gi in g:
        s = mul[gi, s]
        seq_fn = [perms[gi][x] for x in seq_fn]
    assert perms[s] == tuple(seq_fn), "running product == naive application"
    print("cayley s5: identity/associativity(200)/known-product/naive-xcheck OK")

    mulz, nz = cayley("z10")
    assert nz == 10 and mulz[3, 4] == 7 and mulz[7, 8] == 5
    print("cayley z10 OK")

    # khop: answers actually lie k hops along the planted cycle
    KM = 8
    tok, msk, diff = make_khop_batch(np.random.default_rng(1), 4, 16, 6, KM)
    assert msk.sum() == 4 * 6
    sym0 = BASE + KM
    # rebuild succ from emitted pairs and re-walk every query
    for b in range(4):
        succ = {}
        p = 1
        for _ in range(16):
            succ[tok[b, p] - sym0] = tok[b, p + 2] - sym0
            p += 3
        for _ in range(6):
            assert tok[b, p] == QUERY
            a = tok[b, p + 1] - sym0
            k = tok[b, p + 2] - BASE + 1
            for _ in range(k):
                a = succ[a]
            assert tok[b, p + 3] - sym0 == a
            p += 4
    print("khop: emitted pairs re-walk to emitted answers OK")

    tok, msk = make_parity_batch(np.random.default_rng(2), 4, 32)
    assert ((tok[:, 2::2] - 10) == np.bitwise_xor.accumulate(tok[:, 1::2] - 8,
                                                             axis=1)).all()
    print("parity OK")

    # a5: even subgroup, order 60, closed, identity first
    mula, na = cayley("a5")
    assert na == 60 and mula[0].tolist() == list(range(60))
    for _ in range(200):
        i, j = rng.integers(0, 60, 2)
        assert 0 <= mula[i, j] < 60
    print("cayley a5: order-60 even subgroup, closed, identity OK")

    # v3 tagging: every target re-derives from the ops prefix
    tokt, tgtt, mskt = make_group_tagged_batch(np.random.default_rng(4),
                                               4, 24, mula, na)
    for b in range(4):
        st = 0
        for i in range(1, 25):
            st = mula[tokt[b, i] - BASE, st]
            assert tgtt[b, i] - (BASE + na) == st and mskt[b, i]
    assert not (tokt >= BASE + na).any(), "states leaked into input!"
    print("tagged format: targets re-derived, zero state leak OK")

    br = match3_base_rate()
    print(f"match3 base rate (M=151, W=16, full-window only): {br:.3f} "
          f"(Trap 3: want ~0.5)")
    assert 0.35 < br < 0.65, "tune M/W"

    # v2 final-state format: every episode's target re-derives from its ops
    mul5, n5 = mul, n
    tok, msk, diff = make_group_final_batch(np.random.default_rng(3), 8, 96,
                                            mul5, n5, k_max=12)
    n_ep = 0
    for b in range(8):
        for pos in np.nonzero(msk[b])[0]:
            k = diff[b, pos]
            g = tok[b, pos - 1 - k:pos - 1] - BASE
            assert tok[b, pos - 1] == QUERY
            s = 0
            for gi in g:
                s = mul5[gi, s]
            assert tok[b, pos] - (BASE + n5) == s
            n_ep += 1
    assert n_ep > 8 * 3, "episode packing too sparse"
    print(f"group final-format: {n_ep} episodes re-derived OK")
    print("all statetrack generator tests passed")


# ---------------------------------------------------------------- matinv ----
# NC2 rung: matrix inversion / linear-system solving over GF(p). Inversion (and DET,
# solving Ax=b) is complete for the class DET ⊆ NC2 (Csanky/Berkowitz) and is NOT known
# to be in NC1 — one rung above A5 (NC1-complete word problem). The "mul" variant (A·B)
# is the shallow twin: each output entry is a parity of n products, TC0.
# Silent format (uniform across archs, same convention as the CA rung):
#   inv  : [BOS, OP, SIZE_n, A(n*n row-major), THINK*pad, ANS*(n*n)]  -> A^{-1}
#   solve: [BOS, OP, SIZE_n, A(n*n), b(n),      THINK*pad, ANS*n]      -> x with Ax=b
#   mul  : [BOS, OP, SIZE_n, A(n*n), B(n*n),    THINK*pad, ANS*(n*n)]  -> A·B
# Entries e in 0..p-1 are tokens BASE+e; OP tokens BASE+p+{0,1,2}; SIZE token BASE+p+3+n.
MI_OPS = {"inv": 0, "solve": 1, "mul": 2}


def mi_op_tok(p, op):
    return BASE + p + MI_OPS[op]


def mi_size_tok(p, n):
    return BASE + p + 3 + n


def mi_vocab(p, n_max):
    return BASE + p + 3 + n_max + 1


def gf_inv_batch(A, p):
    """Batched Gauss-Jordan inverse over GF(p). A [B,n,n] ints. Returns (inv [B,n,n],
    ok [B] bool). Singular rows get ok=False (inv garbage there)."""
    B, n, _ = A.shape
    M = np.concatenate([A % p, np.tile(np.eye(n, dtype=np.int64), (B, 1, 1))], axis=2)
    ok = np.ones(B, dtype=bool)
    bidx = np.arange(B)
    for c in range(n):
        # pivot row r>=c with M[b,r,c]!=0
        nz = M[:, c:, c] != 0                                   # [B, n-c]
        has = nz.any(1)
        ok &= has
        piv = c + np.argmax(nz, axis=1)                          # first nonzero (or c if none)
        # swap rows c and piv
        rc, rp = M[bidx, c].copy(), M[bidx, piv].copy()
        M[bidx, c], M[bidx, piv] = rp, rc
        # normalise pivot row
        pv = M[:, c, c] % p
        inv_pv = np.array([pow(int(v), p - 2, p) if v else 1 for v in pv], dtype=np.int64)
        M[:, c] = (M[:, c] * inv_pv[:, None]) % p
        # eliminate column c from all other rows
        f = M[:, :, c].copy(); f[:, c] = 0                       # [B,n]
        M = (M - f[:, :, None] * M[:, c][:, None, :]) % p
    return M[:, :, n:], ok


def gl_sample(rng, B, n, p):
    """B uniformly random INVERTIBLE n×n matrices over GF(p) (rejection; accept≈0.29 at p=2)."""
    A = rng.integers(0, p, size=(B, n, n))
    inv, ok = gf_inv_batch(A, p)
    tries = 0
    while not ok.all():
        k = int((~ok).sum())
        A[~ok] = rng.integers(0, p, size=(k, n, n))
        inv2, ok2 = gf_inv_batch(A[~ok], p)
        inv[~ok] = inv2
        ok[np.where(~ok)[0][ok2]] = True
        tries += 1
        if tries > 200:
            raise RuntimeError("gl_sample: rejection stalled")
    return A, inv


def mi_certify(rng, n, p, op, B=2048):
    """Non-degeneracy certificate: (answer-symbol marginal max over symbols, identity-fraction).
    marginal ~ 1/p wanted; identity-fraction = share of answers equal to a trivial guess
    (the input itself for inv, b for solve, A for mul) — must be ~ chance."""
    tok, tgt, msk = make_matinv_batch(rng, B, n, p, op)
    ans = (tgt[msk] - BASE)
    marg = max(float((ans == s).mean()) for s in range(p))
    A = (tok[:, 3:3 + n * n] - BASE).reshape(B, n, n)
    tg = tgt[msk].reshape(B, -1) - BASE
    triv = A.reshape(B, -1) if op in ("inv", "mul") else (tok[:, 3 + n * n:3 + n * n + n] - BASE)
    ident = float((tg == triv[:, :tg.shape[1]]).mean())
    return marg, ident


def make_matinv_batch(rng, B, n, p=2, op="inv", pad=0):
    """Returns (tok[B,T], tgt[B,T], msk[B,T]) in the tagged (at-position) convention."""
    A, Ainv = gl_sample(rng, B, n, p)
    nn = n * n
    if op == "inv":
        extra, ans = np.zeros((B, 0), dtype=np.int64), Ainv.reshape(B, nn)
    elif op == "solve":
        b = rng.integers(0, p, size=(B, n, 1))
        x = (Ainv @ b) % p                                       # A x = b  <=>  x = A^{-1} b
        extra, ans = b.reshape(B, n), x.reshape(B, n)
    elif op == "mul":
        Bm = rng.integers(0, p, size=(B, n, n))
        extra, ans = Bm.reshape(B, nn), ((A @ Bm) % p).reshape(B, nn)
    else:
        raise ValueError(op)
    T = 3 + nn + extra.shape[1] + pad + ans.shape[1]
    tok = np.zeros((B, T), dtype=np.int64)
    tgt = np.zeros((B, T), dtype=np.int64)
    msk = np.zeros((B, T), dtype=bool)
    tok[:, 0] = BOS
    tok[:, 1] = mi_op_tok(p, op)
    tok[:, 2] = mi_size_tok(p, n)
    tok[:, 3:3 + nn] = BASE + A.reshape(B, nn)
    c = 3 + nn
    tok[:, c:c + extra.shape[1]] = BASE + extra
    c += extra.shape[1]
    tok[:, c:c + pad] = PAD
    c += pad
    tok[:, c:c + ans.shape[1]] = QUERY
    tgt[:, c:c + ans.shape[1]] = BASE + ans
    msk[:, c:c + ans.shape[1]] = True
    return tok, tgt, msk


def _test_matinv():
    rng = np.random.default_rng(0)
    for p in (2, 3):
        for n in (2, 3, 4, 6, 8):
            A, Ainv = gl_sample(rng, 512, n, p)
            I = np.tile(np.eye(n, dtype=np.int64), (512, 1, 1))
            assert (((A @ Ainv) % p) == I).all() and (((Ainv @ A) % p) == I).all(), (p, n)
            tok, tgt, msk = make_matinv_batch(rng, 64, n, p, "solve")
            Ab = (tok[:, 3:3 + n * n] - BASE).reshape(64, n, n); b = tok[:, 3 + n * n:3 + n * n + n] - BASE
            x = tgt[msk].reshape(64, n) - BASE
            assert (((Ab @ x[:, :, None]) % p).reshape(64, n) == b).all(), (p, n)
            m, ident = mi_certify(rng, n, p, "inv", B=1024)
            print(f"matinv p{p} n{n}: inverse verified; inv answer marginal max={m:.3f} trivial-guess agreement={ident:.3f}")
    print("matinv OK")


# ------------------------------------------------------------------- dfa ----
# C-RASP length-generalization protocol (Yang, Veseli et al. 2026, "Algebraic Decomposition
# Theory for Transformer Length Generalization", Fig. 3): DFA STATE PREDICTION over prefixes —
# input = symbol stream, target at every position = the DFA state after that symbol (tagged
# convention). Train on lengths [lmin, 50], evaluate in bins out to 500. Languages IN C-RASP
# (iterated wreath products of Z) length-generalize on transformers; FLIP-FLOPS and GROUPS
# (Krohn-Rhodes units) are NOT in C-RASP and collapse right after the training range.
def dfa_lib(name):
    """Returns (trans[n_states, n_sym] int, n_states, n_sym, in_crasp). trans[q, a] = next state.
    Groups: state = running product, alphabet = all elements (Cayley automaton; same convention as
    the tagged group task: s_t = g_t @ s_{t-1})."""
    if name in ("z2", "z3", "z5", "z10", "s3", "a5", "a4xz5"):
        mul, n = cayley(name)
        trans = np.asarray(mul).T.astype(np.int64)           # trans[q, g] = mul[g, q]
        return trans, n, n, False                              # finite groups: NOT in C-RASP
    if name == "flipflop":        # Sigma* b : state = last symbol (U3 flip-flop; AC0, NOT C-RASP)
        return np.array([[0, 1], [0, 1]]), 2, 2, False
    if name == "contains_a":      # Sigma* a Sigma* : IN C-RASP
        return np.array([[1, 0], [1, 1]]), 2, 2, True
    if name == "contains_ab":     # Sigma* ab Sigma* : IN C-RASP (states: none / saw a / saw ab)
        return np.array([[1, 0], [1, 2], [2, 2]]), 3, 2, True
    if name == "count_a_ge3":     # #a >= 3 : IN C-RASP (counting), states 0..3 saturating
        return np.array([[1, 0], [2, 1], [3, 2], [3, 3]]), 4, 2, True
    raise ValueError(name)


def make_dfa_batch(rng, B, T, trans, n_states, n_sym, sep=False):
    """sep=False: [BOS, a1..aT], target q_t AT the symbol position (tagged).
    sep=True (the C-RASP paper's protocol): [BOS, &, a1, &, a2, &, ..., aT, &]; the state is
    predicted ONLY at separator positions (& before a1 -> initial state q0; & after a_t -> q_t), so
    the most recent symbol is reachable only through attention — kills the current-symbol shortcut
    (needed for flip-flop-type languages; groups are unaffected in substance)."""
    a = rng.integers(0, n_sym, size=(B, T))
    q = np.zeros((B, T), dtype=np.int64)
    prev = np.zeros(B, dtype=np.int64)
    for t in range(T):
        prev = trans[prev, a[:, t]]
        q[:, t] = prev
    if not sep:
        tok = np.zeros((B, 1 + T), dtype=np.int64)
        tgt = np.zeros((B, 1 + T), dtype=np.int64)
        msk = np.zeros((B, 1 + T), dtype=bool)
        tok[:, 0] = BOS
        tok[:, 1:] = BASE + a
        tgt[:, 1:] = BASE + n_sym + q
        msk[:, 1:] = True
        return tok, tgt, msk
    L = 2 + 2 * T                                   # BOS, &, (a, &) x T
    tok = np.zeros((B, L), dtype=np.int64)
    tgt = np.zeros((B, L), dtype=np.int64)
    msk = np.zeros((B, L), dtype=bool)
    tok[:, 0] = BOS
    tok[:, 1::2] = ARROW                            # separator '&' (reuse the ARROW id)
    tok[:, 2::2] = BASE + a
    tgt[:, 1] = BASE + n_sym + 0                    # initial state at the first separator
    tgt[:, 3::2] = BASE + n_sym + q                 # q_t at the separator after a_t
    msk[:, 1::2] = True
    return tok, tgt, msk


# ------------------------------------------------------------------ cvp ----
# Keyed CIRCUIT VALUE PROBLEM (P-complete; Ladner 1975) over XOR-MAJORITY gates
#   v_k = v_c XOR MAJ(v_a, v_b, v_d)      ({XOR, MAJ} + constants is universal: MAJ(a,b,0)=AND,
# MAJ(a,b,1)=OR; input literals supply the constants). The gate keeps values BALANCED
# (MAJ of 3 balanced bits is balanced, XOR with it stays balanced) and has NO strong depth-1
# shortcut: copying any single operand agrees only ~50-60% (Toffoli's copy-c leaks 75%;
# AND/OR chains drift to constants). Gates arrive in topological order, one block per gate
# [GID_k, GID_a, GID_b, GID_d, GID_c] (all < k); inputs are the first m gates [GID_k, BIT_b].
# Sequential computation is ALIGNED WITH THE TOKEN ORDER: with `chain=1` the carry operand
# c is always gate k-1, so circuit depth == gate index — the streaming-P setting where
# token-recurrence gets depth for free and a depth-L transformer must chain L lookups per
# position. m (input bits) sets the entropy: m>=32 keeps per-circuit marginals ~0.5.
#   tagged : target VAL at the last token of every gate block (dense, leak-free: values
#            are targets only, never inputs) + queries [QUERY, GID_k] -> VAL at the end.
#   silent : no dense targets; [THINK x pad] then q queries. pad = extra sequential
#            positions granted after the circuit (the latent-thinking axis).
# Vocab (n_max fixes the layout across an n-sweep): GID BASE..BASE+m+n_max-1; BIT 2
# (input literals); VAL 2 (targets, disjoint).
CVP_TOK_PER_GATE = 5


def cvp_vocab(n_max, m):
    return BASE + (m + n_max) + 2 + 2


CVP_GATE = ["xormaj"]   # gate ablation for the depth-1 control: xormaj (paper) | copy (v_k = v_a, one pointer) | xor2 (v_k = v_a ^ v_c)


def _cvp_gate(va, vb, vd, vc):
    g = CVP_GATE[0]
    if g == "copy":
        return va
    if g == "xor2":
        return va ^ vc
    return vc ^ _maj(va, vb, vd)


def _maj(a, b, d):
    return ((a + b + d) >= 2).astype(np.int64)


def make_cvp_batch(rng, B, n, m=32, fmt="tagged", pad=0, chain=1.0, n_queries=4,
                   n_max=None, n_topo=1, window=0, m_max=None, leak_every=0, addr="abs"):
    """n_topo > 1: split the batch over n_topo INDEPENDENT circuits (eval sets must not be a
    single topology — a constant-per-gate predictor scores the same every eval otherwise)."""
    if n_topo > 1:
        sizes = [B // n_topo + (1 if i < B % n_topo else 0) for i in range(n_topo)]
        parts = [make_cvp_batch(rng, sz, n, m, fmt, pad, chain, n_queries, n_max, 1, window, m_max, leak_every, addr) for sz in sizes if sz > 0]
        return tuple(np.concatenate([pt[i] for pt in parts], axis=0) for i in range(4))
    return _make_cvp_batch_one(rng, B, n, m, fmt, pad, chain, n_queries, n_max, window, m_max, leak_every, addr)


def _make_cvp_batch_one(rng, B, n, m=32, fmt="tagged", pad=0, chain=1.0, n_queries=4,
                        n_max=None, window=0, m_max=None, leak_every=0, addr="abs"):
    """addr="rel": operand tokens are RELATIVE offsets (how many logical gates back, 1..k) instead of absolute
    gate ids — the attention-friendly addressing (RoPE relative position) suggested as the baseline-fairness
    control; the chain operand is then always offset 1. Same vocab layout (offset d uses id slot d-1)."""
    """leak_every=g>0 (tagged): after every g-th gate block the TRUE value is appended as an INPUT token
    (BIT range), so the silent depth between revealed values is bounded by g — the LEAKY curriculum
    (cf. CA leaky gap-doubling): g=1 is depth-1 supervised lookup+gate, g doubles, g=0 is fully silent."""
    """Layout is fixed by (m_max, n_max) so a curriculum over (m, n) keeps every token id stable:
    input ids 0..m-1 (of m_max), gate ids m_max..m_max+n-1, BIT/VAL after m_max+n_max."""
    """Returns (tok[B,T], tgt[B,T], msk[B,T], depth[B,T]). n = #computed gates, m = #inputs.
    ONE circuit topology per batch (shared schedule, like keyed); rows differ in the input
    bits => vectorised over B. `chain` = probability that gate k's carry operand c is gate
    k-1 (1.0 => depth grows 1 per gate; 0.0 => random DAG, depth ~ log n)."""
    if n_max is None:
        n_max = n
    if m_max is None:
        m_max = m
    G = m + n                                           # logical gates: inputs 0..m-1, gates m..G-1
    gid = np.concatenate([np.arange(m), m_max + np.arange(n)])   # logical index -> token id offset
    GIDB = BASE
    BITB = BASE + (m_max + n_max)
    VALB = BITB + 2
    src = np.zeros((G, 4), dtype=np.int64)          # a, b, d, c (logical indices)
    depth_g = np.zeros(G, dtype=np.int64)
    for k in range(m, G):
        lo = max(0, k - window) if window else 0          # window: operands from the last `window` gates only
        src[k, :3] = lo + rng.choice(k - lo, size=3, replace=(k - lo < 3))
        src[k, 3] = k - 1 if rng.random() < chain else int(rng.integers(lo, k))
        depth_g[k] = 1 + depth_g[src[k]].max()
    val = np.zeros((B, G), dtype=np.int64)
    val[:, :m] = rng.integers(0, 2, size=(B, m))
    for k in range(m, G):
        a, b, d, c = src[k]
        val[:, k] = _cvp_gate(val[:, a], val[:, b], val[:, d], val[:, c])
    q = min(n_queries, n)
    qs = np.concatenate([[G - 1], rng.choice(np.arange(m, G - 1), size=q - 1, replace=False)]) if q > 1 else np.array([G - 1])
    leak_ks = set(k for k in range(m, G) if leak_every and fmt == "tagged" and ((k - m + 1) % leak_every == 0))
    T = 1 + 2 * m + CVP_TOK_PER_GATE * n + len(leak_ks) + (pad if fmt == "silent" else 0) + 2 * q
    tok = np.zeros((B, T), dtype=np.int64); tgt = np.zeros((B, T), dtype=np.int64)
    msk = np.zeros((B, T), dtype=bool); depth = np.zeros((B, T), dtype=np.int64)
    tok[:, 0] = BOS
    p = 1
    for k in range(m):
        tok[:, p], tok[:, p + 1] = GIDB + gid[k], BITB + val[:, k]
        p += 2
    for k in range(m, G):
        tok[:, p] = GIDB + gid[k]
        if addr == "rel":
            tok[:, p + 1:p + 5] = GIDB + (k - src[k] - 1)          # offset d = k - src  (>=1) -> slot d-1
        else:
            tok[:, p + 1:p + 5] = GIDB + gid[src[k]]
        if fmt == "tagged":
            tgt[:, p + 4] = VALB + val[:, k]; msk[:, p + 4] = True; depth[:, p + 4] = depth_g[k]
        p += CVP_TOK_PER_GATE
        if k in leak_ks:                                   # revealed value (input token, BIT range)
            tok[:, p] = BITB + val[:, k]; p += 1
    if fmt == "silent":
        tok[:, p:p + pad] = PAD
        p += pad
    for k in qs:
        tok[:, p], tok[:, p + 1] = QUERY, GIDB + (int(G - 1 - int(k)) if addr == "rel" else int(gid[int(k)]))   # rel: gates back from the last
        tgt[:, p + 1] = VALB + val[:, int(k)]; msk[:, p + 1] = True; depth[:, p + 1] = depth_g[int(k)]
        p += 2
    assert p == T
    return tok, tgt, msk, depth


def _cvp_values_from_tokens(tok, m, n):
    """Only valid for the default layout (m_max == m, n_max == n) — used by the certificate."""
    G = m + n; GIDB = BASE; BITB = BASE + m + n
    B = tok.shape[0]; vals = np.zeros((B, G), dtype=np.int64); p = 1
    for k in range(m):
        vals[:, k] = tok[:, p + 1] - BITB; p += 2
    ops = []
    for k in range(m, G):
        a, b, d, c = (tok[0, p + 1:p + 5] - GIDB).tolist()
        vals[:, k] = _cvp_gate(vals[:, a], vals[:, b], vals[:, d], vals[:, c]); ops.append((a, b, d, c)); p += CVP_TOK_PER_GATE
    return vals, ops


def cvp_certify(rng, n, m=32, chain=1.0, B=4096, n_batches=8, window=0):
    # (addressing does not change the values, so the certificate is addr-independent)
    """(marginal, shortcut) over the FINAL gate: marginal = max P(v); shortcut = best
    agreement of {copy/negate one operand} with the true value (depth-1 leak)."""
    marg = short = 0.0
    for _ in range(n_batches):
        tok, tgt, msk, depth = make_cvp_batch(rng, B, n, m=m, fmt="silent", pad=0, chain=chain, n_queries=1, window=window)
        vals, ops = _cvp_values_from_tokens(tok, m, n)
        v = vals[:, -1]
        marg += max(v.mean(), 1 - v.mean())
        short += max(max((vals[:, x] == v).mean(), (vals[:, x] != v).mean()) for x in ops[-1])
    return marg / n_batches, short / n_batches


def cvp_cell_ok(marg, short):
    return marg < 0.65 and short < 0.75


def _test_cvp():
    rng = np.random.default_rng(0)
    tok, tgt, msk, depth = make_cvp_batch(rng, 4, 16, m=8, fmt="tagged", chain=1.0)
    assert msk.sum() == 4 * (16 + 4) and depth[msk].max() == 16, (msk.sum(), depth[msk].max())
    tok2, tgt2, msk2, d2 = make_cvp_batch(rng, 4, 16, m=8, fmt="silent", pad=5, chain=0.0)
    assert tok2.shape[1] == 1 + 16 + 80 + 5 + 8 and msk2.sum() == 16 and d2[msk2].max() < 16
    tok3, tgt3, msk3, d3 = make_cvp_batch(rng, 4, 16, m=8, fmt="tagged", leak_every=4)
    assert tok3.shape[1] == 1 + 16 + 80 + 4 + 8 and msk3.sum() == 4 * 20, tok3.shape
    vals, _ = _cvp_values_from_tokens(tok, 8, 16)
    VALB = BASE + 24 + 2
    assert (tgt[msk].reshape(4, -1)[:, :16] - VALB == vals[:, 8:]).all()
    for m in (16, 32):
        for n_ in (32, 64):
            m_, s_ = cvp_certify(np.random.default_rng(1), n_, m=m, chain=1.0, B=2048, n_batches=6)
            print(f"cvp certify m{m} n{n_} chain=1: marginal {m_:.3f} shortcut {s_:.3f} ok={cvp_cell_ok(m_, s_)}")
    m_, s_ = cvp_certify(np.random.default_rng(1), 64, m=32, chain=0.0, B=2048, n_batches=6)
    print(f"cvp certify m32 n64 chain=0: marginal {m_:.3f} shortcut {s_:.3f}  (depth ~ log n)")
    print("cvp self-test ok")


if __name__ == "__main__" and "cvp" in sys.argv:
    _test_cvp()
