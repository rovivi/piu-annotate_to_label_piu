"""Viterbi decoder for limb label sequences.

Replaces argmax + greedy `fix_impossible_predictions` with globally optimal
sequence decoding under hard (multihit bracketability) and soft (transition)
constraints.

Emissions come from the transformer's logits over 3 classes:
    0 = left foot, 1 = right foot, 2 = either / hand

Hard constraints:
    Within a multihit group (simultaneous arrows), any pair of non-bracketable
    panels cannot share the same foot (L or L, R or R). Class 2 ("either") is
    treated as a wildcard that escapes the bracket constraint.

Soft constraints (transition log-penalties, all negative):
    - same-foot double-step on consecutive arrows: `double_step_penalty`
    - either->either (consecutive "hand" labels): `repeat_either_penalty`
    Tune by passing kwargs. Default values are mild so the model's emissions
    still dominate.

State expansion:
    For multihit groups of size n we expand a joint state over the n arrows
    in that group (3**n combinations). Singles use the trivial 3-state form.
    PIU multihits are tiny (n <= 4 in practice), so 3**n stays tractable.

Cost: O(sum_g 3^{n_g} * 3^{n_{g+1}}) — for n=1 (singletons) this is 9 per step,
identical to classical Viterbi.
"""
from __future__ import annotations

import itertools

import numpy as np

from piu_annotate.formats.notelines import same_foot_is_impossible


NEG_INF = -1e18


def _group_states(n: int) -> list[tuple[int, ...]]:
    """All 3**n joint limb assignments for a multihit group of size n."""
    return list(itertools.product((0, 1, 2), repeat=n))


def _group_emission(
    logp: np.ndarray,
    group_idx: list[int],
    state: tuple[int, ...],
) -> float:
    """Sum of per-arrow log-probs for assignment `state` over `group_idx`."""
    s = 0.0
    for arrow_i, limb in zip(group_idx, state):
        s += float(logp[arrow_i, limb])
    return s


def _group_violates_hard(
    state: tuple[int, ...],
    panels: list[int],
) -> bool:
    """True if any pair within group assigns same foot to non-bracketable panels."""
    n = len(state)
    if n < 2:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            f1, f2 = state[i], state[j]
            if f1 == f2 and f1 in (0, 1):
                if same_foot_is_impossible(panels[i], panels[j]):
                    return True
    return False


def _transition_cost(
    prev_state: tuple[int, ...],
    cur_state: tuple[int, ...],
    double_step_penalty: float,
    repeat_either_penalty: float,
) -> float:
    """Log-penalty for transitioning from last limb of prev group to first of cur group.

    Only the *last* arrow of the previous group and the *first* arrow of the
    current group define the consecutive-arrow transition that we penalize.
    """
    if not prev_state or not cur_state:
        return 0.0
    a = prev_state[-1]
    b = cur_state[0]
    cost = 0.0
    if a == b:
        if a in (0, 1):
            cost += double_step_penalty
        elif a == 2:
            cost += repeat_either_penalty
    return cost


def viterbi_decode(
    logits: np.ndarray,
    arrow_positions: np.ndarray,
    num_dp: np.ndarray,
    double_step_penalty: float = -0.5,
    repeat_either_penalty: float = -0.2,
) -> np.ndarray:
    """Globally optimal limb assignment under hard/soft constraints.

    Args:
        logits: (N, 3) raw logits from the transformer (any monotone score works
            — log-softmax is computed internally for numerical stability).
        arrow_positions: (N,) int panel index per arrow (0..9).
        num_dp: (N,) int multihit-group size repeated for each arrow in the
            group (same convention as `fix_impossible_predictions`).
        double_step_penalty: log-penalty added when consecutive arrows share
            the same foot (L-L or R-R). Default mild.
        repeat_either_penalty: log-penalty for consecutive "either" labels.

    Returns:
        (N,) int32 limb predictions.
    """
    N = int(len(logits))
    if N == 0:
        return np.zeros(0, dtype=np.int32)

    # numerically stable log-softmax
    m = logits.max(axis=1, keepdims=True)
    logp = (logits - m) - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))
    logp = logp.astype(np.float64)

    # Build multihit groups: list of (list[arrow_idx], list[panel])
    groups: list[tuple[list[int], list[int]]] = []
    i = 0
    while i < N:
        n = max(1, int(num_dp[i]))
        idx = list(range(i, min(i + n, N)))
        panels = [int(arrow_positions[k]) for k in idx]
        groups.append((idx, panels))
        i += len(idx)

    # Precompute legal states + emission for each group
    legal_states: list[list[tuple[int, ...]]] = []
    emissions: list[np.ndarray] = []
    for idx, panels in groups:
        states = _group_states(len(idx))
        legal = [s for s in states if not _group_violates_hard(s, panels)]
        if not legal:
            # fallback: drop hard constraint for this pathological group
            legal = states
        emit = np.array(
            [_group_emission(logp, idx, s) for s in legal],
            dtype=np.float64,
        )
        legal_states.append(legal)
        emissions.append(emit)

    G = len(groups)
    # DP tables
    dp: list[np.ndarray] = [None] * G  # type: ignore[list-item]
    back: list[np.ndarray] = [None] * G  # type: ignore[list-item]
    dp[0] = emissions[0].copy()
    back[0] = np.full(len(legal_states[0]), -1, dtype=np.int32)

    for g in range(1, G):
        prev_states = legal_states[g - 1]
        cur_states = legal_states[g]
        prev_dp = dp[g - 1]
        cur_emit = emissions[g]
        n_prev = len(prev_states)
        n_cur = len(cur_states)

        # transition matrix (n_prev, n_cur)
        trans = np.empty((n_prev, n_cur), dtype=np.float64)
        for pi, ps in enumerate(prev_states):
            for ci, cs in enumerate(cur_states):
                trans[pi, ci] = _transition_cost(
                    ps, cs, double_step_penalty, repeat_either_penalty
                )

        # scores[pi, ci] = prev_dp[pi] + trans[pi, ci] + cur_emit[ci]
        scores = prev_dp[:, None] + trans + cur_emit[None, :]
        back[g] = np.argmax(scores, axis=0).astype(np.int32)
        dp[g] = scores[back[g], np.arange(n_cur)]

    # backtrack
    preds = np.zeros(N, dtype=np.int32)
    cur_idx = int(np.argmax(dp[-1]))
    for g in range(G - 1, -1, -1):
        state = legal_states[g][cur_idx]
        for arrow_i, limb in zip(groups[g][0], state):
            preds[arrow_i] = int(limb)
        if g > 0:
            cur_idx = int(back[g][cur_idx])

    return preds
