"""Tabular Q-tables for the Motion Controller.

Two tables are used:
  - Q_route    (keyed by route state, actions = neighbor node names)
  - Q_velocity (keyed by velocity state, actions = {0: decel, 1: maintain, 2: accel})

Both use dict-based sparse storage so that only visited states consume memory.
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class TabularQTable:
    """Sparse dict-based Q-table with epsilon-greedy action selection.

    For ``Q_route``, values are stored as ``{state: {action_name: q_value}}``
    because the action set varies per state (neighbor nodes differ per node).

    For ``Q_velocity``, values are stored as ``{state: [q_decel, q_maintain, q_accel]}``
    with fixed action indices 0/1/2.

    Parameters
    ----------
    alpha:
        Learning rate.
    gamma:
        Discount factor.
    default_q:
        Initial Q-value for unseen (state, action) pairs.
    fixed_n_actions:
        If provided (e.g. 3 for velocity), store Q-values as a list of that
        length.  If None, store as a dict keyed by action name (for route).
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        default_q: float = 0.0,
        fixed_n_actions: Optional[int] = None,
    ) -> None:
        self.alpha = alpha
        self.gamma = gamma
        self.default_q = default_q
        self.fixed_n_actions = fixed_n_actions
        # For fixed: {state: List[float]}
        # For variable: {state: Dict[Any, float]}
        self._table: Dict[Tuple, Any] = {}

    # ------------------------------------------------------------------
    # Core access
    # ------------------------------------------------------------------

    def get_q(self, state: Tuple, action: Any) -> float:
        """Return Q(state, action), defaulting to ``default_q``."""
        entry = self._table.get(state)
        if entry is None:
            return self.default_q
        if self.fixed_n_actions is not None:
            idx = int(action)
            return entry[idx]
        return entry.get(action, self.default_q)

    def set_q(self, state: Tuple, action: Any, value: float) -> None:
        """Write Q(state, action) = value, initialising the entry if needed."""
        if state not in self._table:
            if self.fixed_n_actions is not None:
                self._table[state] = [self.default_q] * self.fixed_n_actions
            else:
                self._table[state] = {}
        entry = self._table[state]
        if self.fixed_n_actions is not None:
            entry[int(action)] = value
        else:
            entry[action] = value

    def best_action(self, state: Tuple, available_actions: List[Any]) -> Any:
        """Return the action with the highest Q-value among *available_actions*.

        Ties are broken randomly (uniform over all tied actions) to avoid
        systematic bias toward the first action in the list — critical at
        the start of training when all Q-values are equal.

        Q-values are snapshotted in a single pass before comparison so that
        concurrent Hogwild updates by other threads cannot produce an empty
        tied-action list between the max() and the filter step.
        """
        q_snapshot = [(a, self.get_q(state, a)) for a in available_actions]
        best_q = max(q for _, q in q_snapshot)
        tied = [a for a, q in q_snapshot if q == best_q]
        return random.choice(tied)

    def max_q(self, state: Tuple, available_actions: List[Any]) -> float:
        """Return max Q-value over *available_actions* for *state*."""
        if not available_actions:
            return self.default_q
        return max(self.get_q(state, a) for a in available_actions)

    # ------------------------------------------------------------------
    # Q-learning update
    # ------------------------------------------------------------------

    def update(
        self,
        state: Tuple,
        action: Any,
        reward: float,
        next_state: Tuple,
        next_available_actions: List[Any],
        done: bool = False,
    ) -> float:
        """Single-step Q-learning update.

        Returns the TD error (useful for diagnostics).
        """
        current_q = self.get_q(state, action)
        if done or not next_available_actions:
            target = reward
        else:
            target = reward + self.gamma * self.max_q(next_state, next_available_actions)
        td_error = target - current_q
        self.set_q(state, action, current_q + self.alpha * td_error)
        return td_error

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def size(self) -> int:
        """Number of states stored."""
        return len(self._table)   # len() on dict is GIL-atomic in CPython

    def n_entries(self) -> int:
        """Total number of (state, action) pairs stored."""
        count = 0
        for v in list(self._table.values()):
            count += len(v)
        return count

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the table to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "alpha": self.alpha,
                "gamma": self.gamma,
                "default_q": self.default_q,
                "fixed_n_actions": self.fixed_n_actions,
                "table": self._table,
            }, f)

    @classmethod
    def load(cls, path: str | Path) -> "TabularQTable":
        """Load a previously saved table."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            alpha=data["alpha"],
            gamma=data["gamma"],
            default_q=data["default_q"],
            fixed_n_actions=data["fixed_n_actions"],
        )
        obj._table = data["table"]
        return obj
