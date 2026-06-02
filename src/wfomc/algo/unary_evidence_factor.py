from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
from typing import Callable, Iterator, TYPE_CHECKING

from wfomc.network import EvidenceGroups
from wfomc.utils import MultinomialCoefficients, Rational, RingElement, multinomial

if TYPE_CHECKING:
    from wfomc.cell_graph import Cell


def make_unary_evidence_factor(
    cells: list[Cell],
    evidence_groups: EvidenceGroups,
) -> Callable[[tuple[int, ...]], Rational]:
    """Return a config -> multiplier function for factorized unary evidence."""
    if not evidence_groups:
        return lambda _config: Rational(1, 1)

    target_sizes = tuple(size for _, size in evidence_groups)
    compatibility = tuple(
        tuple(
            all(cell.is_positive(lit.pred) == lit.positive for lit in evidence)
            for evidence, _ in evidence_groups
        )
        for cell in cells
    )
    n_groups = len(target_sizes)

    @lru_cache(maxsize=None)
    def allocations(
        total: int,
        remaining: tuple[int, ...],
        compatible: tuple[bool, ...],
    ) -> tuple[tuple[int, ...], ...]:
        alloc = [0] * n_groups
        results = []

        def rec(pos: int, left: int) -> None:
            if pos == n_groups:
                if left == 0:
                    results.append(tuple(alloc))
                return

            max_value = min(remaining[pos], left) if compatible[pos] else 0
            for value in range(max_value + 1):
                alloc[pos] = value
                rec(pos + 1, left - value)
            alloc[pos] = 0

        rec(0, total)
        return tuple(results)

    @lru_cache(maxsize=None)
    def factor(config: tuple[int, ...]) -> Rational:
        @lru_cache(maxsize=None)
        def dp(cell_idx: int, remaining: tuple[int, ...]) -> int:
            if cell_idx == len(config):
                return 1 if all(count == 0 for count in remaining) else 0

            total = 0
            for allocation in allocations(
                config[cell_idx], remaining, compatibility[cell_idx]
            ):
                next_remaining = tuple(
                    old - used for old, used in zip(remaining, allocation)
                )
                total += (
                    MultinomialCoefficients.coef(allocation)
                    * dp(cell_idx + 1, next_remaining)
                )
            return total

        return Rational(dp(0, target_sizes), 1)

    return factor



def iter_consistent_configs(
    cells: list[Cell],
    evidence_groups: EvidenceGroups,
    domain_size: int,
) -> Iterator[tuple[tuple[int, ...], RingElement]]:
    """Yield ``(cell_count_config, coefficient)`` for every configuration that
    is consistent with the factorized unary evidence.

    Without evidence this enumerates all cell-count configurations of
    ``domain_size`` elements together with their multinomial coefficient,
    exactly like iterating ``multinomial(n_cells, domain_size)``.

    With evidence, each evidence bucket ``(literals, size)`` is distributed
    *only* over the cells compatible with its literals. A forward DP over the
    buckets accumulates the per-bucket multinomial coefficients into the
    running cell-count config, so inconsistent configs are never generated
    (no enumerate-then-filter pass). The emitted coefficient already accounts
    for every bucket assignment yielding the same totals, so no global
    repeat-factor correction is required by the caller.
    """
    n_cells = len(cells)
    if not evidence_groups:
        for config in multinomial(n_cells, domain_size):
            yield config, MultinomialCoefficients.coef(config)
        return

    states: dict[tuple[int, ...], RingElement] = {(0,) * n_cells: Rational(1, 1)}
    for evidence, size in evidence_groups:
        compatible = [
            i for i, cell in enumerate(cells)
            if all(cell.is_positive(lit.pred) == lit.positive for lit in evidence)
        ]
        if not compatible:
            # No cell can host this bucket's elements: the whole problem has
            # no consistent configuration.
            return
        new_states: dict[tuple[int, ...], RingElement] = defaultdict(
            lambda: Rational(0, 1)
        )
        for base_config, acc in states.items():
            for dist in multinomial(len(compatible), size):
                mult = MultinomialCoefficients.coef(dist)
                new_config = list(base_config)
                for cell_idx, count in zip(compatible, dist):
                    new_config[cell_idx] += count
                new_states[tuple(new_config)] += acc * mult
        states = new_states

    yield from states.items()