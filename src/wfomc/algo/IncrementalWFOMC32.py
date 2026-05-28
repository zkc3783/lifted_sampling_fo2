from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
from loguru import logger

from wfomc.algo.unary_evidence_factor import make_unary_evidence_factor
from wfomc.cell_graph import build_cell_graphs
from wfomc.context import CountingState, IncrementalWFOMC3Context
from wfomc.fol import Const, Pred
from wfomc.utils import (
    MultinomialCoefficients,
    Rational,
    RingElement,
    coeff_dict,
    expand,
    multinomial,
)


Config = tuple[int, ...]
State = tuple[int, ...] # 元素当前所处的 cell + 存在量词满足状态


class ConfigSpace:
    """
    Compact immutable representation for DP configurations.

    A configuration is just a flat tuple of counts in C-order;
    shape/offset bookkeeping is centralized in this helper.
    """

    __slots__ = (
        "shape",
        "zero",
        "offset_to_state",
        "state_to_offset",
        "_nonzero_cache",
        "_cell_ext_one_offsets_cache",
    )

    def __init__(self, shape: tuple[int, ...]):
        self.shape = tuple(shape)
        ranges = [range(dim) for dim in self.shape]
        self.offset_to_state: tuple[State, ...] = tuple(product(*ranges))
        self.state_to_offset: dict[State, int] = {
            state: offset for offset, state in enumerate(self.offset_to_state)
        }
        self.zero: Config = (0,) * len(self.offset_to_state)
        self._nonzero_cache: dict[Config, tuple[State, ...]] = {}
        self._cell_ext_one_offsets_cache: dict[int, tuple[tuple[int, ...], ...]] = {}

    def offset(self, state: State) -> int:
        return self.state_to_offset[state]

    def count(self, config: Config, state: State) -> int:
        return config[self.state_to_offset[state]]

    def inc(self, config: Config, state: State, amount: int = 1) -> Config:
        offset = self.state_to_offset[state]
        return config[:offset] + (config[offset] + amount,) + config[offset + 1 :]

    def dec(self, config: Config, state: State, amount: int = 1) -> Config:
        offset = self.state_to_offset[state]
        return config[:offset] + (config[offset] - amount,) + config[offset + 1 :]

    @staticmethod
    def add(left: Config, right: Config) -> Config:
        return tuple(a + b for a, b in zip(left, right))

    @staticmethod
    def sub(left: Config, right: Config) -> Config:
        return tuple(a - b for a, b in zip(left, right))

    def nonzero_states(self, config: Config) -> tuple[State, ...]:
        cached = self._nonzero_cache.get(config)
        if cached is None:
            cached = tuple(
                self.offset_to_state[offset]
                for offset, count in enumerate(config)
                if count > 0
            )
            self._nonzero_cache[config] = cached
        return cached

    def cell_ext_one_offsets(self, num_ext: int) -> tuple[tuple[int, ...], ...]:
        """
        Offsets grouped by cell whose existential-counter slots are all 1.

        This is the tuple-config equivalent of v3's
        nowK.array[:, 1, 1, ...] slicing used to remove the 1-type
        cardinality budget before recursive traceback sampling.
        """

        cached = self._cell_ext_one_offsets_cache.get(num_ext)
        if cached is not None:
            return cached

        offsets_by_cell: list[list[int]] = [[] for _ in range(self.shape[0])]
        for offset, state in enumerate(self.offset_to_state):
            if all(state[1 + idx] == 1 for idx in range(num_ext)):
                offsets_by_cell[state[0]].append(offset)

        cached = tuple(tuple(offsets) for offsets in offsets_by_cell)
        self._cell_ext_one_offsets_cache[num_ext] = cached
        return cached


@dataclass(slots=True)
class SamplingDrNode:
    # Recursive choice: ((target_c, next_config), (weight_next, weight_current)).
    dp_recursion: list = field(default_factory=list)

    # target_c -> next_config -> [(last_target_c, weight)].
    dp_starter: dict = field(default_factory=dict)

    # target_c -> list of DP layers used during traceback.
    dp_traceback: dict = field(default_factory=dict)

    # target_c -> other states traversed by the G/H dynamic program.
    dp_order: dict = field(default_factory=dict)


@dataclass(slots=True)
class CellGraphDpTrace:
    space: ConfigSpace
    cells: list
    cell_weights: dict
    data_root: list
    data_T: dict
    data_H: dict
    data_Evi: dict
    one_type_offsets: tuple[tuple[int, ...], ...]


class ConfigUpdater:
    """
    Memoised configuration updater for efficient state transitions.

    _cache structure: {(target_c, other_c): {j: H_dict}}
    where H_dict maps (tc_new, H_config_new) → Rational weight,
    recording the cumulative weight of pairing target_c with j other_c elements.
    """

    def __init__(self, t_update_dict, space: ConfigSpace, data_H: dict):
        self.t_update_dict = t_update_dict
        self.space = space
        self._cache: dict[tuple[State, State], dict[int, dict]] = {}
        self.data_H = data_H

    def f(self, target_c: State, other_c: State, l: int):
        """Return the weighted outcome of pairing target_c with l other_c elements."""
        key = (target_c, other_c)
        sub = self._cache.get(key)
        if sub is None:
            sub = {}
            self._cache[key] = sub
            self.data_H[key] = {}
            num_start = 0
        else:
            num_start = l
            while num_start not in sub and num_start > 0:
                num_start -= 1

        if num_start == 0:
            H = {(target_c, self.space.zero): Rational(1, 1)}
        else:
            H = sub[num_start]

        for j in range(num_start + 1, l + 1):
            H_new = defaultdict(lambda: Rational(0, 1))
            H_layer = defaultdict(list)
            for (tc_old, hc_old), W in H.items():
                for (tc_new, oc_new), rij in self.t_update_dict[(tc_old, other_c)].items():
                    hc_new = self.space.inc(hc_old, oc_new)
                    H_new[(tc_new, hc_new)] += W * rij
                    H_layer[(tc_new, hc_new)].append(((tc_old, oc_new), (W, rij)))

            H = H_new
            sub[j] = H
            self.data_H[key][j] = H_layer

        return H

# ---------------------------------------------------------------------------
# Cell-graph weight builders
# ---------------------------------------------------------------------------

def build_sample_weight(cells, cell_graph, state: CountingState) -> tuple:
    """
    Construct the initial-state mapping w2t, cell weight dict w, and
    binary relationship dict rs from cell-graph data and counting state.
    """
    n_cells = len(cells)
    w2t = {}
    w = defaultdict(lambda: Rational(0, 1))
    rs = defaultdict(lambda: defaultdict(lambda: Rational(0, 1)))

    for i in range(n_cells):
        cell_weight = cell_graph.get_cell_weight(cells[i])
        logger.debug("Cell {} weight: {}", i, cell_weight)
        t = []

        for pred in state.ext_preds:
            t.append(0 if cells[i].is_positive(pred) else 1)
        logger.debug("Cell {} existential quantifier state: {}", i, t)

        for idx, (pred, param) in enumerate(zip(state.cnt_preds, state.cnt_params)):
            if cells[i].is_positive(pred):
                t.append(
                    state.cnt_remainder[idx] - 1
                    if state.exist_mod and idx in state.mod_pred_index
                    else param - 1
                )
            else:
                t.append(
                    state.cnt_remainder[idx]
                    if state.exist_mod and idx in state.mod_pred_index
                    else param
                )
        logger.debug("Cell {} counting quantifier state: {}", i, t)

        w2t[i] = tuple(t)
        w[i] = w[i] + cell_weight

        for j in range(n_cells):
            for evi_idx, evidence in enumerate(state.binary_evidence):
                
                two_tables = cell_graph.get_two_tables((cells[i], cells[j]), evidence)
                two_table_weight = sum((weight for _, weight in two_tables), Rational(0, 1))
                # two_table_weight = cell_graph.get_two_table_weight((cells[i], cells[j]), evidence)
                if two_table_weight == Rational(0, 1):
                    continue
                t_fwd, t_rev = [], []
                for pred_idx, _ in enumerate(state.ext_preds + state.cnt_preds):
                    t_rev.append(1 if (evi_idx >> (2 * pred_idx)) & 1 else 0)
                    t_fwd.append(1 if (evi_idx >> (2 * pred_idx + 1)) & 1 else 0)
                rs[(i, j)][(tuple(t_fwd), tuple(t_rev))] = (two_table_weight, two_tables)

    return w2t, w, rs


def build_sample_t_update_dict(rs, n_cells: int, state: CountingState) -> tuple:
    """Build the state transition lookup table for all cell-pair combinations."""
    t_update_dict = defaultdict(lambda: defaultdict(lambda: Rational(0, 1)))
    t_sample = defaultdict(lambda: defaultdict(list))

    n_ext = len(state.ext_preds)
    n_cnt = len(state.cnt_params)

    if state.exist_mod:
        ranges = [tuple(range(2)) for _ in state.ext_preds]
        for p, k in enumerate(state.cnt_params):
            ranges.append(
                tuple(range(k)) if p in state.mod_pred_index else tuple(range(k + 1))
            )
        all_ts = list(product(*ranges))
    else:
        all_ts = list(product(*(
            [tuple(range(2)) for _ in state.ext_preds]
            + [tuple(range(k + 1)) for k in state.cnt_params]
        )))

    for i in range(n_cells):
        for j in range(n_cells):
            for t1 in all_ts:
                for t2 in all_ts:
                    for (dt, reverse_dt), (rijt, two_tables) in rs[(i, j)].items():
                        t1_new = [x - y for x, y in zip(t1, dt)]
                        t2_new = [x - y for x, y in zip(t2, reverse_dt)]

                        if state.exist_mod:
                            for p, k_i in enumerate(state.cnt_params):
                                slot = n_ext + p
                                if p in state.mod_pred_index:
                                    t1_new[slot] %= k_i
                                    t2_new[slot] %= k_i

                        if any(
                            t1_new[n_ext + p] < 0 or t2_new[n_ext + p] < 0
                            for p in range(n_cnt)
                        ):
                            continue

                        for slot in range(n_ext):
                            t1_new[slot] = max(t1_new[slot], 0)
                            t2_new[slot] = max(t2_new[slot], 0)

                        c1 = (i,) + t1
                        c2 = (j,) + t2
                        c1_new = (i,) + tuple(t1_new)
                        c2_new = (j,) + tuple(t2_new)
                        t_update_dict[(c1, c2)][(c1_new, c2_new)] += rijt
                        t_sample[(c1, c2)][(c1_new, c2_new)].append((two_tables, rijt))

    return t_update_dict, t_sample


def _stop_condition(target_c: State, state: CountingState) -> bool:
    """Check whether the target element's state satisfies all counting constraints."""
    pred_state = target_c[1:]
    if state.exist_le:
        for i in range(len(pred_state)):
            if i not in state.le_index and pred_state[i] != 0:
                return False
    else:
        return all(s == 0 for s in pred_state)
    
# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------

def _make_domain_recursion(
    t_update_dict,
    space: ConfigSpace,
    cs: CountingState,
    has_linear_order: bool,
    data_T: dict,
    data_H: dict,
) -> Callable[[Config], RingElement]:
    """Return a memoised domain_recursion function scoped to one cell graph."""

    updater = ConfigUpdater(t_update_dict, space, data_H)
    cache: dict[Config, RingElement] = {}

    def domain_recursion(config: Config):
        cached = cache.get(config)
        if cached is not None:
            return cached

        if sum(config) == 0:
            return Rational(1, 1)

        result = Rational(0, 1)
        node = SamplingDrNode()
        
        nonzero_states = space.nonzero_states(config)
        target_c_list = nonzero_states if has_linear_order else (nonzero_states[-1],)

        for target_c in target_c_list:
            T = defaultdict(lambda: Rational(0, 1))
            config_new = space.dec(config, target_c)

            G = {(target_c, space.zero): Rational(1, 1)}
            dp_layers = []
            other_cs = space.nonzero_states(config_new)
            node.dp_order[target_c] = other_cs

            for other_c in other_cs:
                G_new = defaultdict(lambda: Rational(0, 1))
                l = space.count(config_new, other_c)
                G_layer = defaultdict(list)
                

                for (tc, G_config), W in G.items():
                    for (tc_new, H_config_new), weight_H in updater.f(tc, other_c, l).items():
                        G_config_new = space.add(G_config, H_config_new)

                        if has_linear_order:
                            denom = 1
                            for count in H_config_new:
                                if count > 1:
                                    denom *= math.factorial(count)
                            weight_H = weight_H * Rational(1, math.factorial(l) // denom)

                        G_new[(tc_new, G_config_new)] += W * weight_H
                        G_layer[(tc_new, G_config_new)].append(((tc, G_config), (W, weight_H)))

                G = G_new
                dp_layers.append(G_layer)

            node.dp_traceback[target_c] = dp_layers
            node.dp_starter[target_c] = defaultdict(list)

            for (l_target_c, G_config), W in G.items():
                if _stop_condition(l_target_c, cs):
                    T[G_config] += W
                    node.dp_starter[target_c][G_config].append((l_target_c, W))

            result_of_target_c = Rational(0, 1)
            for T_config, weight in T.items():
                W = domain_recursion(T_config)
                result_of_target_c += weight * W
                node.dp_recursion.append(((target_c, T_config), (W, weight)))

            result += result_of_target_c

        data_T[config] = node
        cache[config] = result
        return result

    return domain_recursion


def incremental_wfomc32(context: IncrementalWFOMC3Context) -> tuple[RingElement, list[CellGraphDpTrace]]:
    domain: set[Const] = context.domain
    formula = context.formula
    get_weight = context.get_weight
    leq_pred: Pred = context.leq_pred
    cs = context.counting_state
    has_lo = context.contain_linear_order_axiom()

    WFOMC_result = Rational(0, 1)
    domain_size = len(domain)
    MultinomialCoefficients.setup(domain_size)
    all_sample_data = []
    for cell_graph, graph_weight in build_cell_graphs(formula, get_weight, leq_pred):
        cells = cell_graph.get_cells()
        n_cells = len(cells)

        w2t, w, rs = build_sample_weight(cells, cell_graph, cs)
        logger.debug("Weight mapping w2t: {}", w2t)
        logger.debug("Weight w: {}", w)

        unary_mask = context.unary_handler.build_mask(cells)
        unary_evidence_factor = make_unary_evidence_factor(cells, context.factorized_unary_evidence)

        t_update_dict, data_Evi = build_sample_t_update_dict(rs, n_cells, cs)
        space = ConfigSpace((n_cells,) + tuple(cs.c_type_shape))
        data_T = {}
        data_H = {}
        domain_recursion = _make_domain_recursion(t_update_dict, space, cs, has_lo, 
                                                  data_T, data_H)
        data_root = []

        for config in multinomial(n_cells, domain_size):
            logger.debug("Config: {}", config)
            if any(context.unary_handler.check(config, unary_mask)):
                continue

            evidence_factor = unary_evidence_factor(config)
            if evidence_factor == Rational(0, 1):
                continue

            init_list = list(space.zero)
            W = Rational(1, 1)
            for i, n in enumerate(config):
                init_state = (i,) + w2t[i]
                init_list[space.offset(init_state)] = n
                W = W * (w[i] ** n)

            init_config = tuple(init_list)
            result_config = domain_recursion(init_config)

            if has_lo:
                term_weight = W * result_config * graph_weight * evidence_factor
            else:
                term_weight = MultinomialCoefficients.coef(config) * W * result_config * graph_weight * evidence_factor

            WFOMC_result += term_weight
            data_root.append((init_config, term_weight))

        all_sample_data.append(
            CellGraphDpTrace(
                space=space,
                cells=cells,
                cell_weights=w,
                data_root=data_root,
                data_T=data_T,
                data_H=data_H,
                data_Evi=data_Evi,
                one_type_offsets=space.cell_ext_one_offsets(len(context._ext_preds)),
            )
        )

    return expand(WFOMC_result), all_sample_data


class AliasTable:
    """
    Vose's Alias Method for O(1) sampling from a discrete distribution.
    Maintains exact arithmetic during O(n) construction using Rational types.
    """

    def __init__(self, choices, weights):
        if choices is None or weights is None or len(choices) != len(weights) or not choices:
            raise ValueError("Choices and weights must be non-empty lists of the same length.")

        self.choices = choices
        self.n = len(choices)
        
        # 1. 直接对 Rational 对象求和，保持精确算术。提供 Rational(0,1) 确保类型安全
        total = sum(weights, Rational(0, 1))
        if total == 0:
            raise ValueError("Cannot sample from a zero-weight distribution.")

        self._single = choices[0] if self.n == 1 else None
        if self._single is not None:
            self.prob = None
            self.alias = None
            return

        # 2. 计算平均概率权重：w = weight * n / total (始终保持为 Rational 精确类型)
        w = [(weight * self.n) / total for weight in weights]
        
        # 使用 Rational 暂存概率数组，防止过程溢出
        prob = [Rational(0, 1)] * self.n
        alias = [0] * self.n
        small, large = [], []

        # 3. p < 1 的判断对于 fmpq (Rational) 是完美支持的
        for i, p in enumerate(w):
            if p < 1:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            l = small.pop()
            g = large.pop()
            prob[l] = w[l]
            alias[l] = g
            
            # 4. 精确的有理数加减法
            w[g] = (w[g] + w[l]) - 1
            
            if w[g] < 1:
                small.append(g)
            else:
                large.append(g)

        while large:
            prob[large.pop()] = Rational(1, 1)
        while small:
            prob[small.pop()] = Rational(1, 1)

        self.prob = np.array([self._safe_fraction_to_float(p) for p in prob], dtype=np.float64)
        self.alias = np.array(alias, dtype=int)


    # 5. 终极安全转换：对依然过于庞大的分子分母进行按位右移，强行压入 float 的安全范围内

    @staticmethod
    def _safe_fraction_to_float(r) -> float:
        if isinstance(r, (int, float)):
            return float(r)

        # 提取分子和分母的 Python 大整数
        num = int(r.numer() if hasattr(r, "numer") else r.numerator)
        den = int(r.denom() if hasattr(r, "denom") else r.denominator)

        if num == 0:
            return 0.0

        # CPython 的 float 最大只能容纳约 1024 位的二进制数
        # 如果分母超过 1000 位，则将它们同时右移（相当于同时除以 2^shift），避免 OverflowError
        bit_length = den.bit_length()
        if bit_length > 1000:
            shift = bit_length - 1000
            num >>= shift
            den >>= shift

        return float(num) / float(den)

    def sample(self):
        """Returns a sampled choice in O(1)."""
        if self._single is not None:
            return self._single

        i = np.random.randint(self.n)
        idx = i if np.random.rand() < self.prob[i] else self.alias[i]
        return self.choices[idx]


def incremental_wfoms32(
    context: IncrementalWFOMC3Context,
    all_sample_data: list[CellGraphDpTrace],
    sample_times: int,
):
    cache_split_poly = {}
    cache_poly = {}
    cache_root = {}
    coeff_cache = {}
    has_cardinality = (
        context.contain_cardinality_constraint()
        and not context.cardinality_constraint.empty()
    )
    gen_vars = context.cardinality_constraint.gen_vars if has_cardinality else []
    zero_degree = tuple(0 for _ in gen_vars)

    def get_coeffs(poly):
        if not has_cardinality:
            return {(): poly}

        cache_key = id(poly)
        cached = coeff_cache.get(cache_key)
        if cached is not None and cached[0] is poly:
            return cached[1]

        if (
            isinstance(poly, Rational)
            or isinstance(poly, (int, float))
            or (hasattr(poly, "is_Number") and poly.is_Number)
        ):
            coeffs = {zero_degree: poly}
        else:
            coeffs = dict(coeff_dict(expand(poly), gen_vars))

        coeff_cache[cache_key] = (poly, coeffs)
        return coeffs

    def remove_1type_budget(global_target_degree, nowK: Config, graph_data: CellGraphDpTrace):
        if not global_target_degree:
            return ()

        W_1type = Rational(1, 1)
        for cell_idx, offsets in enumerate(graph_data.one_type_offsets):
            n = sum(nowK[offset] for offset in offsets)
            if n > 0:
                W_1type = W_1type * (graph_data.cell_weights[cell_idx] ** n)

        d_1type_dict = get_coeffs(W_1type)
        d_1type = next(iter(d_1type_dict), ())
        return tuple(a - b for a, b in zip(global_target_degree, d_1type))

    # 多项式乘积拆解采样器 在 (choices, weight_tuples) 中采样 (choice, dA, dB)，权重为多项式AB乘积中target_degree项的系数，一个choice可能有多个合法的(dA, dB)拆分组合
    def sample_split_poly(pairs, target_degree):
        if not pairs:
            raise ValueError("No choices available for split sampling.")

        cache_key = (id(pairs), target_degree)
        sampler = cache_split_poly.get(cache_key)
        if sampler is None:
            valid_choices = []
            valid_weights = []
            for choice, (poly_A, poly_B) in pairs:
                coeffs_A = get_coeffs(poly_A)
                coeffs_B = get_coeffs(poly_B)
                for d_A, w_A in coeffs_A.items():
                    for d_B, w_B in coeffs_B.items():
                        d_sum = tuple(a + b for a, b in zip(d_A, d_B)) if d_A else ()
                        if d_sum == target_degree:
                            prod = w_A * w_B
                            if prod != 0:
                                valid_choices.append((choice, d_A, d_B))
                                valid_weights.append(prod)
            sampler = AliasTable(valid_choices, valid_weights)
            cache_split_poly[cache_key] = sampler

        return sampler.sample()

    #在pairs即(choice, poly)中采样 (choice), 权重为其 poly 中 target_degree 项的系数，没有则权重为0，返回被采样的 choice
    def sample_poly(pairs, target_degree):
        if not pairs:
            raise ValueError("No choices available for sampling.")

        cache_key = (id(pairs), target_degree)
        sampler = cache_poly.get(cache_key)
        if sampler is None:
            valid_choices = []
            valid_weights = []
            for choice, poly in pairs:
                coeffs = get_coeffs(poly)
                weight = coeffs.get(target_degree)
                if weight is not None and weight != 0:
                    valid_choices.append(choice)
                    valid_weights.append(weight)
            sampler = AliasTable(valid_choices, valid_weights)
            cache_poly[cache_key] = sampler

        return sampler.sample()

    # 根节点采样器
    def sample_poly_root(data_root):
        cache_key = id(data_root)
        sampler = cache_root.get(cache_key)
        if sampler is None:
            valid_choices = []
            valid_weights = []
            for choice, poly in data_root:
                for degree, weight in get_coeffs(poly).items():
                    if weight == 0:
                        continue
                    if not has_cardinality or context.cardinality_constraint.valid(list(degree)):
                        valid_choices.append((choice, degree))
                        valid_weights.append(weight)
            sampler = AliasTable(valid_choices, valid_weights)
            cache_root[cache_key] = sampler

        return sampler.sample()

    all_sample_results = []
    for sample_idx in range(sample_times):
        if sample_idx % 100 == 0:
            logger.debug("Sample {}/{} complete", sample_idx, sample_times)

        one_sample_result = []
        for graph_data in all_sample_data:
            space = graph_data.space
            nowK, global_budget = sample_poly_root(graph_data.data_root)
            domain_size = sum(nowK)
            current_target_degree = remove_1type_budget(global_budget, nowK, graph_data)
            sampled_1type = np.empty(domain_size, dtype=object)
            sampled_2table_matrix = np.empty((domain_size, domain_size), dtype=object)
            sampled_2table_matrix[:] = None

            for n in range(domain_size, 0, -1):
                node: SamplingDrNode = graph_data.data_T[nowK]
                (target_c, nextK), nextK_target_degree, G_target_degree = (
                    sample_split_poly(node.dp_recursion, current_target_degree)
                )
                current_target_degree = nextK_target_degree
                other_cs = node.dp_order[target_c]

                sampled_1type[n - 1] = graph_data.cells[target_c[0]]
                if n == 1:
                    break

                tgc_pairs = node.dp_starter[target_c].get(nextK)
                nowc = sample_poly(tgc_pairs, G_target_degree)
                nowcfg = nextK

                # 暂不支持线性序！ 外层循环每次选取 target_c 时，总是选择 np.argwhere(...) [-1]
                # 因此可以预判剩余的 n-1 个元素分别会被赋予什么状态，并为每个状态保留正确的索引池。
                state_to_indices = {}
                curr_idx = n - 2
                for state in reversed(space.nonzero_states(nextK)):
                    count = space.count(nextK, state)
                    state_to_indices[state] = deque(range(curr_idx, curr_idx - count, -1))
                    curr_idx -= count

                for i in range(len(other_cs) - 1, -1, -1):
                    G_layer = node.dp_traceback[target_c][i]
                    other_c = other_cs[i]

                    l = space.count(nowK, other_c)
                    if other_c == target_c:
                        l -= 1

                    G_pairs = G_layer[(nowc, nowcfg)]
                    (lastc, lastcfg), d_WG, H_target_degree = sample_split_poly(
                        G_pairs, G_target_degree
                    )
                    G_target_degree = d_WG

                    tc_new = nowc
                    hc_new = space.sub(nowcfg, lastcfg)

                    for j in range(l, 0, -1):
                        H_layer = graph_data.data_H[(lastc, other_c)][j]
                        (tc_old, oc_new), d_WH, d_rij = sample_split_poly(
                            H_layer[(tc_new, hc_new)], H_target_degree
                        )
                        H_target_degree = d_WH

                        all_two_tables = graph_data.data_Evi[(tc_old, other_c)][
                            (tc_new, oc_new)
                        ]
                        two_tables = sample_poly(all_two_tables, d_rij)
                        sampled_2table = sample_poly(two_tables, d_rij)

                        assigned_m = state_to_indices[oc_new].popleft()
                        sampled_2table_matrix[n - 1][assigned_m] = sampled_2table

                        tc_new = tc_old
                        hc_new = space.dec(hc_new, oc_new)

                    nowc = lastc
                    nowcfg = lastcfg

                nowK = nextK

            # perm = np.random.permutation(domain_size)
            # sampled_1type = sampled_1type[perm]
            # sampled_2table_matrix = sampled_2table_matrix[perm][:, perm]
            one_sample_result.append((sampled_1type, sampled_2table_matrix))

        all_sample_results.append(one_sample_result)

    return all_sample_results
