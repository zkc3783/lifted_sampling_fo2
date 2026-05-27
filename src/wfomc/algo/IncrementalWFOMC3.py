from __future__ import annotations
import math
from collections import defaultdict, Counter
from itertools import product
from typing import Callable

import numpy as np
from loguru import logger

from wfomc.cell_graph import build_cell_graphs
from wfomc.context import IncrementalWFOMC3Context, CountingState
from wfomc.fol import Const, Pred
from wfomc.utils import Rational, coeff_dict, multinomial, MultinomialCoefficients, Rational, expand, RingElement

import matplotlib.pyplot as plt 
from wfomc.fol.syntax import AtomicFormula, Const, Pred, X, a, b
from wfomc.cell_graph.components import Cell
from typing import FrozenSet
from flint import fmpq
# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

class HashableArrayWrapper:
    """Wraps a NumPy array to make it hashable (for use as a dict key)."""

    def __init__(self, input_array: np.ndarray):
        array = np.array(input_array, dtype=np.uint8, copy=True, order="C")
        array.setflags(write=False)
        self.array = array
        self._key = (array.shape, array.tobytes())
        self._hash = hash(self._key)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, HashableArrayWrapper):
            return self._key == other._key
        return False

    def __repr__(self):
        return f"HashableArrayWrapper({self.array})"


class ConfigUpdater:
    """
    Memoised configuration updater for efficient state transitions.

    _cache structure: {(target_c, other_c): {j: H_dict}}
    where H_dict maps (tc_new, H_config_new) → Rational weight,
    recording the cumulative weight of pairing target_c with j other_c elements.
    """

    def __init__(self, t_update_dict, c1_type_shape, data_H):
        self.t_update_dict = t_update_dict
        self.c1_type_shape = c1_type_shape
        self._cache: dict = {}
        self.data_H = data_H

    def f(self, target_c, other_c, l):
        """Return the weighted outcome of pairing target_c with l other_c elements."""
        if (target_c, other_c) in self._cache:
            sub = self._cache[(target_c, other_c)]
            num_start = l
            while num_start not in sub and num_start > 0:
                num_start -= 1
        else:
            self._cache[(target_c, other_c)] = {}
            self.data_H[(target_c, other_c)] = {}
            num_start = 0

        if num_start == 0:
            H_config = HashableArrayWrapper(np.zeros(self.c1_type_shape, dtype=np.uint8))
            H = {(target_c, H_config): Rational(1, 1)}
        else:
            H = self._cache[(target_c, other_c)][num_start]

        for j in range(num_start + 1, l + 1):
            H_new = defaultdict(lambda: Rational(0, 1))
            H_layer = defaultdict(list)
            for (tc_old, hc_old), W in H.items():
                for (tc_new, oc_new), rij in self.t_update_dict[(tc_old, other_c)].items():
                    hc_new_array = np.array(hc_old.array, copy=True)
                    hc_new_array[oc_new] += 1
                    hc_new = HashableArrayWrapper(hc_new_array)
                    H_new[(tc_new, hc_new)] += W * rij
                    H_layer[(tc_new, hc_new)].append(((tc_old, oc_new),(W, rij)))

            H = H_new
            self._cache[(target_c, other_c)][j] = H
            self.data_H[(target_c, other_c)][j] = H_layer

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
                two_table_weight = sum((w for atoms, w in two_tables), Rational(0, 1))
                #two_table_weight = cell_graph.get_two_table_weight((cells[i], cells[j]), evidence)
                if two_table_weight == Rational(0, 1):
                    continue
                t_fwd, t_rev = [], []
                for pred_idx, pred in enumerate(state.ext_preds + state.cnt_preds):
                    t_rev.append(1 if (evi_idx >> (2 * pred_idx)) & 1 else 0)
                    t_fwd.append(1 if (evi_idx >> (2 * pred_idx + 1)) & 1 else 0)
                rs[(i, j)][(tuple(t_fwd), tuple(t_rev))] = (two_table_weight, two_tables)

    return w2t, w, rs


def build_sample_t_update_dict(rs, n_cells: int, state: CountingState) -> defaultdict:
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


def _stop_condition(target_c, state: CountingState):
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
    t_update_dict, c1_type_shape: tuple, cs: CountingState, has_linear_order: bool
    , data_T: dict, data_H: dict) -> Callable:
    """Return a memoised domain_recursion function scoped to one cell graph."""

    updater = ConfigUpdater(t_update_dict, c1_type_shape, data_H)
    f = updater.f
    cache: dict = {}

    def domain_recursion(config):
        if config in cache:
            return cache[config]

        if config.array.sum() == 0:
            return Rational(1, 1)

        result = Rational(0, 1)
        node = SamplingDrNode()

        if has_linear_order:
            target_c_list = [tuple(i) for i in np.argwhere(config.array > 0)]
        else:
            target_c_list = [tuple(np.argwhere(config.array > 0)[-1])]

        for target_c in target_c_list:
            T = defaultdict(lambda: Rational(0, 1))
            config_new_array = np.array(config.array, copy=True, dtype=np.uint8)
            config_new_array[target_c] -= 1
            config_new = HashableArrayWrapper(config_new_array)

            G = {(target_c, HashableArrayWrapper(np.zeros(c1_type_shape, dtype=np.uint8))): Rational(1, 1)}
            dp_layers = []
            other_cs = [tuple(x.flatten()) for x in np.argwhere(config_new.array > 0)]
            node.dp_order[target_c] = other_cs
            for other_c in other_cs: 

                G_new = defaultdict(lambda: Rational(0, 1))
                l = config_new.array[other_c]
                G_layer = defaultdict(list)

                for (tc, G_config), W in G.items():
                    for (tc_new, H_config_new), weight_H in f(tc, other_c, l).items():
                        G_config_new = HashableArrayWrapper(G_config.array + H_config_new.array)

                        if has_linear_order:
                            denom = 1
                            for count in H_config_new.array.flatten():
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

def incremental_wfomc3(context: IncrementalWFOMC3Context) -> RingElement:
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
        t_update_dict, data_Evi = build_sample_t_update_dict(rs, n_cells, cs)
        c1_type_shape = (n_cells,) + tuple(cs.c_type_shape)
        data_T = {} 
        data_H = {} 
        domain_recursion = _make_domain_recursion(t_update_dict, c1_type_shape, cs, has_lo,
                                                   data_T, data_H)
        data_root= []

        for config in multinomial(n_cells, domain_size):
            logger.debug("Config: {}", config)
            if any(context.unary_handler.check(config, unary_mask)):
                continue

            init_config = np.zeros(c1_type_shape, dtype=np.uint8)
            W = Rational(1, 1)
            for i, n in enumerate(config):
                init_config[(i,) + w2t[i]] = n
                W = W * (w[i] ** n)

            init_config=HashableArrayWrapper(init_config)
            result_config = domain_recursion(init_config)

            if has_lo:
                term_weight = W * result_config * graph_weight
            else:
                term_weight = MultinomialCoefficients.coef(config) * W * result_config * graph_weight

            WFOMC_result += term_weight
            data_root.append((init_config, term_weight))



        all_sample_data.append((cells, w, data_root, data_T, data_H, data_Evi))

    return expand(WFOMC_result), all_sample_data

class SamplingDrNode:
    def __init__(self):
        self.dp_recursion = []
        # (nk_choices, next_step_weights)
        # choices: (target_c, K')
        self.dp_starter = {} 
        # key = K' (sampled in 2.1)
        # value = (lastc, weight) 
        self.dp_traceback = {} 
        # key = target_c
        # value = dp_layers
        # dp_layers: List[G_layer]
        # G_layer:  dict: state -> (prev_states, contributions)
        # state: (curr_target_c, curr_u_config), state = (last_target_c, K') when begin
        self.dp_order = {} 
        # key = target_c
        # value = list of other_c

class AliasTable:
    """
    Vose's Alias Method for O(1) sampling from a discrete distribution.
    Maintains exact arithmetic during O(n) construction using Rational types.
    """
    def __init__(self, choi, wgt):
        if choi is None or wgt is None or len(choi) != len(wgt) or len(choi) == 0:
            raise ValueError(f"Choices and weights must be non-empty lists of the same length.")
        
        self.choices = choi
        self.n = len(self.choices)
        
        # 1. 直接对 Rational 对象求和，保持精确算术。提供 Rational(0,1) 确保类型安全
        total = sum(wgt, Rational(0, 1)) 
        
        if total == 0:
            self.prob = np.zeros(self.n, dtype=np.float64)
            self.alias = np.zeros(self.n, dtype=int)
            return   
            
        # 2. 计算平均概率权重：w = weight * n / total (始终保持为 Rational 精确类型)
        w = [(val * self.n) / total for val in wgt]
        
        # 使用 Rational 暂存概率数组，防止过程溢出
        prob = [Rational(0, 1)] * self.n 
        alias = [0] * self.n
        small = []
        large = []
        
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

        # 5. 终极安全转换：对依然过于庞大的分子分母进行按位右移，强行压入 float 的安全范围内
        def safe_fraction_to_float(r):
            if isinstance(r, (int, float)):
                return float(r)
            
            # 提取分子和分母的 Python 大整数
            num = int(r.numer() if hasattr(r, 'numer') else r.numerator)
            den = int(r.denom() if hasattr(r, 'denom') else r.denominator)
            
            if num == 0:
                return 0.0
                
            # CPython 的 float 最大只能容纳约 1024 位的二进制数
            # 如果分母超过 1000 位，则将它们同时右移（相当于同时除以 2^shift），避免 OverflowError
            bl = den.bit_length()
            if bl > 1000:
                shift = bl - 1000
                num >>= shift
                den >>= shift
                
            return float(num) / float(den)

        self.prob = np.array([safe_fraction_to_float(p) for p in prob], dtype=np.float64)
        self.alias = np.array(alias, dtype=int)

    def sample(self):
        """Returns a sampled choice in O(1)."""
        i = np.random.randint(self.n)
        if np.random.rand() < self.prob[i]:
            idx = i
        else:
            idx = self.alias[i]
        return self.choices[idx]
    
def incremental_wfoms3(context: IncrementalWFOMC3Context, all_sample_data: tuple, sample_times: int):
    
    cache_split_poly = {}
    cache_poly = {}
    cache_root = {}
    # 工具函数：提取多项式中特定阶数和对应的系数
    def get_coeffs(poly):
        if not context.contain_cardinality_constraint() or context.cardinality_constraint.empty():
            return {(): poly}
        gen_vars = context.cardinality_constraint.gen_vars
        if isinstance(poly, Rational) or isinstance(poly, (int, float)) or (hasattr(poly, 'is_Number') and poly.is_Number):
            return {tuple([0]*len(gen_vars)): poly}
        poly_exp = expand(poly)
        return dict(coeff_dict(poly_exp, gen_vars))
    
    def remove_1type_budget(global_target_degree, nowK, cell_weights):
        num_ext = len(context._ext_preds) 
        slicer = tuple([slice(None)] + [1] * num_ext + [Ellipsis])
        valid_slice = nowK.array[slicer] #extpred维度必全为1，切出来ctype扣除预算
        W_1type = Rational(1, 1) 
        for i in range(len(valid_slice)):
            n = int(valid_slice[i].sum())
            if n > 0:
                W_1type = W_1type * (cell_weights[i] ** n)
        d_1type_dict = get_coeffs(W_1type)
        d_1type = list(d_1type_dict.keys())[0] if d_1type_dict else ()
        current_target_degree = tuple(a - b for a, b in zip(global_target_degree, d_1type)) if global_target_degree else ()
        return current_target_degree
    
    # 多项式乘积拆解采样器 在 (choices, weight_tuples) 中采样 (choice, dA, dB)，权重为多项式AB乘积中target_degree项的系数，一个choice可能有多个合法的(dA, dB)拆分组合
    def sample_split_poly(pairs, target_degree):
        if not pairs:
            raise ValueError("No choices available for split sampling.")
        cid = id(pairs)
        cache_key = (cid, target_degree)
        if cache_key not in cache_split_poly:
            valid_choices = []
            valid_weights = []
            for choice, (poly_A, poly_B) in pairs:   
                coeffs_A = get_coeffs(poly_A)
                coeffs_B = get_coeffs(poly_B)
                # 尝试所有合法的拆分 d_A + d_B = target_degree
                for d_A, w_A in coeffs_A.items():
                    for d_B, w_B in coeffs_B.items():
                        d_sum = tuple(a + b for a, b in zip(d_A, d_B)) if d_A else ()
                        if d_sum == target_degree:
                            prod = w_A * w_B
                            if prod != 0:
                                valid_choices.append((choice, d_A, d_B))
                                valid_weights.append(prod)
            # 计算完后直接将构建好的 AliasTable 存入缓存
            cache_split_poly[cache_key] = AliasTable(valid_choices, valid_weights)
            
        return cache_split_poly[cache_key].sample()
    
    #在pairs即(choice, poly)中采样 (choice), 权重为其 poly 中 target_degree 项的系数，没有则权重为0，返回被采样的 choice
    def sample_poly(pairs, target_degree):
        if not pairs:
            raise ValueError("No choices available for sampling.")
        cid = id(pairs) # 这里的 pairs 必须是内存中的持久对象，不能是临时对象
        cache_key = (cid, target_degree)
        if cache_key not in cache_poly:
            valid_choices = []
            valid_weights = []
            for choice, poly in pairs:
                coeffs = get_coeffs(poly)
                if target_degree in coeffs and coeffs[target_degree] != 0:
                    valid_choices.append(choice)
                    valid_weights.append(coeffs[target_degree])
            cache_poly[cache_key] = AliasTable(valid_choices, valid_weights)
            
        return cache_poly[cache_key].sample()
       
    # 根节点采样器
    def sample_poly_root(data_root):
        cid = id(data_root)# 根节点没有外部传入的 target_degree，只有全局合法性约束，所以键仅需 cid
        cache_key = cid
        if cache_key not in cache_root:
            valid_root_choices = []
            valid_root_weights = []
            for choice, poly in data_root:
                coeffs = get_coeffs(poly)
                for d, w in coeffs.items():
                    #print(list(d), w)
                    if not context.contain_cardinality_constraint() or context.cardinality_constraint.valid(list(d)):
                        if w != 0:
                            #print(f"Valid root choice: {choice} with degree {d} and weight {w}")
                            valid_root_choices.append((choice, d))
                            valid_root_weights.append(w)
            cache_root[cache_key] = AliasTable(valid_root_choices, valid_root_weights)

        return cache_root[cache_key].sample()

    all_sample_results = []
    for sample_idx in range(sample_times):
        if sample_idx % 100 == 0:
            logger.info(f"Sample {sample_idx}/{sample_times} complete ")
        one_sample_result = []
        for cells, cell_weights, data_root, data_T, data_H, data_Evi in all_sample_data:

            (nowK, global_budget) = sample_poly_root(data_root) 
            domain_size = int(nowK.array.sum())
            current_target_degree = remove_1type_budget(global_budget, nowK, cell_weights)
            Sampled_1type = np.empty(domain_size, dtype=object)
            Sampled_2table_matrix = np.empty((domain_size, domain_size), dtype=object)
            
            for n in range(domain_size, 0, -1): 

                node: SamplingDrNode = data_T[nowK]
                nk_pairs = node.dp_recursion # [(target_c, nextK), (weight_tuple)]
               
                (target_c, nextK), nextK_target_degree, G_target_degree  = sample_split_poly(nk_pairs, current_target_degree)       
                current_target_degree = nextK_target_degree # 剩余的递归预算留给下个节点
                othercs = node.dp_order[target_c] 

                Sampled_1type[n-1] = cells[target_c[0]]
                if n == 1:  
                    break   

                tgc_pairs = node.dp_starter[target_c].get(nextK)
                nowc = sample_poly(tgc_pairs, G_target_degree)
                nowcfg = nextK 

                # 暂不支持线性序！ 外层循环每次选取 target_c 时，总是选择 np.argwhere(...) [-1]
                # 因此可以预判剩余的 n-1 个元素分别会被赋予什么状态，并为每个状态保留正确的索引池。
                available_states = [tuple(x) for x in np.argwhere(nextK.array > 0)]
                state_to_indices = {}
                curr_idx = n - 2
                for state in reversed(available_states): # 按外层循环选取次序分配从大到小的矩阵索引
                    count = int(nextK.array[state])
                    # 为该状态分配连续的 count 个索引
                    state_to_indices[state] = list(range(curr_idx, curr_idx - count, -1))
                    curr_idx -= count
                
        
                for i in range(len(othercs)-1, -1, -1):  
                    G_layer = node.dp_traceback[target_c][i]
                    other_c = othercs[i]

                    l = nowK.array[other_c]
                    if other_c == target_c:
                        l -= 1

                    G_pairs = G_layer[(nowc, nowcfg)] 
                    (lastc, lastcfg), d_WG, H_target_degree = sample_split_poly(G_pairs, G_target_degree)
                    G_target_degree = d_WG

                    tc_new = nowc 
                    hc_new = HashableArrayWrapper(nowcfg.array - lastcfg.array) 

                    for j in range(l, 0, -1): 
                        H_layer = data_H[(lastc, other_c)][j]
                        (tc_old, oc_new), d_WH, d_rij = sample_split_poly(H_layer[(tc_new, hc_new)], H_target_degree)
                        H_target_degree = d_WH 

                        all_two_tables = data_Evi[(tc_old, other_c)][(tc_new, oc_new)] 
                        two_tables = sample_poly(all_two_tables, d_rij)   
                        samp_2table = sample_poly(two_tables, d_rij)

                        # 不使用 m -= 1，而是根据 oc_new 获取准确的预分配索引 
                        assigned_m = state_to_indices[oc_new].pop(0)
                        Sampled_2table_matrix[n-1][assigned_m] = samp_2table

                        tc_new = tc_old
                        hc_new_array = np.array(hc_new.array, copy=True)
                        hc_new_array[oc_new] -= 1
                        hc_new = HashableArrayWrapper(hc_new_array)
                        
                    nowc = lastc
                    nowcfg = lastcfg
                    
                nowK = nextK

            perm = np.random.permutation(domain_size)
            Sampled_1type = Sampled_1type[perm]
            Sampled_2table_matrix = Sampled_2table_matrix[perm][:, perm]
            one_sample_result.append((Sampled_1type, Sampled_2table_matrix))
        all_sample_results.append(one_sample_result)

    return all_sample_results

def analyze_all_sample(all_sample_results):

   
    def make_hashable(obj):
        if isinstance(obj, dict):
            # 排序以保证字典顺序一致
            return tuple((k, make_hashable(obj[k])) for k in sorted(obj.keys()))
        elif isinstance(obj, list):
            return tuple(make_hashable(i) for i in obj)
        else:
            # 将 R1(X,X) 等逻辑对象转为字符串
            return str(obj)
        
    def format_cell(cell): # 去除难看的 list() 和中括号
        if cell is None:
            return "-"
        return ", ".join(str(item) for item in cell)
    
    def clean1type(cell: Cell) -> list[AtomicFormula]:
        evidences: set[AtomicFormula] = set()
        for i, p in enumerate(cell.preds):
            if p.name.startswith('@') or 'aux' in p.name.lower(): 
                continue
            atom = p(*([X] * p.arity))
            evidences.add(atom) if (cell.code[i]) else evidences.add(~atom)
        return list(evidences)
    
    def clean2table(two_table: FrozenSet[AtomicFormula] = None) -> tuple[list[AtomicFormula], list[AtomicFormula]]:
        tmp = list(
            atom for atom in two_table 
            if not (atom.pred.name.startswith('@') or 'skolem' in atom.pred.name.lower())
            and len(set(atom.args)) > 1
            and atom.positive == True
        )
        tableab=list(atom for atom in tmp if atom.args[0].name == 'a')
        tableba_normalized = []
        for atom in tmp:
            if atom.args[0].name == 'b':
                new_atom = AtomicFormula(atom.pred, (a, b), atom.positive)
                tableba_normalized.append(new_atom)
        return tableab, tableba_normalized
    
    def goodprint(print_1type, print_2table):
        domain_size = len(print_1type)
        # 1. 定义统一的列宽。如果谓词很长，可以把 15 改为 20。
        W = 20
        
        print("-" * (W * (domain_size + 1)))
        print("    Sampled 1-type:")
        for i in range(domain_size,0,-1):
            print(f"      e{i}: [{format_cell(print_1type[i-1])}]")

        print("\n    Sampled 2-table:")
        # 2. 打印表头 (Col 标注)
        # 第一列是空的，用来给行标签留位置
        header = f"{'':<{W}}" 
        for j in range(domain_size, 0, -1):
            col_label = f"e{j}_b"
            header += f"{col_label:<{W}}"
        print(header)
        # 3. 打印每一行 (Row 标注 + 矩阵内容)
        for i in range(domain_size, 0, -1):
            row_label = f"      e{i}_a:"
            # 行首标签也占 W 宽，左对齐
            row_str = f"{row_label:<{W}}"
            
            for j in range(domain_size, 0, -1):
                cell_val = format_cell(print_2table[i-1][j-1])
                # 每个单元格都占 W 宽，左对齐
                row_str += f"{cell_val:<{W}}"
            print(row_str)
        print("-" * (W * (domain_size + 1)))
    
    def cleansample(sample):
        Sampled_1type, Sampled_2table_matrix = sample
        domain_size = len(Sampled_1type) 
        print_1type = np.empty(domain_size, dtype=object)
        print_2table = np.empty((domain_size, domain_size), dtype=object)
        print_2table[:] = None

        for i in range(domain_size):
            print_1type[i] = clean1type(Sampled_1type[i])

        for i in range(domain_size): #这里因为shuffle了所以不是上三角矩阵了
            for j in range(domain_size):
                if i == j or ( Sampled_2table_matrix[i][j] is None):
                    continue
                print_2table[i][j], print_2table[j][i] = clean2table(Sampled_2table_matrix[i][j])
        
        return print_1type, print_2table


    samples = [res[0] for res in all_sample_results] # 假设采样只有一个graph
    total_samples = len(samples)

    print(f"Displaying all {total_samples} samples:")
    #return #
    for idx, sample in enumerate(samples):
        print(f"Sample {idx+1}:")
        goodprint(*cleansample(sample))
        break #

    return #
    # 1. 快速计数: 将对象转换为签名并利用 Counter 统计
    signatures= []
    revprint = {}
    for sample in samples:
        mh=make_hashable(sample)
        signatures.append(mh)
        if mh not in revprint:
            revprint[mh] = sample

    signatures = [make_hashable(s) for s in samples]
    counts = Counter(signatures)

    sorted_configs = counts.most_common()

    labels = [f"Config {i+1}" for i in range(len(sorted_configs))]
    frequencies = [count / total_samples for _, count in sorted_configs]

    print(f"{'ID':<10} | {'Count':<10} | {'Frequency':<10}")
    print("=" * 35)
    for i, (sig, count) in enumerate(sorted_configs):
        print(f"Config {i+1:<4} | {count:<10} | {count/total_samples:<10.4f}")
        goodprint(*cleansample(revprint[sig]))  # 打印对应签名的样本内容

    # 绘图部分
    plt.figure(figsize=(12, 6))
    bars = plt.bar(labels, frequencies, color='steelblue', alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2%}', ha='center', va='bottom', fontsize=9)

    plt.ylabel('Empirical Probability')
    plt.title(f'Sampling Distribution (N={total_samples})')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
