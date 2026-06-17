import argparse
import os
import sys
from typing import Optional, Union

from contexttimer import Timer
from loguru import logger

from wfomc.algo import (
    Algo,
    incremental_wfomc3,
    incremental_wfoms3,
    analyze_all_sample,
)
from wfomc.context import (
    IncrementalWFOMC3Context,
    UnaryEvidenceStrategy,
    WFOMCContext,
)
from wfomc.fol import Counting, QuantifiedFormula
from wfomc.parser import parse_input
from wfomc.problems import WFOMCProblem
from wfomc.result import WFOMCResult
from wfomc.utils import MultinomialCoefficients, Rational, round_rational

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def _counting_formula_kind_and_comparator(formula: QuantifiedFormula) -> tuple[str, str | None]:
    if isinstance(formula.quantified_formula, QuantifiedFormula):
        scope = formula.quantified_formula.quantifier_scope
        kind = "binary"
    else:
        scope = formula.quantifier_scope
        kind = "unary"
    comparator = scope.comparator if isinstance(scope, Counting) else None
    return kind, comparator


def _validate_counting_quantifiers(problem: WFOMCProblem, algo: Algo) -> None:
    if algo == Algo.INCREMENTAL3:
        supported = {
            "unary": {"=", "<=", "mod"},
            "binary": {"=", "<=", "mod"},
        }
    else:
        supported = {
            "unary": {"=", "!=", "<", ">", "<=", ">="},
            "binary": {"="},
        }

    for formula in problem.sentence.cnt_formulas:
        kind, comparator = _counting_formula_kind_and_comparator(formula)
        if comparator is None or comparator in supported[kind]:
            continue
        allowed = ", ".join(sorted(supported[kind]))
        raise RuntimeError(
            f"{kind.capitalize()} counting comparator '{comparator}' is not "
            f"supported by the {algo} algorithm. Supported comparators: {allowed}."
        )



def wfomc(problem: WFOMCProblem, algo: Algo = Algo.INCREMENTAL3,
          unary_evidence_strategy: UnaryEvidenceStrategy = UnaryEvidenceStrategy.AUTO,
          sample_time: int = 1,
          debug: bool = False) -> WFOMCResult:
    level = "DEBUG" if debug else "INFO"
    _handler_id = logger.add(
        sys.stderr, level=level, filter="wfomc", colorize=True, format=_LOG_FORMAT,
    )
    logger.enable("wfomc")
    try:
        MultinomialCoefficients.setup(len(problem.domain))

        if problem.contain_linear_order_axiom():
            logger.info('Linear order axiom with the predicate LEQ is found')  
            raise RuntimeError("LEQ not supported") 
        if problem.contain_predecessor_axiom():
            logger.info('Predecessor predicate PRED is found')
            raise RuntimeError("PRED not supported") 

        if problem.contain_unary_evidence():
            logger.info('Unary evidence is found, strategy: {}',unary_evidence_strategy)

        _validate_counting_quantifiers(problem, algo)

        if problem.sentence.contain_modulo_counting_quantifier():
            logger.info('Modulo counting quantifier is found')

        if algo != Algo.INCREMENTAL3:
            raise RuntimeError("illegal algo") 
        
        context = IncrementalWFOMC3Context( problem, unary_evidence_strategy )
        with Timer() as t:
            res, all_sample_data = incremental_wfomc3(context)
            res = context.decode_result(res)
        logger.info('WFOMC time: {}', t.elapsed)

        logger.info('WFOMC value: {}', res)
        round_val = round_rational(res)
        logger.info('WFOMC (round): {} (exp({}))', round_val, round_val.ln())

        if sample_time <= 0:
            logger.info('No sampling since sample_time is {} ', sample_time)
            return res
        
        if res == Rational(0):
            logger.info('No sampling since WFOMC value is 0')
            return res
        
        logger.info('Start sampling for {} times', sample_time)
        with Timer() as t:
            all_sample_results = incremental_wfoms3(context, all_sample_data, sample_time)
        logger.info('Sampling time: {}', t.elapsed)
        
        analyze_samples = True if sample_time <= 50 else False
        if analyze_samples:
            analyze_all_sample(all_sample_results)

        return res
    
    finally:
        logger.remove(_handler_id)
        logger.disable("wfomc")


def parse_args():
    parser = argparse.ArgumentParser(
        description='WFOMC for MLN',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='mln file')
    parser.add_argument('--output_dir', '-o', type=str,
                        default='./check-points')
    parser.add_argument('--algo', '-a', type=Algo,
                        choices=list(Algo), default=Algo.INCREMENTAL3)
    parser.add_argument('--unary_evidence_strategy', '-e', type=UnaryEvidenceStrategy,
                        choices=list(UnaryEvidenceStrategy),
                        default=UnaryEvidenceStrategy.AUTO)
   
    parser.add_argument('--sample-time', '-s', type=int, 
                        help='sample time', default = 0) 
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    level = "DEBUG" if args.debug else "INFO"

    try:
        logger.remove(0)
    except ValueError:
        pass

    logger.add(
        f'{args.output_dir}/log.txt',
        mode='w',
        level=level,
        filter="wfomc",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} - {message}",
    )

    with Timer() as t:
        problem = parse_input(args.input, debug=args.debug)
    print(f'Parse input: {t.elapsed:.4f}s')

    res = wfomc(
        problem, algo=args.algo,
        unary_evidence_strategy=args.unary_evidence_strategy,
        sample_time=args.sample_time,
        debug=args.debug,
    )
    res=WFOMCResult(res)

    print(f'WFOMC (arbitrary precision): {res}')
    const_res = res.constant_value()
    if const_res is not None:
        round_val = round_rational(const_res)
        print(f'WFOMC (round): {round_val} (exp({round_val.ln()}))')
