import os
import sys
import argparse
from loguru import logger
from contexttimer import Timer

from wfomc.context import WFOMCContext, IncrementalWFOMC3Context
from wfomc.network import UnaryEvidenceEncoding
from wfomc.problems import WFOMCProblem
from wfomc.algo import (
    Algo,
    analyze_all_sample,
    incremental_wfomc3,
    incremental_wfomc32,
    incremental_wfoms3,
    incremental_wfoms32,
)
from wfomc.utils import MultinomialCoefficients, Rational, round_rational
from wfomc.parser import parse_input

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

def wfomc(problem: WFOMCProblem, algo: Algo = Algo.INCREMENTAL3,
          unary_evidence_encoding: UnaryEvidenceEncoding = UnaryEvidenceEncoding.CCS,
          sample_time: int = 1,
          debug: bool = False) -> Rational:
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
            logger.info(f'Unary evidence is found, using {unary_evidence_encoding} encoding')
        if problem.sentence.contain_modulo_counting_quantifier():
            logger.info('Modulo counting quantifier is found')
        logger.info(f'Invoke WFOMC with {algo} algorithm and {unary_evidence_encoding} encoding')

        context = IncrementalWFOMC3Context(problem)
        if algo == Algo.INCREMENTAL3:
            wfomc_fn = incremental_wfomc3
            wfoms_fn = incremental_wfoms3
            analyze_samples = True
        elif algo == Algo.INCREMENTAL32:
            wfomc_fn = incremental_wfomc32
            wfoms_fn = incremental_wfoms32
            analyze_samples = False
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")
    
        with Timer() as t:
            res, all_sample_data = wfomc_fn(context)
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
            all_sample_results = wfoms_fn(context, all_sample_data, sample_time)
        logger.info('Sampling time: {}', t.elapsed)
        
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
    parser.add_argument('--unary_evidence_encoding', '-e', type=UnaryEvidenceEncoding,
                        choices=list(UnaryEvidenceEncoding),
                        default=UnaryEvidenceEncoding.CCS)
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

    # Remove the default loguru stderr handler; parse_input() and wfomc()
    # each add their own scoped handler.
    try:
        logger.remove(0)
    except ValueError:
        pass

    # File sink covers the entire session. It receives records whenever
    # parse_input() or wfomc() call logger.enable("wfomc").
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
        unary_evidence_encoding=args.unary_evidence_encoding,
        sample_time=args.sample_time,
        debug=args.debug,
    )

    print(f'WFOMC (arbitrary precision): {res}')
    round_val = round_rational(res)
    print(f'WFOMC (round): {round_val} (exp({round_val.ln()}))')
