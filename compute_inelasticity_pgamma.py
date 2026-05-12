import argparse

from utils import (
    PGAMMA_INTERACTION,
    add_common_scan_arguments,
    run_scan_from_args,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute p+gamma leading-particle energy fractions from "
            "gamma+p fixed-target Chromo kinematics."
        ),
    )
    add_common_scan_arguments(
        parser,
        default_model=("Pythia8",),
        default_emin=1e2,
        default_emax=1e7,
        default_stat_help="Particle statistic to average. Defaults to leading.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run_scan_from_args(args, [PGAMMA_INTERACTION], parser)


if __name__ == "__main__":
    main()
