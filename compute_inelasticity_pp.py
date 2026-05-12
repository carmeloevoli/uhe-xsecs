import argparse

from utils import (
    PP_INTERACTION,
    add_common_scan_arguments,
    run_scan_from_args,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Compute p+p final-state particle energy moments.",
    )
    add_common_scan_arguments(
        parser,
        default_model=("Sibyll23d",),
        default_emin=1e7,
        default_emax=1e11,
        default_stat_help="Particle statistic to average. Defaults to leading.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    run_scan_from_args(args, [PP_INTERACTION], parser)


if __name__ == "__main__":
    main()
