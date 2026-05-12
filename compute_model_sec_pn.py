import argparse

from utils import (
    INTERACTIONS,
    PID_ALIASES,
    PID_LABELS,
    PGAMMA_INTERACTION,
    PP_INTERACTION,
    PROTON_MASS_GEV,
    Interaction,
    add_common_scan_arguments,
    available_model_names,
    build_energy_grid,
    chromo_pid,
    custom_interaction,
    default_output_filename,
    model_from_name,
    parse_interaction,
    parse_pid,
    pid_label,
    run_scan_from_args,
    sanitize_filename_piece,
    scan_mean_energy,
    scan_mean_energy_by_pid,
    scan_mean_neutron_energy,
    unsupported_reason,
    validate_scan_args,
    write_mean_energy_to_file,
    write_mean_neutron_energy_to_file,
    write_scan_results_to_file,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility entry point for nucleon secondary scans. "
            "Prefer compute_inelasticity_pp.py or compute_inelasticity_pgamma.py."
        ),
    )
    add_common_scan_arguments(
        parser,
        default_model=("Sibyll23d",),
        default_emin=1e7,
        default_emax=1e11,
        default_stat_help=(
            "Particle statistic to average. Defaults to leading for pp and "
            "pgamma."
        ),
    )
    parser.add_argument(
        "--interaction",
        nargs="+",
        type=parse_interaction,
        help=(
            "Interaction(s) to run. Use pp for p+p and pgamma for p+gamma. "
            "Defaults to pp unless custom --projectile/--target values are supplied."
        ),
    )
    parser.add_argument(
        "--projectile",
        default="p",
        help=(
            "Custom projectile passed to chromo.kinematics.FixedTarget. "
            "Cannot be combined with --interaction."
        ),
    )
    parser.add_argument(
        "--target",
        default="p",
        help=(
            "Custom target passed to chromo.kinematics.FixedTarget. "
            "Cannot be combined with --interaction."
        ),
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    custom_beams = args.projectile != "p" or args.target != "p"
    if args.interaction is not None and custom_beams:
        parser.error("--interaction cannot be combined with --projectile/--target")
    if args.interaction is None:
        interactions = (
            [custom_interaction(args.projectile, args.target)]
            if custom_beams
            else [INTERACTIONS["pp"]]
        )
    else:
        interactions = args.interaction

    run_scan_from_args(args, interactions, parser)


if __name__ == "__main__":
    main()
