import argparse
from pathlib import Path

import numpy as np

from chromo.constants import GeV
from chromo.kinematics import FixedTarget

from utils import (
    PP_INTERACTION,
    available_model_names,
    build_energy_grid,
    model_from_name,
    unsupported_reason,
)


ENERGY_BUDGET_GROUPS = {
    "protons": (2212,),
    "neutrons": (2112,),
    "pions": (111, 211, -211),
    "kaons": (321, -321, 311, -311, 130, 310),
}


def default_output_filename(model_name):
    return f"chromo_pp_energy_budget_{model_name}.txt"


def grouped_pid_mask(pid_array, group_pids):
    mask = np.zeros(pid_array.shape, dtype=bool)
    for pid in group_pids:
        mask |= pid_array == pid
    return mask


def scan_energy_budget_pp(
    Eproj_grid,
    model_class,
    Nevt=2000,
    groups=ENERGY_BUDGET_GROUPS,
):
    """
    Compute final-state energy sums for p+p fixed-target events.

    Returns:
      results[group]["sum"][i] = total group energy over all generated events
      results["other"]["sum"][i] = energy outside the requested groups
      results["all_final_state"]["sum"][i] = total final-state energy
    """
    Emax = float(np.max(Eproj_grid))
    gen = model_class(FixedTarget(Emax, "p", "p"))

    results = {
        group: {"sum": np.zeros_like(Eproj_grid, dtype=float)}
        for group in [*groups, "other", "all_final_state"]
    }

    for i, Ep in enumerate(Eproj_grid):
        gen.kinematics = FixedTarget(float(Ep), "p", "p")
        sum_energy = {group: 0.0 for group in groups}
        total_energy = 0.0

        for ev in gen(Nevt):
            fs = ev.final_state()
            total_energy += float(fs.en.sum())

            for group, group_pids in groups.items():
                mask = grouped_pid_mask(fs.pid, group_pids)
                sum_energy[group] += float(fs.en[mask].sum()) if mask.any() else 0.0

        tracked_energy = sum(sum_energy.values())
        for group, energy in sum_energy.items():
            results[group]["sum"][i] = energy
        results["other"]["sum"][i] = max(total_energy - tracked_energy, 0.0)
        results["all_final_state"]["sum"][i] = total_energy

    return results


def write_energy_budget_to_file(
    E_proj,
    results,
    model_name,
    filename,
    Nevt,
    groups=ENERGY_BUDGET_GROUPS,
):
    with open(filename, "w") as f:
        f.write(f"# model {model_name}\n")
        f.write(f"# interaction {PP_INTERACTION.tag}\n")
        f.write(f"# interaction_label {PP_INTERACTION.label}\n")
        f.write("# projectile p\n")
        f.write("# target p\n")
        f.write(f"# energy_note {PP_INTERACTION.energy_note}\n")
        f.write("# observable energy_budget\n")
        f.write("# statistic sum_over_events\n")
        f.write("# fraction_final_state_note group sum divided by all final-state energy\n")
        f.write("# fraction_projectile_note group sum divided by Nevt * projectile lab energy\n")
        f.write(f"# Nevt {Nevt}\n")
        for group, pids in groups.items():
            f.write(f"# group {group} {' '.join(str(pid) for pid in pids)}\n")
        f.write("# group other all final-state particles outside the listed groups\n")

        budget_groups = [*groups, "other"]
        columns = ["Eproj [GeV]", "sum_E_all_final_state [GeV]"]
        for group in budget_groups:
            columns.extend(
                [
                    f"sum_E_{group} [GeV]",
                    f"frac_final_state_{group}",
                    f"frac_projectile_{group}",
                ]
            )
        f.write("# " + "  ".join(columns) + "\n")

        for i, Ep in enumerate(E_proj):
            total_final_state = results["all_final_state"]["sum"][i]
            total_projectile = Nevt * Ep
            row = [f"{Ep / GeV:.6e}", f"{total_final_state / GeV:.6e}"]
            for group in budget_groups:
                energy_sum = results[group]["sum"][i]
                row.extend(
                    [
                        f"{energy_sum / GeV:.6e}",
                        f"{energy_sum / total_final_state:.6e}",
                        f"{energy_sum / total_projectile:.6e}",
                    ]
                )
            f.write(" ".join(row) + "\n")

    return Path(filename)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compute p+p event-level energy budget in final-state protons, "
            "neutrons, pions, and kaons."
        ),
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=available_model_names(),
        default=["Sibyll23d"],
        help="One or more chromo.models classes to run.",
    )
    parser.add_argument(
        "--nevt",
        type=int,
        default=10000,
        help="Number of events per energy point.",
    )
    parser.add_argument(
        "--emin",
        type=float,
        default=1e7,
        help="Minimum projectile energy in GeV.",
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=1e11,
        help="Maximum projectile energy in GeV.",
    )
    parser.add_argument(
        "--n-energies",
        type=int,
        default=16,
        help="Number of logarithmically spaced energy points.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output filename. Only valid for a single model.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for default output filenames.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available top-level chromo.models names and exit.",
    )
    parser.add_argument(
        "--skip-unsupported",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip models that do not support p+p.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        for name in available_model_names():
            print(name)
        return

    if args.nevt <= 0:
        parser.error("--nevt must be positive")
    if args.emin <= 0 or args.emax <= 0:
        parser.error("--emin and --emax must be positive")
    if args.n_energies < 1:
        parser.error("--n-energies must be at least 1")
    if args.n_energies > 1 and args.emin >= args.emax:
        parser.error("--emin must be smaller than --emax when --n-energies > 1")
    if args.output is not None and len(args.model) != 1:
        parser.error("--output can only be used with one --model")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    E_proj = build_energy_grid(args.emin, args.emax, args.n_energies)

    ran_any = False
    for model_name in args.model:
        model_class = model_from_name(model_name)
        reason = unsupported_reason(
            model_class,
            PP_INTERACTION.projectile,
            PP_INTERACTION.target,
        )
        if reason is not None:
            message = f"Skipping {model_name} {PP_INTERACTION.label}: {reason}"
            if args.skip_unsupported:
                print(message)
                continue
            parser.error(message)

        print(
            f"Running {model_name}: interaction={PP_INTERACTION.label}, "
            f"groups={','.join(ENERGY_BUDGET_GROUPS)}, Nevt={args.nevt}"
        )
        results = scan_energy_budget_pp(
            E_proj,
            model_class,
            Nevt=args.nevt,
        )

        filename = (
            args.output
            if args.output is not None
            else args.output_dir / default_output_filename(model_name)
        )
        print(f"Writing {model_name}: output={filename}")
        write_energy_budget_to_file(
            E_proj,
            results,
            model_name=model_name,
            filename=filename,
            Nevt=args.nevt,
        )
        ran_any = True

    if not ran_any:
        parser.error("no supported p+p model combinations to run")


if __name__ == "__main__":
    main()
