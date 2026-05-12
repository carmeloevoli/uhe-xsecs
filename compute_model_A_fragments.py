import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    int(os.environ.get("DEBUG", "0"))
except ValueError:
    os.environ["DEBUG"] = "0"

from chromo import util
from chromo.constants import GeV
from chromo.kinematics import FixedTarget
import chromo.models as chromo_models


ENERGY_FACTORS_GEV = {
    "eV": 1e-9,
    "GeV": 1.0,
    "TeV": 1e3,
}


def available_model_names():
    return tuple(getattr(chromo_models, "__all__", ()))


def model_from_name(name):
    return getattr(chromo_models, name)


def parse_particle(value):
    """
    Parse a chromo particle/nucleus name, a PDG ID, or an A,Z pair.

    Examples:
      p, proton, Si28, 1000140280, 28,14
    """
    text = value.strip()

    for separator in (",", ":"):
        if separator in text:
            parts = text.split(separator)
            if len(parts) != 2:
                break
            try:
                A = int(parts[0])
                Z = int(parts[1])
            except ValueError as exc:
                raise argparse.ArgumentTypeError(
                    f"could not parse A,Z particle '{value}'"
                ) from exc
            return util.AZ2pdg(A, Z)

    try:
        return util.name2pdg(text)
    except KeyError:
        pass

    try:
        return util.PDGID(int(text))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"unknown particle '{value}'. Use names like p or Si28, a PDG ID, or A,Z."
        ) from exc


def particle_name(pid):
    return util.pdg2name(pid).replace("Unknown", "PDG")


def sanitize_filename_piece(text):
    return (
        text.replace(" ", "")
        .replace("+", "")
        .replace("-", "minus")
        .replace(".", "p")
        .replace(",", "_")
        .replace(":", "_")
    )


def compact_number(value):
    return f"{value:.6g}".replace("+", "").replace(".", "p")


def default_output_filename(model_name, projectile, target, energy, energy_unit):
    projectile_name = sanitize_filename_piece(particle_name(projectile))
    target_name = sanitize_filename_piece(particle_name(target))
    energy_tag = sanitize_filename_piece(f"{compact_number(energy)}{energy_unit}")
    return f"fragment_counts_byA_{projectile_name}_on_{target_name}_{model_name}_{energy_tag}.txt"


def default_isotopes_filename(model_name, projectile, target, energy, energy_unit):
    projectile_name = sanitize_filename_piece(particle_name(projectile))
    target_name = sanitize_filename_piece(particle_name(target))
    energy_tag = sanitize_filename_piece(f"{compact_number(energy)}{energy_unit}")
    return f"fragment_counts_byAZ_{projectile_name}_on_{target_name}_{model_name}_{energy_tag}.txt"


def sem_from_sums(total, sumsq, nevt):
    mean = total / nevt
    var = max(sumsq / nevt - mean * mean, 0.0)
    return mean, np.sqrt(var / nevt)


def scan_fragment_counts(
    model_class,
    energy_gev,
    projectile,
    target,
    nevt,
    min_a=1,
    max_a=None,
    record="final",
    progress_every=0,
):
    projectile_A, _ = util.pdg2AZ(projectile)
    if max_a is None:
        max_a = projectile_A

    gen = model_class(FixedTarget(energy_gev * GeV, projectile, target))

    total_by_a = Counter()
    sumsq_by_a = defaultdict(float)
    events_with_a = Counter()

    total_by_az = Counter()
    sumsq_by_az = defaultdict(float)
    events_with_az = Counter()

    for iev, ev in enumerate(gen(nevt), start=1):
        particles = ev.final_state() if record == "final" else ev

        event_by_a = Counter()
        event_by_az = Counter()
        for raw_pid in particles.pid:
            A, Z = util.pdg2AZ(int(raw_pid))
            if A < min_a or A > max_a:
                continue
            event_by_a[A] += 1
            event_by_az[(A, Z)] += 1

        for A, count in event_by_a.items():
            total_by_a[A] += count
            sumsq_by_a[A] += count * count
            events_with_a[A] += 1

        for key, count in event_by_az.items():
            total_by_az[key] += count
            sumsq_by_az[key] += count * count
            events_with_az[key] += 1

        if progress_every and iev % progress_every == 0:
            print(f"  processed {iev}/{nevt} events", flush=True)

    a_values = range(min_a, max_a + 1)
    rows_by_a = []
    for A in a_values:
        total = total_by_a[A]
        mean, sem = sem_from_sums(total, sumsq_by_a[A], nevt)
        rows_by_a.append((A, total, mean, sem, events_with_a[A]))

    rows_by_az = []
    for (A, Z) in sorted(total_by_az):
        total = total_by_az[(A, Z)]
        mean, sem = sem_from_sums(total, sumsq_by_az[(A, Z)], nevt)
        rows_by_az.append((A, Z, total, mean, sem, events_with_az[(A, Z)]))

    return rows_by_a, rows_by_az


def write_by_a(filename, rows, metadata):
    with open(filename, "w") as f:
        for key, value in metadata.items():
            f.write(f"# {key} {value}\n")
        f.write(
            "# A  total_fragments  mean_fragments_per_event  "
            "sem_fragments_per_event  events_with_fragment\n"
        )
        for A, total, mean, sem, events_with in rows:
            f.write(f"{A:d} {total:d} {mean:.8e} {sem:.8e} {events_with:d}\n")

    return Path(filename)


def write_by_az(filename, rows, metadata):
    with open(filename, "w") as f:
        for key, value in metadata.items():
            f.write(f"# {key} {value}\n")
        f.write(
            "# A  Z  total_fragments  mean_fragments_per_event  "
            "sem_fragments_per_event  events_with_fragment\n"
        )
        for A, Z, total, mean, sem, events_with in rows:
            f.write(f"{A:d} {Z:d} {total:d} {mean:.8e} {sem:.8e} {events_with:d}\n")

    return Path(filename)


def build_parser():
    model_names = available_model_names()
    parser = argparse.ArgumentParser(
        description="Compute nuclear fragment multiplicities from chromo events.",
    )
    parser.add_argument(
        "--model",
        choices=model_names,
        default="EposLHC",
        help="chromo.models class to run.",
    )
    parser.add_argument(
        "--projectile",
        type=parse_particle,
        default=parse_particle("Si28"),
        help="Projectile nucleus, e.g. Si28, 1000140280, or 28,14.",
    )
    parser.add_argument(
        "--target",
        type=parse_particle,
        default=parse_particle("p"),
        help="Fixed target particle/nucleus, e.g. p.",
    )
    parser.add_argument(
        "--energy",
        type=float,
        default=1e17,
        help="Projectile lab energy in the selected unit.",
    )
    parser.add_argument(
        "--energy-unit",
        choices=tuple(ENERGY_FACTORS_GEV),
        default="eV",
        help="Unit for --energy.",
    )
    parser.add_argument(
        "--nevt",
        type=int,
        default=1000,
        help="Number of events.",
    )
    parser.add_argument(
        "--min-a",
        type=int,
        default=1,
        help="Minimum fragment mass number A to count.",
    )
    parser.add_argument(
        "--max-a",
        type=int,
        help="Maximum fragment mass number A to count. Defaults to projectile A.",
    )
    parser.add_argument(
        "--record",
        choices=("final", "all"),
        default="final",
        help="Count final-state particles or the raw event record.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for the A histogram.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        help="Directory for default output filenames.",
    )
    parser.add_argument(
        "--isotopes-output",
        type=Path,
        help="Optional output file for isotope-resolved A,Z counts.",
    )
    parser.add_argument(
        "--write-isotopes",
        action="store_true",
        help="Also write a default isotope-resolved A,Z table.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print progress every N events. Use 0 to disable.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available top-level chromo.models names and exit.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_models:
        for name in available_model_names():
            print(name)
        return

    projectile_A, projectile_Z = util.pdg2AZ(args.projectile)
    target_A, target_Z = util.pdg2AZ(args.target)
    if projectile_A <= 1:
        parser.error("--projectile must be a nucleus with A > 1")
    if args.energy <= 0:
        parser.error("--energy must be positive")
    if args.nevt <= 0:
        parser.error("--nevt must be positive")
    if args.min_a < 1:
        parser.error("--min-a must be at least 1")
    if args.max_a is not None and args.max_a < args.min_a:
        parser.error("--max-a must be greater than or equal to --min-a")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")

    max_a = args.max_a if args.max_a is not None else projectile_A
    energy_gev = args.energy * ENERGY_FACTORS_GEV[args.energy_unit]
    model_class = model_from_name(args.model)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output
    if output is None:
        output = args.output_dir / default_output_filename(
            args.model, args.projectile, args.target, args.energy, args.energy_unit
        )

    isotopes_output = args.isotopes_output
    if isotopes_output is None and args.write_isotopes:
        isotopes_output = args.output_dir / default_isotopes_filename(
            args.model, args.projectile, args.target, args.energy, args.energy_unit
        )

    print(
        f"Running {args.model}: {particle_name(args.projectile)} "
        f"(A={projectile_A}, Z={projectile_Z}) on {particle_name(args.target)} "
        f"at {args.energy:.6g} {args.energy_unit} ({energy_gev:.6g} GeV), "
        f"Nevt={args.nevt}"
    )

    rows_by_a, rows_by_az = scan_fragment_counts(
        model_class,
        energy_gev,
        args.projectile,
        args.target,
        args.nevt,
        min_a=args.min_a,
        max_a=max_a,
        record=args.record,
        progress_every=args.progress_every,
    )

    metadata = {
        "model": args.model,
        "projectile": particle_name(args.projectile),
        "projectile_pdg": int(args.projectile),
        "projectile_A": projectile_A,
        "projectile_Z": projectile_Z,
        "target": particle_name(args.target),
        "target_pdg": int(args.target),
        "target_A": target_A,
        "target_Z": target_Z,
        "energy_lab_eV": f"{energy_gev * 1e9:.8e}",
        "energy_lab_GeV": f"{energy_gev:.8e}",
        "Nevt": args.nevt,
        "record": args.record,
        "min_A": args.min_a,
        "max_A": max_a,
    }

    write_by_a(output, rows_by_a, metadata)
    print(f"Wrote {output}")

    if isotopes_output is not None:
        write_by_az(isotopes_output, rows_by_az, metadata)
        print(f"Wrote {isotopes_output}")


if __name__ == "__main__":
    main()
