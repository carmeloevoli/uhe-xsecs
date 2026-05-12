import argparse
import os
from dataclasses import dataclass
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


PROTON_MASS_GEV = util.mass(2212)


PID_ALIASES = {
    "gamma": 22,
    "photon": 22,
    "pi0": 111,
    "pizero": 111,
    "pi+": 211,
    "piplus": 211,
    "pi-": -211,
    "piminus": -211,
    "n": 2112,
    "neutron": 2112,
    "nbar": -2112,
    "antineutron": -2112,
    "anti-neutron": -2112,
    "p": 2212,
    "proton": 2212,
    "pbar": -2212,
    "antiproton": -2212,
    "anti-proton": -2212,
}

PID_LABELS = {
    22: "photon",
    111: "pi0",
    211: "piplus",
    -211: "piminus",
    2112: "neutron",
    -2112: "antineutron",
    2212: "proton",
    -2212: "antiproton",
}


@dataclass(frozen=True)
class Interaction:
    tag: str
    label: str
    projectile: str
    target: str
    energy_note: str
    quantity: str = "energy"
    default_stat: str = "all"


PP_INTERACTION = Interaction(
    tag="pp",
    label="p+p",
    projectile="p",
    target="p",
    energy_note="proton lab energy in a proton fixed-target frame",
    default_stat="leading",
)

PGAMMA_INTERACTION = Interaction(
    tag="pgamma",
    label="p+gamma",
    projectile="gamma",
    target="p",
    energy_note=(
        "photon energy in the proton rest frame; final-state values are "
        "reported as high-energy proton-beam fractions"
    ),
    quantity="pgamma_x",
    default_stat="leading",
)

INTERACTIONS = {
    PP_INTERACTION.tag: PP_INTERACTION,
    PGAMMA_INTERACTION.tag: PGAMMA_INTERACTION,
}

INTERACTION_ALIASES = {
    "pp": "pp",
    "p+p": "pp",
    "p-p": "pp",
    "p_on_p": "pp",
    "p-on-p": "pp",
    "proton-proton": "pp",
    "pgamma": "pgamma",
    "p-gamma": "pgamma",
    "p+gamma": "pgamma",
    "p-photon": "pgamma",
    "p+photon": "pgamma",
    "gamma-p": "pgamma",
    "gamma+p": "pgamma",
    "gamma_on_p": "pgamma",
    "gamma-on-p": "pgamma",
    "photon-proton": "pgamma",
}


def parse_pid(value):
    """Parse a PDG PID integer or a common particle alias."""
    key = value.strip().lower()
    if key in PID_ALIASES:
        return PID_ALIASES[key]

    try:
        return int(value)
    except ValueError as exc:
        aliases = ", ".join(sorted(PID_ALIASES))
        raise argparse.ArgumentTypeError(
            f"unknown PID '{value}'. Use a PDG integer or one of: {aliases}"
        ) from exc


def pid_label(pid):
    if pid in PID_LABELS:
        return PID_LABELS[pid]
    if pid < 0:
        return f"pidminus{abs(pid)}"
    return f"pid{pid}"


def available_model_names():
    return tuple(getattr(chromo_models, "__all__", ()))


def model_from_name(name):
    return getattr(chromo_models, name)


def sanitize_filename_piece(text):
    return (
        str(text)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("+", "")
        .replace("-", "minus")
        .replace(".", "p")
        .replace(",", "_")
        .replace(":", "_")
    )


def default_output_filename(model_name, pid, interaction_tag="pp"):
    return f"chromo_{interaction_tag}_{pid_label(pid)}_{model_name}.txt"


def parse_interaction(value):
    key = value.strip().lower().replace("_", "-")
    canonical = INTERACTION_ALIASES.get(key)
    if canonical is None:
        names = ", ".join(sorted(INTERACTION_ALIASES))
        raise argparse.ArgumentTypeError(
            f"unknown interaction '{value}'. Use one of: {names}"
        )
    return INTERACTIONS[canonical]


def custom_interaction(projectile, target):
    projectile_tag = sanitize_filename_piece(projectile)
    target_tag = sanitize_filename_piece(target)
    return Interaction(
        tag=f"{projectile_tag}_on_{target_tag}",
        label=f"{projectile}+{target}",
        projectile=projectile,
        target=target,
        energy_note="projectile lab energy in the requested fixed-target frame",
    )


def chromo_pid(value):
    try:
        return util.name2pdg(str(value))
    except KeyError:
        return util.PDGID(int(value))


def unsupported_reason(model_class, projectile, target):
    projectile_pid = chromo_pid(projectile)
    target_pid = chromo_pid(target)

    if (
        hasattr(model_class, "projectiles")
        and projectile_pid not in model_class.projectiles
    ):
        return (
            f"projectile {projectile}[{int(projectile_pid)}] is not allowed "
            f"for {model_class.__name__}"
        )
    if hasattr(model_class, "targets") and target_pid not in model_class.targets:
        return (
            f"target {target}[{int(target_pid)}] is not allowed "
            f"for {model_class.__name__}"
        )

    return None


def scan_mean_energy_by_pid(
    Eproj_grid,
    model_class,
    pids,
    Nevt=2000,
    projectile="p",
    target="p",
    quantity="energy",
    stat="all",
):
    """
    Returns:
      mean[i] = mean selected-particle value at projectile energy Eproj_grid[i]
      sem[i]  = standard error of the mean
      N[i]    = total number of selected samples
    """
    Emax = float(np.max(Eproj_grid))

    # Initialize once at the maximum energy, then scan by updating kinematics.
    gen = model_class(FixedTarget(Emax, projectile, target))

    pids = tuple(pids)
    mean_E = {pid: np.zeros_like(Eproj_grid, dtype=float) for pid in pids}
    sem_E = {pid: np.zeros_like(Eproj_grid, dtype=float) for pid in pids}
    Npid = {pid: np.zeros_like(Eproj_grid, dtype=int) for pid in pids}

    for i, Ep in enumerate(Eproj_grid):
        gen.kinematics = FixedTarget(float(Ep), projectile, target)

        sumE = {pid: 0.0 for pid in pids}
        sumE2 = {pid: 0.0 for pid in pids}
        cnt = {pid: 0 for pid in pids}

        for ev in gen(Nevt):
            fs = ev.final_state()

            if quantity == "pgamma_x":
                # Chromo generates photonuclear events as gamma(+z) on a
                # proton at rest. A UHE proton beam frame is obtained by
                # boosting the proton-rest frame opposite to the photon
                # direction, so E_lab/E_p -> (E - pz) / m_p at high gamma.
                event_values = (fs.en - fs.pz) / PROTON_MASS_GEV
            else:
                event_values = fs.en

            for pid in pids:
                mask = fs.pid == pid
                if not mask.any():
                    continue

                values = event_values[mask]
                if stat == "leading":
                    values = np.array([np.max(values)])

                sumE[pid] += float(values.sum())
                sumE2[pid] += float((values * values).sum())
                cnt[pid] += int(values.size)

        for pid in pids:
            Npid[pid][i] = cnt[pid]
            if cnt[pid] > 0:
                mu = sumE[pid] / cnt[pid]
                mean_E[pid][i] = mu
                if cnt[pid] > 1:
                    var = max(sumE2[pid] / cnt[pid] - mu * mu, 0.0)
                    sem_E[pid][i] = np.sqrt(var / cnt[pid])
                else:
                    sem_E[pid][i] = np.nan
            else:
                mean_E[pid][i] = np.nan
                sem_E[pid][i] = np.nan

    return {pid: (mean_E[pid], sem_E[pid], Npid[pid]) for pid in pids}


def scan_mean_energy(
    Eproj_grid,
    model_class,
    Nevt=2000,
    pid=2112,
    projectile="p",
    target="p",
    quantity="energy",
    stat="all",
):
    return scan_mean_energy_by_pid(
        Eproj_grid,
        model_class,
        [pid],
        Nevt=Nevt,
        projectile=projectile,
        target=target,
        quantity=quantity,
        stat=stat,
    )[pid]


def write_scan_results_to_file(
    E_proj,
    mean_E,
    sem_E,
    Npid,
    model_name,
    filename,
    PID,
    Nevt,
    projectile,
    target,
    interaction,
    quantity,
    stat,
):
    with open(filename, "w") as f:
        f.write(f"# model {model_name}\n")
        if interaction is not None:
            f.write(f"# interaction {interaction.tag}\n")
            f.write(f"# interaction_label {interaction.label}\n")
        f.write(f"# pid {PID}\n")
        f.write(f"# projectile {projectile}\n")
        f.write(f"# target {target}\n")
        if interaction is not None:
            f.write(f"# energy_note {interaction.energy_note}\n")
        f.write(f"# observable {quantity}\n")
        f.write(f"# statistic {stat}\n")
        f.write(f"# Nevt {Nevt}\n")
        if quantity == "pgamma_x":
            f.write("# Egamma_prf [GeV]  mean_x  sem_x  Nsample\n")
        else:
            f.write("# Eproj [GeV]  mean_E [GeV]  sem_E [GeV]  Nsample\n")
        for Ep, mu, sem, nn in zip(E_proj, mean_E, sem_E, Npid):
            if quantity == "pgamma_x":
                f.write(f"{Ep / GeV:.6e} {mu:.6e} {sem:.6e} {nn}\n")
            else:
                f.write(f"{Ep / GeV:.6e} {mu / GeV:.6e} {sem / GeV:.6e} {nn}\n")

    return Path(filename)


def write_mean_energy_to_file(
    E_proj,
    model_name="Sibyll23d",
    filename=None,
    PID=2112,
    Nevt=4000,
    projectile="p",
    target="p",
    interaction=None,
    stat=None,
):
    if filename is None:
        interaction_tag = interaction.tag if interaction else "pp"
        filename = default_output_filename(model_name, PID, interaction_tag)

    model_class = model_from_name(model_name)
    if interaction is not None:
        projectile = interaction.projectile
        target = interaction.target
        quantity = interaction.quantity
        stat = interaction.default_stat if stat is None else stat
    else:
        quantity = "energy"
        stat = PP_INTERACTION.default_stat if stat is None else stat

    mean_E, sem_E, Npid = scan_mean_energy(
        E_proj,
        model_class,
        Nevt=Nevt,
        pid=PID,
        projectile=projectile,
        target=target,
        quantity=quantity,
        stat=stat,
    )

    return write_scan_results_to_file(
        E_proj,
        mean_E,
        sem_E,
        Npid,
        model_name=model_name,
        filename=filename,
        PID=PID,
        Nevt=Nevt,
        projectile=projectile,
        target=target,
        interaction=interaction,
        quantity=quantity,
        stat=stat,
    )


def scan_mean_neutron_energy(Eproj_grid, Nevt=2000, pid=2112):
    return scan_mean_energy(
        Eproj_grid,
        chromo_models.Sibyll23d,
        Nevt=Nevt,
        pid=pid,
    )


def write_mean_neutron_energy_to_file(
    E_proj,
    filename="chromo_pp_neutron_Sibyll23d.txt",
    PID=2112,
    Nevt=4000,
):
    return write_mean_energy_to_file(
        E_proj,
        model_name="Sibyll23d",
        filename=filename,
        PID=PID,
        Nevt=Nevt,
    )


def add_common_scan_arguments(
    parser,
    default_model=("Sibyll23d",),
    default_emin=1e7,
    default_emax=1e11,
    default_stat_help="Defaults to the interaction-specific statistic.",
):
    model_names = available_model_names()
    parser.add_argument(
        "--model",
        nargs="+",
        choices=model_names,
        default=list(default_model),
        help="One or more chromo.models classes to run.",
    )
    parser.add_argument(
        "--pid",
        nargs="+",
        type=parse_pid,
        default=[2212],
        help="One or more final-state PDG PIDs or aliases, e.g. proton neutron 2212 2112.",
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
        default=default_emin,
        help="Minimum projectile energy in GeV.",
    )
    parser.add_argument(
        "--emax",
        type=float,
        default=default_emax,
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
        help="Output filename. Only valid for a single model/PID combination.",
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
        help="Skip model/interaction combinations not supported by the model.",
    )
    parser.add_argument(
        "--stat",
        choices=("all", "leading"),
        help=default_stat_help,
    )


def build_energy_grid(emin, emax, n_energies):
    if n_energies == 1:
        return np.array([emin]) * GeV
    return np.logspace(np.log10(emin), np.log10(emax), n_energies) * GeV


def validate_scan_args(parser, args, interactions):
    if args.nevt <= 0:
        parser.error("--nevt must be positive")
    if args.emin <= 0 or args.emax <= 0:
        parser.error("--emin and --emax must be positive")
    if args.n_energies < 1:
        parser.error("--n-energies must be at least 1")
    if args.n_energies > 1 and args.emin >= args.emax:
        parser.error("--emin must be smaller than --emax when --n-energies > 1")

    combinations = len(args.model) * len(args.pid) * len(interactions)
    if args.output is not None and combinations != 1:
        parser.error(
            "--output can only be used with one --model, one --pid, "
            "and one interaction"
        )


def run_scan_from_args(args, interactions, parser):
    if args.list_models:
        for name in available_model_names():
            print(name)
        return

    validate_scan_args(parser, args, interactions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    E_proj = build_energy_grid(args.emin, args.emax, args.n_energies)

    ran_any = False
    for model_name in args.model:
        model_class = model_from_name(model_name)
        for interaction in interactions:
            reason = unsupported_reason(
                model_class,
                interaction.projectile,
                interaction.target,
            )
            if reason is not None:
                message = f"Skipping {model_name} {interaction.label}: {reason}"
                if args.skip_unsupported:
                    print(message)
                    continue
                parser.error(message)

            stat = interaction.default_stat if args.stat is None else args.stat

            print(
                f"Running {model_name}: interaction={interaction.label} "
                f"(chromo {interaction.projectile}+{interaction.target}), "
                f"pids={','.join(pid_label(pid) for pid in args.pid)}, "
                f"stat={stat}, Nevt={args.nevt}"
            )
            results = scan_mean_energy_by_pid(
                E_proj,
                model_class,
                args.pid,
                Nevt=args.nevt,
                projectile=interaction.projectile,
                target=interaction.target,
                quantity=interaction.quantity,
                stat=stat,
            )

            for pid in args.pid:
                ran_any = True
                if args.output is not None:
                    filename = args.output
                else:
                    filename = args.output_dir / default_output_filename(
                        model_name,
                        pid,
                        interaction.tag,
                    )

                print(
                    f"Writing {model_name}: pid={pid} ({pid_label(pid)}), "
                    f"output={filename}"
                )
                mean_E, sem_E, Npid = results[pid]
                write_scan_results_to_file(
                    E_proj,
                    mean_E,
                    sem_E,
                    Npid,
                    model_name=model_name,
                    filename=filename,
                    PID=pid,
                    Nevt=args.nevt,
                    projectile=interaction.projectile,
                    target=interaction.target,
                    interaction=interaction,
                    quantity=interaction.quantity,
                    stat=stat,
                )

    if not ran_any:
        parser.error("no supported model/interaction combinations to run")
