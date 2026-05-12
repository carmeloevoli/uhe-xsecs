import matplotlib

matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('review.mplstyle')
import numpy as np
from pathlib import Path


RUN_DIR = Path("runs")


def data_filename(model_name, particle_type, interaction="pp"):
    filename = RUN_DIR / f"chromo_{interaction}_{particle_type}_{model_name}.txt"
    if filename.exists():
        return filename
    else:
        raise FileNotFoundError(f"Data file not found: {filename}. "
                                "Run compute_inelasticity_pp.py or "
                                "compute_inelasticity_pgamma.py to generate it.")

def load_data(model_name, particle_type, interaction="pp"):
    filename = data_filename(model_name, particle_type, interaction)
    metadata = {}
    with open(filename) as f:
        for line in f:
            if not line.startswith("# "):
                break
            parts = line[2:].split(maxsplit=1)
            if len(parts) == 2:
                metadata[parts[0]] = parts[1].strip()

    observable = metadata.get("observable")
    if interaction == "pgamma" and observable != "pgamma_x":
        raise RuntimeError(
            f"{filename} is an old raw gamma+p-frame output. "
            "Rerun compute_inelasticity_pgamma.py."
        )

    return np.loadtxt(filename, unpack=True), metadata


def plot_mean_nucleon_energy(interaction="pp", particle_type="neutron", FIGNAME="uhe_nucleon_energy.pdf"):
    statistics = set()

    def _add_mean_energy(ax, model_name, particle_type, interaction, model_color, model_label):
        (E_proj, mean_En, sem_En, _), metadata = load_data(
            model_name,
            particle_type,
            interaction,
        )
        observable = metadata.get("observable")
        statistics.add(metadata.get("statistic", "unknown"))

        if observable == "pgamma_x":
            y = mean_En
            yerr = sem_En
        else:
            y = mean_En / E_proj
            yerr = sem_En / E_proj

        ax.errorbar(E_proj, y, yerr=yerr, fmt='o', 
                markersize=5, elinewidth=2, markeredgecolor=model_color, capsize=3, capthick=2, color=model_color, zorder=1, label=model_label)

    fig, ax = plt.subplots(figsize=(11, 8))

    if interaction == "pgamma":
        _add_mean_energy(ax, "Sophia20", particle_type, interaction, "tab:purple", "SOPHIA 2.0")
    elif interaction == "pp":
        _add_mean_energy(ax, "Sibyll23e", particle_type, interaction, "tab:red", "Sibyll 2.3e")
        _add_mean_energy(ax, "QGSJetII04", particle_type, interaction, "tab:blue", "QGSJet II-04")
    else:
        raise ValueError(f"Unknown interaction type: {interaction}")

    ax.set_xscale("log")
    #ax.set_yscale("log")
    if statistics == {"leading"}:
        ax.set_ylabel(r"Leading-particle energy fraction $x$")
    else:
        ax.set_ylabel(r"Mean particle energy fraction $\langle E \rangle / E_p$")

    if interaction == "pgamma":
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"Photon energy in proton rest frame $E'_\gamma$ [GeV]")
    else:
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"Projectile energy $E_p$ [GeV]")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=18, loc="upper right")

    quote = r"$\gamma$ + p" if interaction == "pgamma" else "p + p"
    quote += r" $\rightarrow$ "
    quote += " (neutron)" if particle_type == "neutron" else " (proton)"

    ax.text(0.05, 0.95, quote, fontsize=22, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    fig.tight_layout()
    FIGNAME = FIGNAME.replace(".pdf", f"_{interaction}_{particle_type}.pdf")
    fig.savefig(f'figs/{FIGNAME}', bbox_inches="tight")


def plot_mean_branching_ratios(interaction="pp", FIGNAME="uhe_branching_ratios.pdf"):
    def _add_ratio(ax, model_name, interaction, model_color, model_label):
        (E_proj, _, _, Nn), _ = load_data(model_name, "neutron", interaction)
        (E_proj, _, _, Np), _ = load_data(model_name, "proton", interaction)

        y = Nn / (Nn + Np)
        yerr = np.sqrt((Nn * Np) / (Nn + Np)**3.) 
        
        ax.errorbar(E_proj, y, yerr=yerr, fmt='o', 
                    markersize=5, elinewidth=2, markeredgecolor=model_color, capsize=3, capthick=2, color=model_color, zorder=1, label=model_label)

    fig, ax = plt.subplots(figsize=(11, 8))

    if interaction == "pgamma":
        _add_ratio(ax, "Sophia20", "pgamma", "tab:purple", "SOPHIA 2.0")
    elif interaction == "pp":
        _add_ratio(ax, "Sibyll23e", "pp", "tab:red", "Sibyll 2.3e")
        _add_ratio(ax, "QGSJetII04", "pp", "tab:blue", "QGSJet II-04")
        #_add_ratio(ax, "EposLHC", "pp", "tab:green", "EPOS LHC")
    else:
        raise ValueError(f"Unknown interaction type: {interaction}")

    ax.set_xscale("log")
    #ax.set_yscale("log")
    if interaction == "pgamma":
        ax.set_xlabel(r"Photon energy in proton rest frame $E'_\gamma$ [GeV]")
    else:
        ax.set_xlabel(r"Projectile energy $E_p$ [GeV]")
    ax.set_ylabel(r"neutrons / (neutrons + protons)")
    ax.set_ylim(0, 1)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=18, loc="upper right")

    quote = r"$\gamma$ + p" if interaction == "pgamma" else "p + p"
    ax.text(0.05, 0.95, quote, fontsize=25, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    fig.tight_layout()
    FIGNAME = FIGNAME.replace(".pdf", f"_{interaction}.pdf")
    fig.savefig(f'figs/{FIGNAME}', bbox_inches="tight")


# run
if __name__== "__main__":
    plot_mean_branching_ratios("pp")
    plot_mean_nucleon_energy("pp", "neutron")
    plot_mean_nucleon_energy("pp", "proton")

    plot_mean_branching_ratios("pgamma")
    plot_mean_nucleon_energy("pgamma", "neutron")
    plot_mean_nucleon_energy("pgamma", "proton")
