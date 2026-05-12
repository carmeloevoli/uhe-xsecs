import matplotlib

matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
from pathlib import Path

plt.style.use("review.mplstyle")
import numpy as np


SECONDARIES = {
    "neutron": {
        "label": "neutron",
        "pid": 2112,
        "ylabel": r"$x\,dN_n/dx$ per inelastic event",
        "filename": "scaling_model_neutron.pdf",
    },
    "gamma": {
        "label": r"$\gamma$",
        "pid": 22,
        "ylabel": r"$x\,dN_\gamma/dx$ per inelastic event",
        "filename": "scaling_model_gamma.pdf",
    },
}

ENERGY_COLORS = {
    1e9: "tab:blue",
    1e10: "tab:orange",
    1e11: "tab:red",
}


def read_scaling_table(filename):
    metadata = {}
    with open(filename) as f:
        for line in f:
            if not line.startswith("# "):
                break
            parts = line[2:].split(maxsplit=1)
            if len(parts) == 2:
                metadata[parts[0]] = parts[1].strip()

    data = np.loadtxt(filename, comments="#")
    if data.ndim == 1:
        data = data[None, :]

    observable = metadata.get("observable", "")
    secondary = observable.removesuffix("_scaling_pp")
    if secondary not in SECONDARIES:
        stem_parts = Path(filename).stem.split("_")
        secondary = stem_parts[2] if len(stem_parts) > 2 else "unknown"

    model = metadata.get("model")
    if model is None:
        prefix = f"scaling_model_{secondary}_"
        stem = Path(filename).stem
        model = stem[len(prefix):] if stem.startswith(prefix) else stem

    return {
        "filename": Path(filename),
        "metadata": metadata,
        "model": model,
        "secondary": secondary,
        "Eproj_GeV": data[:, 0],
        "x": data[:, 1],
        "Esecondary_GeV": data[:, 2],
        "dNdx": data[:, 3],
        "x_dNdx": data[:, 4],
        "raw_bin_count": data[:, 5],
    }


def read_fit_curve(filename):
    data = np.loadtxt(filename, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    return {
        "filename": Path(filename),
        "x": data[:, 0],
        "x_dNdx": data[:, 1],
    }


def plot_fx(
    secondary,
    model,
    table_dir="scaling_tables",
    fit_dir="scaling_fits",
    Eproj_grid=(1e11, 1e10, 1e9),
    ylim=None,
    show_fit=True,
    output=None,
):
    if secondary not in SECONDARIES:
        names = ", ".join(sorted(SECONDARIES))
        raise ValueError(f"unknown secondary '{secondary}'. Use one of: {names}")

    Eproj_grid = tuple(float(Ep) for Ep in Eproj_grid)
    cfg = SECONDARIES[secondary]
    filename = Path(table_dir) / f"scaling_model_{secondary}_{model}.txt"
    if not filename.exists():
        raise FileNotFoundError(f"Scaling table does not exist: {filename}")

    table = read_scaling_table(filename)
    if table["secondary"] != secondary:
        raise ValueError(
            f"{filename} contains secondary={table['secondary']}, expected {secondary}"
        )
    if table["model"] != model:
        raise ValueError(f"{filename} contains model={table['model']}, expected {model}")

    fig, ax = plt.subplots(figsize=(11, 8))
    for Eproj_GeV in Eproj_grid:
        mask = np.isclose(table["Eproj_GeV"], Eproj_GeV)
        if not mask.any():
            continue
        ax.plot(
            table["x"][mask],
            table["x_dNdx"][mask],
            color=ENERGY_COLORS.get(float(Eproj_GeV)),
            linewidth=2,
            label=rf"log $E_p$={np.log10(Eproj_GeV):.0f}",
        )

    if show_fit:
        fit_filename = Path(fit_dir) / f"scaling_fit_curve_{secondary}_{model}.txt"
        if fit_filename.exists():
            fit_curve = read_fit_curve(fit_filename)
            ax.plot(
                fit_curve["x"],
                fit_curve["x_dNdx"],
                color="black",
                linestyle="--",
                linewidth=3,
                label="fit",
                zorder=5,
            )
        else:
            print(f"Fit curve not found, skipping: {fit_filename}")

    ax.set_xscale("log")
    ax.set_xlim([1e-4, 1])
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_yscale("log")
    ax.set_xlabel(r"$x = E_i/E_p$")
    ax.set_ylabel(cfg["ylabel"])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=22)
    ax.text(
        0.05,
        0.95,
        model,
        transform=ax.transAxes,
        fontsize=20,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    if output is None:
        output = f"figs/scaling_model_{secondary}_{model}.pdf"
    fig.savefig(output, bbox_inches="tight")
    return Path(output)


def main():
    plot_fx("gamma", "Sibyll23e", ylim=[1e-3, 20])
    plot_fx("gamma", "QGSJetII04", ylim=[1e-3, 20])
    plot_fx("gamma", "EposLHC", ylim=[1e-3, 20])

    plot_fx("neutron", "Sibyll23e", ylim=[1e-2, 1])
    plot_fx("neutron", "QGSJetII04", ylim=[1e-2, 1])
    plot_fx("neutron", "EposLHC", ylim=[1e-2, 1])

if __name__ == "__main__":
    main()
