from pathlib import Path

import numpy as np


SECONDARIES = ("neutron", "gamma")


def default_table_filename(secondary, model, table_dir="scaling_tables"):
    return Path(table_dir) / f"scaling_model_{secondary}_{model}.txt"


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
    model = metadata.get("model")

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


def neutron_fit_function(x, logA, alpha, beta, logB, mu, log_sigma):
    sigma = np.exp(log_sigma)
    one_minus_x = np.clip(1.0 - x, 1e-12, None)
    continuum = 10.0**logA * x**(-alpha) * one_minus_x**beta
    bump = 10.0**logB * np.exp(-0.5 * ((np.log10(x) - mu) / sigma) ** 2)
    return continuum + bump


def gamma_fit_function(x, logA, alpha, log_xc, log_k):
    xc = 10.0**log_xc
    k = np.exp(log_k)
    return 10.0**logA * x**(-alpha) * np.exp(-((x / xc) ** k))


FIT_FUNCTIONS = {
    "neutron": neutron_fit_function,
    "gamma": gamma_fit_function,
}

PARAMETER_NAMES = {
    "neutron": ("logA", "alpha", "beta", "logB", "mu", "log_sigma"),
    "gamma": ("logA", "alpha", "log_xc", "log_k"),
}

FORMULAS = {
    "neutron": (
        "x_dNdx = A*x^(-alpha)*(1-x)^beta + "
        "B*exp(-0.5*((log10(x)-mu)/sigma)^2), "
        "A=10^logA, B=10^logB, sigma=exp(log_sigma)"
    ),
    "gamma": (
        "x_dNdx = A*x^(-alpha)*exp[-(x/xc)^k], "
        "A=10^logA, xc=10^log_xc, k=exp(log_k)"
    ),
}


def positive_log(values):
    return np.log10(np.clip(values, 1e-300, None))


def linear_powerlaw_cutoff_guess(x, y):
    design = np.column_stack(
        [
            np.ones_like(x),
            -np.log10(x),
            np.log10(np.clip(1.0 - x, 1e-12, None)),
        ]
    )
    coeff, *_ = np.linalg.lstsq(design, positive_log(y), rcond=None)
    logA, alpha, beta = coeff
    return float(logA), float(alpha), max(float(beta), 0.1)


def initial_guess(secondary, x, y):
    if secondary == "gamma":
        low_x_mask = x < 3e-2
        if low_x_mask.sum() >= 3:
            logA, alpha, _ = linear_powerlaw_cutoff_guess(x[low_x_mask], y[low_x_mask])
        else:
            logA, alpha, _ = linear_powerlaw_cutoff_guess(x, y)
        return np.array([logA, alpha, np.log10(0.25), np.log(1.5)])

    continuum_mask = x < 3e-2
    if continuum_mask.sum() >= 3:
        logA, alpha, _ = linear_powerlaw_cutoff_guess(
            x[continuum_mask],
            y[continuum_mask],
        )
    else:
        logA, alpha, _ = linear_powerlaw_cutoff_guess(x, y)

    bump_mask = x > 3e-2
    if bump_mask.any():
        bump_index = np.argmax(y[bump_mask])
        bump_x = x[bump_mask][bump_index]
        bump_y = y[bump_mask][bump_index]
    else:
        bump_index = np.argmax(y)
        bump_x = x[bump_index]
        bump_y = y[bump_index]

    return np.array(
        [
            logA,
            alpha,
            1.0,
            positive_log(np.array([bump_y]))[0],
            np.log10(bump_x),
            np.log(0.25),
        ]
    )


def parameter_bounds(secondary):
    if secondary == "gamma":
        return (
            np.array([-20.0, -5.0, -4.0, np.log(0.2)]),
            np.array([20.0, 5.0, 0.0, np.log(10.0)]),
        )

    return (
        np.array([-20.0, -5.0, 0.0, -20.0, -6.0, np.log(0.02)]),
        np.array([20.0, 5.0, 30.0, 20.0, 0.0, np.log(2.0)]),
    )


def select_fit_data(
    table,
    x_min=1e-4,
    x_max=0.95,
    min_count=5.0,
    energies=None,
):
    x = table["x"]
    y = table["x_dNdx"]
    count = table["raw_bin_count"]
    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(count)
        & (x >= x_min)
        & (x <= x_max)
        & (y > 0)
        & (count >= min_count)
    )
    if energies is not None:
        energy_mask = np.zeros_like(mask, dtype=bool)
        for energy in energies:
            energy_mask |= np.isclose(table["Eproj_GeV"], float(energy))
        mask &= energy_mask

    return {
        "Eproj_GeV": table["Eproj_GeV"][mask],
        "x": x[mask],
        "y": y[mask],
        "raw_bin_count": count[mask],
    }


def fit_scaling_table(
    table,
    secondary=None,
    x_min=1e-4,
    x_max=0.95,
    min_count=5.0,
    energies=None,
    weighted=False,
):
    try:
        from scipy.optimize import least_squares
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required for fitting. Install it with `pip install scipy` "
            "or `conda install scipy`."
        ) from exc

    if secondary is None:
        secondary = table["secondary"]
    if secondary not in FIT_FUNCTIONS:
        names = ", ".join(SECONDARIES)
        raise ValueError(f"unknown secondary '{secondary}'. Use one of: {names}")

    selected = select_fit_data(
        table,
        x_min=x_min,
        x_max=x_max,
        min_count=min_count,
        energies=energies,
    )
    if selected["x"].size < len(PARAMETER_NAMES[secondary]):
        raise RuntimeError("not enough selected bins to fit")

    x = selected["x"]
    y = selected["y"]
    weights = np.ones_like(y)
    if weighted:
        weights = np.sqrt(np.clip(selected["raw_bin_count"], 1.0, None))
        weights /= np.median(weights)

    p0 = initial_guess(secondary, x, y)
    lower, upper = parameter_bounds(secondary)
    p0 = np.clip(p0, lower + 1e-6, upper - 1e-6)
    fn = FIT_FUNCTIONS[secondary]

    def residual(params):
        model_y = fn(x, *params)
        return weights * (positive_log(model_y) - positive_log(y))

    result = least_squares(
        residual,
        p0,
        bounds=(lower, upper),
        max_nfev=50000,
    )

    fit_y = fn(x, *result.x)
    log_resid = positive_log(fit_y) - positive_log(y)
    rms_log10 = float(np.sqrt(np.mean(log_resid * log_resid)))

    return {
        "secondary": secondary,
        "model": table["model"],
        "table": table["filename"],
        "selected": selected,
        "params": dict(zip(PARAMETER_NAMES[secondary], result.x)),
        "param_values": result.x,
        "success": bool(result.success),
        "message": result.message,
        "cost": float(result.cost),
        "rms_log10": rms_log10,
        "n_points": int(x.size),
    }


def evaluate_fit(secondary, params, x):
    if isinstance(params, dict):
        params = [params[name] for name in PARAMETER_NAMES[secondary]]
    return FIT_FUNCTIONS[secondary](np.asarray(x, dtype=float), *params)


def write_fit_parameters(fit, filename):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "w") as f:
        f.write(f"# secondary {fit['secondary']}\n")
        f.write(f"# model {fit['model']}\n")
        f.write(f"# table {fit['table']}\n")
        f.write(f"# formula {FORMULAS[fit['secondary']]}\n")
        f.write(f"# success {fit['success']}\n")
        f.write(f"# message {fit['message']}\n")
        f.write(f"# n_points {fit['n_points']}\n")
        f.write(f"# rms_log10 {fit['rms_log10']:.8e}\n")
        f.write("# parameter value\n")
        for name, value in fit["params"].items():
            f.write(f"{name} {value:.12e}\n")

    return filename


def write_fit_curve(fit, filename, n_grid=400):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    selected = fit["selected"]
    x_grid = np.logspace(
        np.log10(np.min(selected["x"])),
        np.log10(np.max(selected["x"])),
        n_grid,
    )
    y_grid = evaluate_fit(fit["secondary"], fit["params"], x_grid)

    with open(filename, "w") as f:
        f.write(f"# secondary {fit['secondary']}\n")
        f.write(f"# model {fit['model']}\n")
        f.write("# x  x_dNdx_fit\n")
        for x, y in zip(x_grid, y_grid):
            f.write(f"{x:.12e} {y:.12e}\n")

    return filename


def plot_fit(table, fit, filename):
    try:
        import matplotlib

        matplotlib.use("MacOSX")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for fit plots. Install it or pass --no-plot."
        ) from exc

    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 8))
    selected = fit["selected"]
    for energy in sorted(np.unique(selected["Eproj_GeV"]), reverse=True):
        mask = selected["Eproj_GeV"] == energy
        ax.plot(
            selected["x"][mask],
            selected["y"][mask],
            ".",
            alpha=0.75,
            label=rf"fit data $E_p={energy:.0e}$ GeV",
        )

    x_grid = np.logspace(
        np.log10(np.min(selected["x"])),
        np.log10(np.max(selected["x"])),
        500,
    )
    ax.plot(
        x_grid,
        evaluate_fit(fit["secondary"], fit["params"], x_grid),
        "k-",
        linewidth=2.5,
        label="fit",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$x = E_i/E_p$")
    ax.set_ylabel(r"$x\,dN_i/dx$ per inelastic event")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=12)
    ax.text(
        0.05,
        0.95,
        f"{fit['secondary']} {fit['model']}\nRMS log10 = {fit['rms_log10']:.3f}",
        transform=ax.transAxes,
        fontsize=16,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    return filename


def fit_file(
    secondary,
    model,
    table_dir="scaling_tables",
    output_dir="scaling_fits",
    x_min=1e-4,
    x_max=0.95,
    min_count=5.0,
    energies=None,
    weighted=False,
    make_plot=True,
):
    table_file = default_table_filename(secondary, model, table_dir)
    if not table_file.exists():
        raise FileNotFoundError(f"Scaling table does not exist: {table_file}")

    table = read_scaling_table(table_file)
    fit = fit_scaling_table(
        table,
        secondary=secondary,
        x_min=x_min,
        x_max=x_max,
        min_count=min_count,
        energies=energies,
        weighted=weighted,
    )

    output_dir = Path(output_dir)
    param_file = output_dir / f"scaling_fit_{secondary}_{model}.txt"
    curve_file = output_dir / f"scaling_fit_curve_{secondary}_{model}.txt"
    write_fit_parameters(fit, param_file)
    write_fit_curve(fit, curve_file)

    plot_file = None
    if make_plot:
        plot_file = output_dir / f"scaling_fit_{secondary}_{model}.pdf"
        plot_fit(table, fit, plot_file)

    return {
        "fit": fit,
        "param_file": param_file,
        "curve_file": curve_file,
        "plot_file": plot_file,
    }


def main():
    table_dir = "scaling_tables"
    output_dir = "scaling_fits"
    models = ("Sibyll23e", "QGSJetII04", "EposLHC")
    secondaries = ("neutron", "gamma")

    x_min = 1e-3
    x_max = 0.95
    min_count = 5.0
    energies = (1e10,)
    weighted = False
    make_plot = True

    for secondary in secondaries:
        for model in models:
            outputs = fit_file(
                secondary,
                model,
                table_dir=table_dir,
                output_dir=output_dir,
                x_min=x_min,
                x_max=x_max,
                min_count=min_count,
                energies=energies,
                weighted=weighted,
                make_plot=make_plot,
            )
            print(f"Wrote {outputs['param_file']}")
            print(f"Wrote {outputs['curve_file']}")
            if outputs["plot_file"] is not None:
                print(f"Wrote {outputs['plot_file']}")


if __name__ == "__main__":
    main()
