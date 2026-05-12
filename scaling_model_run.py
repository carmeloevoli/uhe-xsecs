from pathlib import Path

import numpy as np

from chromo.constants import GeV
from chromo.kinematics import FixedTarget
from chromo.models import EposLHC, QGSJetII04, Sibyll23e


PI0_MASS_GEV = 0.1349768

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


def add_pi0_decay_photon_x(hist, pion_x, xbins, Eproj):
    """Add the lab-frame x distribution from isotropic pi0 -> gamma gamma decays."""
    pi0_mass = PI0_MASS_GEV * GeV

    for x_pi in pion_x:
        pion_energy = x_pi * Eproj
        if pion_energy <= pi0_mass:
            x_gamma = 0.5 * pi0_mass / Eproj
            hist += 2.0 * np.histogram([x_gamma], bins=xbins)[0]
            continue

        beta = np.sqrt(max(1.0 - (pi0_mass / pion_energy) ** 2, 0.0))
        x_min = 0.5 * x_pi * (1.0 - beta)
        x_max = 0.5 * x_pi * (1.0 + beta)
        if x_max <= xbins[0] or x_min >= xbins[-1]:
            continue

        width = x_max - x_min
        left = np.maximum(xbins[:-1], x_min)
        right = np.minimum(xbins[1:], x_max)
        overlap = np.maximum(right - left, 0.0)
        hist += 2.0 * overlap / width


def scan_x_distribution(
    Eproj,
    model_class=Sibyll23e,
    gen=None,
    secondary="neutron",
    Nevt=20000,
    xbins=None,
    photon_source="photons",
):
    if xbins is None:
        xbins = np.logspace(-8, 0, 241)

    if secondary not in SECONDARIES:
        names = ", ".join(sorted(SECONDARIES))
        raise ValueError(f"unknown secondary '{secondary}'. Use one of: {names}")
    if photon_source not in ("photons", "pi0-decay", "photons+pi0-decay"):
        raise ValueError("photon_source must be photons, pi0-decay, or photons+pi0-decay")

    if gen is None:
        gen = model_class(FixedTarget(float(Eproj), "p", "p"))
    else:
        gen.kinematics = FixedTarget(float(Eproj), "p", "p")

    hist = np.zeros(len(xbins) - 1, dtype=float)

    for ev in gen(Nevt):
        fs = ev.final_state()

        if secondary == "gamma":
            if photon_source in ("photons", "photons+pi0-decay"):
                x = fs.en[fs.pid == 22] / Eproj
                x = x[np.isfinite(x) & (x > 0)]
                hist += np.histogram(x, bins=xbins)[0]

            if photon_source in ("pi0-decay", "photons+pi0-decay"):
                pion_x = fs.en[fs.pid == 111] / Eproj
                pion_x = pion_x[np.isfinite(pion_x) & (pion_x > 0)]
                add_pi0_decay_photon_x(hist, pion_x, xbins, Eproj)
        else:
            pid = SECONDARIES[secondary]["pid"]
            x = fs.en[fs.pid == pid] / Eproj
            x = x[np.isfinite(x) & (x > 0)]
            hist += np.histogram(x, bins=xbins)[0]

    dNdx = hist / Nevt / np.diff(xbins)
    x_center = np.sqrt(xbins[:-1] * xbins[1:])
    return x_center, dNdx, hist


def write_scaling_table(
    Eproj_grid=(1e11, 1e10, 1e9),
    Nevt=20000,
    secondaries=("neutron", "gamma"),
    photon_source="photons",
    model_class=Sibyll23e,
    output_dir=".",
):
    """Write scaling yields dN/dx and x dN/dx for the requested secondaries."""
    xbins = np.logspace(-6, 0, 241)
    Eproj_grid = np.asarray(Eproj_grid, dtype=float) * GeV
    gen = model_class(FixedTarget(float(np.max(Eproj_grid)), "p", "p"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for secondary in secondaries:
        if secondary not in SECONDARIES:
            names = ", ".join(sorted(SECONDARIES))
            raise ValueError(f"unknown secondary '{secondary}'. Use one of: {names}")

        filename = output_dir / f"scaling_model_{secondary}_{model_class.__name__}.txt"
        with open(filename, "w") as f:
            f.write(f"# model {model_class.__name__}\n")
            f.write(f"# observable {secondary}_scaling_pp\n")
            f.write("# x E_secondary/E_p\n")
            f.write("# dNdx yield per inelastic event per x\n")
            f.write("# x_dNdx plotted scaling function\n")
            if secondary == "gamma":
                f.write(f"# photon_source {photon_source}\n")
            f.write("# Eproj_GeV  x  Esecondary_GeV  dNdx  x_dNdx  raw_bin_count\n")

            for Eproj in Eproj_grid:
                Eproj_GeV = Eproj / GeV
                x, dNdx, hist = scan_x_distribution(
                    Eproj,
                    model_class=model_class,
                    gen=gen,
                    secondary=secondary,
                    Nevt=Nevt,
                    xbins=xbins,
                    photon_source=photon_source,
                )

                Esecondary_GeV = x * Eproj_GeV
                for xi, ei, dndxi, x_dndxi, counti in zip(
                    x,
                    Esecondary_GeV,
                    dNdx,
                    x * dNdx,
                    hist,
                ):
                    f.write(
                        f"{Eproj_GeV:.8e} {xi:.8e} {ei:.8e} {dndxi:.8e} "
                        f"{x_dndxi:.8e} {counti:.8e}\n"
                    )
        written.append(filename)

    return written


def write_neutron_scaling_table(**kwargs):
    return write_scaling_table(secondaries=("neutron",), **kwargs)


def write_gamma_scaling_table(**kwargs):
    return write_scaling_table(secondaries=("gamma",), **kwargs)


def main():
    model_class = EposLHC
    write_scaling_table(
        Eproj_grid=(1e11, 1e10, 1e9),
        Nevt=10000,
        secondaries=("neutron", "gamma"),
        photon_source="photons+pi0-decay",
        model_class=model_class,
        output_dir="scaling_tables",
    )


if __name__ == "__main__":
    main()
