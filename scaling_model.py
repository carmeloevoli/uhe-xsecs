from scaling_model_plot import plot_fx, read_fit_curve, read_scaling_table
from scaling_model_run import (
    EposLHC,
    QGSJetII04,
    SECONDARIES,
    Sibyll23e,
    add_pi0_decay_photon_x,
    scan_x_distribution,
    write_gamma_scaling_table,
    write_neutron_scaling_table,
    write_scaling_table,
)


if __name__ == "__main__":
    plot_fx("gamma", "Sibyll23e", ylim=[1e-3, 20])
    plot_fx("gamma", "QGSJetII04", ylim=[1e-3, 20])
    plot_fx("neutron", "Sibyll23e", ylim=[3e-2, 0.5])
    plot_fx("neutron", "QGSJetII04", ylim=[3e-2, 0.5])
