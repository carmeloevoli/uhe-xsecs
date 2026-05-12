import matplotlib

matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
plt.style.use('review.mplstyle')
import numpy as np


def plot_diffusion(FIGNAME="diffusion.pdf"):
    fig, ax = plt.subplots(figsize=(11, 8))

    E = np.logspace(15, 21, 1000)
    muG = 1e-6 # gauss
    B = 200 * muG # gauss
    r_L = 1.08 * (E / 1e15) * (B / muG)**(-1) # pc
    H = 100 # pc

    D = cLight / 3 


    # ax.set_xscale("log")
    # #ax.set_yscale("log")
    # if interaction == "pgamma":
    #     ax.set_xlabel(r"Photon energy in proton rest frame $E'_\gamma$ [GeV]")
    # else:
    #     ax.set_xlabel(r"Projectile energy $E_p$ [GeV]")
    # ax.set_ylabel(r"neutrons / (neutrons + protons)")
    # ax.set_ylim(0, 1)
    # ax.grid(True, which="both", alpha=0.3)
    # ax.legend(fontsize=18, loc="upper right")

    # quote = r"$\gamma$ + p" if interaction == "pgamma" else "p + p"
    # ax.text(0.05, 0.95, quote, fontsize=25, transform=ax.transAxes, verticalalignment='top', bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    fig.tight_layout()
    fig.savefig(f'figs/{FIGNAME}', bbox_inches="tight")

if __name__ == "__main__":
    plot_diffusion()

