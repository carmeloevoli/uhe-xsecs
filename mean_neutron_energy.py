# def scan_mean_neutron_energy(Eproj_grid, Nevt=2000, pid=2112):
#     """
#     Returns:
#       mean_En[i] = <E_n> at projectile energy Eproj_grid[i]
#       sem_En[i]  = standard error of the mean
#       Nn[i]      = total number of sampled particles
#     """
#     Emax = float(np.max(Eproj_grid))
#     gen = Sibyll23d(FixedTarget(Emax, "p", "p"))

#     mean_En = np.zeros_like(Eproj_grid, dtype=float)
#     sem_En = np.zeros_like(Eproj_grid, dtype=float)
#     Nn = np.zeros_like(Eproj_grid, dtype=int)

#     for i, Ep in enumerate(Eproj_grid):
#         gen.kinematics = FixedTarget(float(Ep), "p", "p")

#         sumE = 0.0
#         sumE2 = 0.0
#         cnt = 0

#         for ev in gen(Nevt):
#             fs = ev.final_state()
#             en = fs.en[fs.pid == pid]
#             if en.size == 0:
#                 continue
#             sumE += float(en.sum())
#             sumE2 += float((en * en).sum())
#             cnt += int(en.size)

#         Nn[i] = cnt
#         if cnt > 0:
#             mu = sumE / cnt
#             mean_En[i] = mu
#             if cnt > 1:
#                 var = max(sumE2 / cnt - mu * mu, 0.0)
#                 sem_En[i] = np.sqrt(var / cnt)
#             else:
#                 sem_En[i] = np.nan
#         else:
#             mean_En[i] = np.nan
#             sem_En[i] = np.nan

#     return mean_En, sem_En, Nn


# def plot_mean_neutron_energy():
#     Eproj = np.logspace(8, 12, 10) * GeV
#     Nevt = 20000

#     mean_En, sem_En, Nn = scan_mean_neutron_energy(Eproj, Nevt=Nevt)

#     fig, ax = plt.subplots(figsize=(11, 8))

#     y = mean_En / Eproj
#     yerr = sem_En / Eproj

#     color = "tab:red"
#     ax.errorbar(
#         Eproj / GeV,
#         y,
#         yerr=yerr,
#         fmt="o",
#         markersize=5,
#         elinewidth=2,
#         markeredgecolor=color,
#         capsize=3,
#         capthick=2,
#         color=color,
#         zorder=1,
#     )

#     ax.set_xscale("log")
#     ax.set_xlabel(r"Projectile energy $E_p$ [GeV]")
#     ax.set_ylabel(r"$\langle E_n \rangle / E_p$")
#     ax.grid(True, which="both", alpha=0.3)

#     fig.tight_layout()
#     fig.savefig("inelasticity_model_neutron.pdf", bbox_inches="tight")

#     for Ep, nn in zip(Eproj / GeV, Nn):
#         print(f"E_p = {Ep: .3e} GeV  |  neutrons sampled = {nn}")
