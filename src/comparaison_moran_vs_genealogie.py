# -*- coding: utf-8 -*-
# Compare le temps de coalescence entre le "vrai" Moran (evenements,
# generation_donnees_backward.py) et l'approximation par genealogie
# (generation_donnees_genealogie.py). Meme principe que
# verif_donnees_steph.py (test KS + QQ plot + histogramme), adapte pour
# lire plusieurs fichiers (un par repetition) au lieu d'un seul gros
# fichier.
# Robin Pioch, stage M1 LIRMM, juillet 2026

# Usage :
#   python comparaison_moran_vs_genealogie.py --dossier_moran donnees --dossier_genealogie donnees_genealogie --n_echantillons 10 --sauvegarder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import argparse
import glob
import os


def charger_moran(dossier):
    """
    Concatene tous les donnees_jeuX.csv du dossier (vrai Moran) en un seul
    DataFrame. Les paires non coalescees (coalesce=0) sont retirees, la
    genealogie n'a pas d'equivalent (elle coalesce toujours).
    """
    chemins = sorted(glob.glob(os.path.join(dossier, "donnees_jeu*.csv")))
    print(f"{len(chemins)} fichiers trouves dans {dossier}")
    tables = [pd.read_csv(c) for c in chemins]
    donnees = pd.concat(tables, ignore_index=True)
    donnees = donnees[donnees["coalesce"] == 1]
    return donnees


def charger_genealogie(dossier, n_echantillons):
    """
    Concatene tous les donnees_jeuX_n{n_echantillons}.csv du dossier
    (genealogie) en un seul DataFrame.
    """
    motif = os.path.join(dossier, f"donnees_jeu*_n{n_echantillons}.csv")
    chemins = sorted(glob.glob(motif))
    print(f"{len(chemins)} fichiers trouves dans {dossier} (n_echantillons={n_echantillons})")
    tables = [pd.read_csv(c) for c in chemins]
    donnees = pd.concat(tables, ignore_index=True)
    return donnees


parser = argparse.ArgumentParser(
    description="Compare le temps de coalescence entre Moran (evenements) et genealogie (random walk)"
)
parser.add_argument("--dossier_moran", type=str, required=True,
                    help="Dossier contenant les donnees_jeuX.csv (vrai Moran)")
parser.add_argument("--dossier_genealogie", type=str, required=True,
                    help="Dossier contenant les donnees_jeuX_nY.csv (genealogie)")
parser.add_argument("--n_echantillons", type=int, required=True,
                    help="Taille d'echantillon a comparer cote genealogie (doit matcher le nom de fichier)")
parser.add_argument("--sauvegarder", action="store_true")
args = parser.parse_args()

moran = charger_moran(args.dossier_moran)
genealogie = charger_genealogie(args.dossier_genealogie, args.n_echantillons)

l = int(moran["l"].iloc[0])
m = float(moran["m"].iloc[0])
lam = float(moran["lam"].iloc[0])
n = l * l
t_theorique = n / (m * lam)

print("\n=== Moran (evenements) ===")
print(f"  Lignes (coalescees) : {len(moran)}")
print(f"  Repetitions         : {moran['id_jeu'].nunique()}")
print(f"  t min/max           : {moran['t'].min():.5f} / {moran['t'].max():.5f}")
print(f"  t moyenne           : {moran['t'].mean():.5f}")
print(f"  t mediane           : {moran['t'].median():.5f}")

print("\n=== Genealogie (random walk) ===")
print(f"  Lignes              : {len(genealogie)}")
print(f"  Repetitions         : {genealogie['id_jeu'].nunique()}")
print(f"  t min/max           : {genealogie['t'].min():.5f} / {genealogie['t'].max():.5f}")
print(f"  t moyenne           : {genealogie['t'].mean():.5f}")
print(f"  t mediane           : {genealogie['t'].median():.5f}")

# KS garde dans la console pour reference, mais pas affiche sur la figure :
# avec plusieurs milliers de points, le p-value devient tres sensible et
# n'est plus tres parlant tout seul (cf. discussion precedente)
ks_stat, ks_pval = stats.ks_2samp(moran["t"].values, genealogie["t"].values)
print(f"\n=== KS test Moran vs genealogie ===")
print(f"  stat = {ks_stat:.5f}, p-value = {ks_pval:.5f}")

# figure : histogramme + QQ plot
# la courbe theorique est affichee volontairement, meme si elle sous-estime
# le temps de coalescence (ecart qui grandit avec l, deja vu avec Stephane) :
# elle sert a montrer l'ecart, pas a valider un bon fit
q99 = np.percentile(np.concatenate([moran["t"].values, genealogie["t"].values]), 99)
bins = np.linspace(0, q99, 50)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Moran vs genealogie - grille {l}x{l}, m={m}, n_echantillons={args.n_echantillons}",
             fontsize=11)

axes[0].hist(moran["t"], bins=bins, density=True, alpha=0.5,
             color="steelblue", label=f"Moran")
axes[0].hist(genealogie["t"], bins=bins, density=True, alpha=0.5,
             color="darkorange", label=f"Genealogie")
taux = 1.0 / t_theorique
x = np.linspace(0, q99, 300)
axes[0].plot(x, taux * np.exp(-taux * x), "k--", linewidth=1.5,
             label=f"Exponentielle theorique (moyenne={t_theorique:.0f})")
axes[0].set_xlabel("Temps de coalescence")
axes[0].set_ylabel("Densite")
axes[0].legend(fontsize=9)
axes[0].set_title("Histogrammes")

quantiles = np.linspace(0, 1, 101)[1:-1]
q_moran = np.quantile(moran["t"], quantiles)
q_genealogie = np.quantile(genealogie["t"], quantiles)
lim = max(q_moran.max(), q_genealogie.max())
axes[1].scatter(q_moran, q_genealogie, s=15, color="darkorange", alpha=0.7)
axes[1].plot([0, lim], [0, lim], "k--", linewidth=1)
axes[1].set_xlabel("Quantiles Moran")
axes[1].set_ylabel("Quantiles genealogie")
axes[1].set_title("QQ plot")

plt.tight_layout()
if args.sauvegarder:
    nom = f"comparaison_moran_genealogie_l{l}_m{m:.2f}_n{args.n_echantillons}.png"
    plt.savefig(nom, dpi=150)
    print(f"\nFigure sauvegardee : {nom}")
plt.show()