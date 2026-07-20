# -*- coding: utf-8 -*-
# Compare (m,l) reel et (m,l) estime (max de vraisemblance), un jeu de
# graphiques par taille d'echantillon :
#   - un graphique (m,l) avec une fleche du point reel vers le point estime
#   - un graphique m seul (reel vs estime, axe y=x)
#   - un graphique l seul (reel vs estime, axe y=x)
# Robin Pioch, stage M1 LIRMM, juillet 2026

# Usage : python graphique_vraisemblance_ml.py --dossier_donnees donnees --dossier_resultats resultats --afficher

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import argparse
import csv
import glob
import os
import sys


def lire_reel(chemin_csv):
    """Renvoie (cle, m_reel, l_reel). cle = nom de fichier sans le prefixe."""
    nom_fichier = os.path.splitext(os.path.basename(chemin_csv))[0]
    cle = nom_fichier.replace("donnees_", "", 1)

    with open(chemin_csv, newline="") as f:
        lecteur = csv.DictReader(f)
        for ligne in lecteur:
            return cle, float(ligne["m"]), int(ligne["l"])

    return cle, None, None


def lire_estime(chemin_csv):
    """Renvoie (cle, m_estime, l_estime), le couple qui maximise la log-vraisemblance."""
    nom_fichier = os.path.splitext(os.path.basename(chemin_csv))[0]
    cle = nom_fichier.replace("resultats_ml_", "", 1)

    # retire les suffixes connus ajoutes par vraisemblance_ml.py (dans
    # n'importe quel ordre/combinaison), pour que la cle corresponde a
    # celle du fichier de donnees (qui n'en a pas)
    change = True
    while change:
        change = False
        for suffixe in ["_mfixe", "_coal", "_ech"]:
            if cle.endswith(suffixe):
                cle = cle[: -len(suffixe)]
                change = True

    m_vals, l_vals, log_vrais_vals = [], [], []
    with open(chemin_csv, newline="") as f:
        lecteur = csv.DictReader(f)
        for ligne in lecteur:
            m_vals.append(float(ligne["m"]))
            l_vals.append(float(ligne["l"]))
            log_vrais_vals.append(float(ligne["log_vraisemblance"]))

    idx_max = np.argmax(log_vrais_vals)
    return cle, m_vals[idx_max], l_vals[idx_max]


def taille_echantillon(cle):
    """Extrait la taille d'echantillon depuis une cle du type 'jeu5_n50' -> '50'."""
    parts = cle.rsplit("_n", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[1]
    return "inconnu"


def tracer_ml(cles, reel_par_cle, estime_par_cle, taille, l_min_global, l_max_global, args):
    """Graphique (m,l) avec une fleche du point reel vers le point estime."""
    m_reel = np.array([reel_par_cle[cle][0] for cle in cles])
    l_reel = np.array([reel_par_cle[cle][1] for cle in cles])
    m_estime = np.array([estime_par_cle[cle][0] for cle in cles])
    l_estime = np.array([estime_par_cle[cle][1] for cle in cles])

    longueurs = np.sqrt((m_estime - m_reel)**2 + (l_estime - l_reel)**2)
    print(f"n={taille} : {len(cles)} jeux, longueur de trait moyenne = {longueurs.mean():.2f}")

    fig, ax = plt.subplots(figsize=(8, 7))

    cmap = plt.colormaps["plasma"]
    norm = mcolors.Normalize(vmin=longueurs.min(), vmax=longueurs.max())

    for i in range(len(cles)):
        couleur = cmap(norm(longueurs[i]))
        ax.annotate(
            "", xy=(m_estime[i], l_estime[i]), xytext=(m_reel[i], l_reel[i]),
            arrowprops=dict(arrowstyle="->", color=couleur, alpha=0.8, lw=1.2)
        )

    ax.scatter(m_reel, l_reel, marker="x", color="black", s=15, label="reel", zorder=3)
    ax.scatter(m_estime, l_estime, marker="o", facecolors="none", edgecolors="black", s=15, label="estime", zorder=3)

    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm, ax=ax, label="longueur du trait (erreur)")

    ax.set_xlim(0, 1)
    ax.set_ylim(l_min_global, l_max_global)
    ax.set_xlabel("m")
    ax.set_ylabel("l")
    titre_suffixe = f" ({args.suffixe})" if args.suffixe else ""
    ax.set_title(f"(m,l) reel vers (m,l) estime - n={taille} - {len(cles)} jeux{titre_suffixe}")
    ax.legend()
    plt.tight_layout()

    if args.sauvegarder:
        suffixe_str = f"_{args.suffixe}" if args.suffixe else ""
        nom = f"comparaison_ml_reel_estime_n{taille}{suffixe_str}.png"
        plt.savefig(nom, dpi=150)
        print(f"Graphique sauvegarde : {nom}")
    if args.afficher:
        plt.show()
    plt.close(fig)


def tracer_simple(valeurs_reel, valeurs_estime, nom_param, taille, args):
    """Graphique reel vs estime pour un seul parametre (m ou l), avec la droite y=x."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(valeurs_reel, valeurs_estime, alpha=0.7)

    lim_min = min(valeurs_reel.min(), valeurs_estime.min())
    lim_max = max(valeurs_reel.max(), valeurs_estime.max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "--", color="gray", label="y = x")

    ax.set_xlabel(f"{nom_param} reel")
    ax.set_ylabel(f"{nom_param} estime")
    titre_suffixe = f" ({args.suffixe})" if args.suffixe else ""
    ax.set_title(f"{nom_param} reel vs estime - n={taille} - {len(valeurs_reel)} jeux{titre_suffixe}")
    ax.legend()
    plt.tight_layout()

    if args.sauvegarder:
        suffixe_str = f"_{args.suffixe}" if args.suffixe else ""
        nom = f"comparaison_{nom_param}_reel_estime_n{taille}{suffixe_str}.png"
        plt.savefig(nom, dpi=150)
        print(f"Graphique sauvegarde : {nom}")
    if args.afficher:
        plt.show()
    plt.close(fig)


# =============================================================================
# Lecture des arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Compare (m,l) reel et (m,l) estime par vraisemblance, un jeu de graphiques par taille d'echantillon"
)
parser.add_argument("--dossier_donnees", type=str, default="donnees",
                    help="Dossier contenant les csv de donnees (defaut : donnees)")
parser.add_argument("--dossier_resultats", type=str, default="resultats",
                    help="Dossier contenant les csv de resultats ml (defaut : resultats)")
parser.add_argument("--afficher", action="store_true",
                    help="Affiche les graphiques")
parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde les graphiques en .png")
parser.add_argument("--suffixe", type=str, default="",
                    help="Suffixe ajoute aux noms de fichiers sauvegardes, pour ne pas "
                         "ecraser d'autres runs (ex: --suffixe optimise, --suffixe cercle_cond)")
args = parser.parse_args()


# =============================================================================
# Lancement
# =============================================================================

fichiers_donnees = glob.glob(os.path.join(args.dossier_donnees, "donnees_jeu*.csv"))
fichiers_resultats = glob.glob(os.path.join(args.dossier_resultats, "resultats_ml_jeu*.csv"))

print(f"{len(fichiers_donnees)} fichiers de donnees trouves.")
print(f"{len(fichiers_resultats)} fichiers de resultats trouves.")

reel_par_cle = {}
for chemin in fichiers_donnees:
    cle, m_reel, l_reel = lire_reel(chemin)
    if m_reel is not None:
        reel_par_cle[cle] = (m_reel, l_reel)

estime_par_cle = {}
for chemin in fichiers_resultats:
    cle, m_estime, l_estime = lire_estime(chemin)
    estime_par_cle[cle] = (m_estime, l_estime)

cles_communes = sorted(set(reel_par_cle.keys()) & set(estime_par_cle.keys()))
print(f"{len(cles_communes)} jeux communs aux deux dossiers.")

if len(cles_communes) == 0:
    print("Aucun jeu commun, impossible de tracer un graphique.")
    sys.exit(0)

if not args.afficher and not args.sauvegarder:
    print("Rien a afficher (utiliser --afficher ou --sauvegarder).")
    sys.exit(0)

# bornes de l sur tout le jeu de donnees, pour que l'axe l ne soit pas
# recadre sur juste ce qui est affiche
toutes_valeurs_l = [l for (_, l) in reel_par_cle.values()]
l_min_global = min(toutes_valeurs_l)
l_max_global = max(toutes_valeurs_l)

# on regroupe les jeux par taille d'echantillon, un jeu de graphiques par groupe
cles_par_taille = {}
for cle in cles_communes:
    taille = taille_echantillon(cle)
    cles_par_taille.setdefault(taille, []).append(cle)

for taille in sorted(cles_par_taille.keys()):
    cles = cles_par_taille[taille]

    tracer_ml(cles, reel_par_cle, estime_par_cle, taille, l_min_global, l_max_global, args)

    m_reel = np.array([reel_par_cle[cle][0] for cle in cles])
    l_reel = np.array([reel_par_cle[cle][1] for cle in cles])
    m_estime = np.array([estime_par_cle[cle][0] for cle in cles])
    l_estime = np.array([estime_par_cle[cle][1] for cle in cles])

    tracer_simple(m_reel, m_estime, "m", taille, args)
    tracer_simple(l_reel, l_estime, "l", taille, args)
