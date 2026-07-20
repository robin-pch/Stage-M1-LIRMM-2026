# -*- coding: utf-8 -*-
# Lance la chaine complete en local avec la methode par liste d'evenements.
# Pour chaque jeu : tire m, ecrit les evenements (ecrire_evenements.py),
# genere les paires (generation_donnees_evenements.py), calcule la
# vraisemblance (vraisemblance.py). Le fichier d'evenements est ecrase
# a chaque jeu, pas besoin de tous les garder.
# Robin Pioch, stage M1 LIRMM, juin 2026

# Usage : python lancer_generation_evenements_local.py --n_jeux 50 --l 20 --n_m 20

import argparse
import subprocess
import os
import sys
import numpy as np


dossier_scripts = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(
    description="Lance la chaine evenements (ecrire_evenements + generation + vraisemblance), n_jeux fois"
)
parser.add_argument("--n_jeux", type=int, default=50,
                    help="Nombre de jeux (defaut : 50)")
parser.add_argument("--l", type=int, default=20,
                    help="Cote de la grille (defaut : 20)")
parser.add_argument("--lam", type=float, default=1.0,
                    help="lambda (defaut : 1.0)")
parser.add_argument("--n_echantillons", type=int, default=50,
                    help="Nombre de cases tirees au present par jeu (defaut : 50)")
parser.add_argument("--n_m", type=int, default=20,
                    help="Nombre de valeurs de m testees en vraisemblance (defaut : 20)")
parser.add_argument("--cond", action="store_true",
                    help="Utilise proba_transition_exp_cond en vraisemblance")
parser.add_argument("--optimiser", action="store_true",
                    help="Utilise minimize_scalar (Brent) au lieu d'une grille de m")
parser.add_argument("--dossier_donnees", type=str, default="donnees",
                    help="Dossier de sortie pour les csv de donnees (defaut : donnees)")
parser.add_argument("--dossier_resultats", type=str, default="resultats",
                    help="Dossier de sortie pour les csv de resultats (defaut : resultats)")
parser.add_argument("--dossier_evenements", type=str, default="evenements",
                    help="Dossier temporaire pour le fichier d'evenements (defaut : evenements)")
args = parser.parse_args()

chemin_ecrire = os.path.join(dossier_scripts, "ecrire_evenements.py")
chemin_backward = os.path.join(dossier_scripts, "generation_donnees_evenements.py")
chemin_vraisemblance = os.path.join(dossier_scripts, "vraisemblance.py")

for id_jeu in range(args.n_jeux):
    print(f"=== jeu {id_jeu + 1}/{args.n_jeux} ===")

    # m tire entre 0.1 et 1.0 : les tres petits m donnent des T enormes
    # (les lignees se deplacent rarement, coalescence tres lente)
    m_vrai = np.random.uniform(0.1, 1.0)
    print(f"m = {m_vrai:.4f}")

    # fichier d'evenements temporaire, meme nom a chaque fois (ecrase a chaque jeu)
    nom_evenements = "evenements_tmp.csv"
    chemin_evenements = os.path.join(args.dossier_evenements, nom_evenements)

    subprocess.run([
        sys.executable, chemin_ecrire,
        "--l", str(args.l),
        "--m", str(m_vrai),
        "--lam", str(args.lam),
        "--dossier", args.dossier_evenements,
        "--nom", nom_evenements,
        "--sauvegarder",
    ])

    subprocess.run([
        sys.executable, chemin_backward,
        "--entree", chemin_evenements,
        "--l", str(args.l),
        "--m", str(m_vrai),
        "--lam", str(args.lam),
        "--n_echantillons", str(args.n_echantillons),
        "--id_jeu", str(id_jeu),
        "--dossier", args.dossier_donnees,
        "--sauvegarder",
    ])

    chemin_donnees = os.path.join(args.dossier_donnees, f"donnees_jeu{id_jeu}.csv")

    cmd_vrais = [
        sys.executable, chemin_vraisemblance,
        "--entree", chemin_donnees,
        "--n_m", str(args.n_m),
        "--dossier", args.dossier_resultats,
        "--sauvegarder",
    ]
    if args.cond:
        cmd_vrais.append("--cond")
    if args.optimiser:
        cmd_vrais.append("--optimiser")
    subprocess.run(cmd_vrais)

print(f"\n{args.n_jeux} jeux generes et traites.")
print(f"Donnees dans {args.dossier_donnees}/, resultats dans {args.dossier_resultats}/.")
print("Lancer graphique_vraisemblance.py pour voir le resultat final.")
