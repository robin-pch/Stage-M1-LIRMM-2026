# -*- coding: utf-8 -*-
# Genere des jeux de donnees en local (sans cluster) avec la generation
# directe par genealogie (generation_donnees_genealogie.py). Cette methode
# est bien moins couteuse que celle par liste d'evenements, pas besoin du
# cluster pour ca.
# Meme logique min/max que lancer_generation_genealogie_cluster.py : pour
# fixer m ou l a une valeur precise, mettre min = max.
# Robin Pioch, stage M1 LIRMM, juillet 2026

# Usage : python lancer_generation_genealogie_local.py --n_jeux 100 --l_min 50 --l_max 50 --m_min 1.0 --m_max 1.0 --echantillons 10

import argparse
import subprocess
import os
import sys
import csv
import numpy as np


# dossier de ce script (= dossier de generation_donnees_genealogie.py)
dossier_scripts = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(
    description="Lance generation_donnees_genealogie.py n_jeux fois, en local"
)
parser.add_argument("--n_jeux", type=int, default=100,
                    help="Nombre de jeux a generer (defaut : 100)")
parser.add_argument("--m_min", type=float, default=0.1,
                    help="Plus petite valeur de m tiree au hasard (defaut : 0.1)")
parser.add_argument("--m_max", type=float, default=1.0,
                    help="Plus grande valeur de m tiree au hasard (defaut : 1.0)")
parser.add_argument("--l_min", type=int, default=5,
                    help="Plus petite valeur de l tiree au hasard (defaut : 5)")
parser.add_argument("--l_max", type=int, default=100,
                    help="Plus grande valeur de l tiree au hasard (defaut : 100)")
parser.add_argument("--lam", type=float, default=1.0,
                    help="lambda (defaut : 1.0)")
parser.add_argument("--echantillons", type=str, default="10,20,50,100",
                    help="Tailles d'echantillon a generer pour chaque jeu, "
                         "separees par des virgules (defaut : 10,20,50,100).")
parser.add_argument("--schema", type=str, default="uniforme", choices=["uniforme", "cercle", "diagonale"],
                    help="Schema d'echantillonnage des positions de depart (defaut : uniforme)")
parser.add_argument("--sigma", type=float, default=1.0,
                    help="Largeur de bande pour le schema diagonale (defaut : 1.0)")
parser.add_argument("--rayon", type=float, default=None,
                    help="Rayon pour le schema cercle (defaut : l/4)")
parser.add_argument("--dossier_donnees", type=str, default="donnees_genealogie",
                    help="Dossier de sortie pour les csv de donnees (defaut : donnees_genealogie)")
args = parser.parse_args()

liste_echantillons = [int(x) for x in args.echantillons.split(",")]

os.makedirs(args.dossier_donnees, exist_ok=True)

chemin_genealogie = os.path.join(dossier_scripts, "generation_donnees_genealogie.py")

# trace des m et l tires pour chaque jeu, pratique pour verifier apres coup
chemin_recap = os.path.join(args.dossier_donnees, "recap_m_generation.csv")
fichier_recap = open(chemin_recap, "w", newline="")
ecrivain_recap = csv.writer(fichier_recap)
ecrivain_recap.writerow(["id_jeu", "m_vrai", "l_vrai"])

for id_jeu in range(args.n_jeux):
    print(f"=== jeu {id_jeu + 1}/{args.n_jeux} ===")

    # chaque jeu a son propre m et son propre l (si min = max, valeur fixe)
    m_vrai = np.random.uniform(args.m_min, args.m_max)
    l_vrai = np.random.randint(args.l_min, args.l_max + 1)
    ecrivain_recap.writerow([id_jeu, m_vrai, l_vrai])
    print(f"m={m_vrai:.4f}, l={l_vrai}")

    for n_ech in liste_echantillons:
        cmd = [
            sys.executable, chemin_genealogie,
            "--l", str(l_vrai),
            "--m", str(m_vrai),
            "--lam", str(args.lam),
            "--n_echantillons", str(n_ech),
            "--id_jeu", str(id_jeu),
            "--schema", args.schema,
            "--sigma", str(args.sigma),
            "--dossier", args.dossier_donnees,
            "--sauvegarder",
        ]
        if args.rayon is not None:
            cmd += ["--rayon", str(args.rayon)]
        subprocess.run(cmd)

fichier_recap.close()

print(f"\n{args.n_jeux} jeux generes.")
print(f"Donnees dans {args.dossier_donnees}/")
print(f"Recap des m et l tires : {chemin_recap}")
