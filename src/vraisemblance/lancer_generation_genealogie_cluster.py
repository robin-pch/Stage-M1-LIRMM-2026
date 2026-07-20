# -*- coding: utf-8 -*-
# Genere les donnees sur le cluster IFB (un job SLURM par jeu). Chaque job
# simule directement la genealogie des echantillons
# (generation_donnees_genealogie.py), sans fichier d'evenements intermediaire.
# Robin Pioch, stage M1 LIRMM, juillet 2026

# Usage : python lancer_generation_genealogie_cluster.py --n_jeux 100 --l_min 10 --l_max 100 --echantillons 10,20,50,100
# Pour fixer m ou l a une valeur precise, mettre min = max (ex: --m_min 0.5 --m_max 0.5)
# Pour un schema d'echantillonnage different de uniforme : --schema cercle --rayon 20 (ou --schema diagonale --sigma 2)

# =============================================================================

MODULE_PYTHON = "python/3.11.2"

# Pas de venv pour l'instant, on utilise le python3 du module charge au dessus
PYTHON = "python3"

# Dossier contenant generation_donnees_genealogie.py
DOSSIER_SCRIPTS = "/shared/projects/spatial_moran_migration/vraisemblance"

# Dossier de travail : c'est la que seront crees les csv de donnees
DOSSIER_TRAVAIL = "/shared/projects/spatial_moran_migration/resultats_cluster"

# Compte SLURM (nom du projet IFB)
COMPTE_SLURM = "spatial_moran_migration"

# Parametres SLURM. La simulation directe par genealogie est bien plus
# legere que la methode par liste d'evenements, pas besoin de grosse
# memoire ni de temps long ici.
PARTITION = "fast"
MEM_MO   = 4000
TEMPS_MAX = "01:00:00"

# =============================================================================

import argparse
import subprocess
import os
import csv
import numpy as np


parser = argparse.ArgumentParser(
    description="Soumet n_jeux jobs SLURM qui generent chacun un jeu de donnees (genealogie)"
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
args = parser.parse_args()

liste_echantillons = [int(x) for x in args.echantillons.split(",")]

dossier_donnees = os.path.join(DOSSIER_TRAVAIL, "donnees")
dossier_logs    = os.path.join(DOSSIER_TRAVAIL, "logs")

for d in [dossier_donnees, dossier_logs]:
    os.makedirs(d, exist_ok=True)

chemin_genealogie = os.path.join(DOSSIER_SCRIPTS, "generation_donnees_genealogie.py")

# trace des m et l tires pour chaque jeu, pratique pour verifier apres coup
chemin_recap = os.path.join(dossier_logs, "recap_m_generation.csv")
fichier_recap = open(chemin_recap, "w", newline="")
ecrivain_recap = csv.writer(fichier_recap)
ecrivain_recap.writerow(["id_jeu", "m_vrai", "l_vrai"])

for id_jeu in range(args.n_jeux):

    # chaque jeu a son propre m et son propre l, tires ici (pas dans le job SLURM)
    m_vrai = np.random.uniform(args.m_min, args.m_max)
    l_vrai = np.random.randint(args.l_min, args.l_max + 1)
    ecrivain_recap.writerow([id_jeu, m_vrai, l_vrai])

    # un appel a generation_donnees_genealogie.py par taille d'echantillon
    option_rayon = f"--rayon {args.rayon} " if args.rayon is not None else ""
    lignes_generation = ""
    for n_ech in liste_echantillons:
        lignes_generation += f"""
{PYTHON} {chemin_genealogie} \\
    --l {l_vrai} --m {m_vrai} --lam {args.lam} \\
    --n_echantillons {n_ech} --id_jeu {id_jeu} \\
    --schema {args.schema} --sigma {args.sigma} {option_rayon}\\
    --dossier {dossier_donnees} --sauvegarder
"""

    script_slurm = f"""#!/bin/bash
#SBATCH --job-name=gen_jeu{id_jeu}
#SBATCH --partition={PARTITION}
#SBATCH --account={COMPTE_SLURM}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem={MEM_MO}
#SBATCH --time={TEMPS_MAX}
#SBATCH --output={dossier_logs}/gen_jeu{id_jeu}_%j.out
#SBATCH --error={dossier_logs}/gen_jeu{id_jeu}_%j.err

module load {MODULE_PYTHON}
{lignes_generation}
"""

    chemin_sh = os.path.join(dossier_logs, f"job_gen_jeu{id_jeu}.sh")
    with open(chemin_sh, "w") as f:
        f.write(script_slurm)

    subprocess.run(["sbatch", chemin_sh])
    print(f"generation jeu {id_jeu} soumise (m={m_vrai:.4f}, l={l_vrai})")

fichier_recap.close()

print(f"\n{args.n_jeux} jobs de generation soumis.")
print(f"Suivre avec : squeue -u $USER")
print(f"Recap des m et l tires : {chemin_recap}")
print(f"Une fois tous les jobs termines (squeue ne montre plus rien), lancer lancer_analyse_cluster.py")
