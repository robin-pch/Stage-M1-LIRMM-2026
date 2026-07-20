# -*- coding: utf-8 -*-
# Calcule la vraisemblance conjointe (m, l) sur le cluster IFB, un job SLURM
# par fichier de donnees. Meme principe que lancer_analyse_cluster.py, mais
# appelle vraisemblance_ml.py au lieu de vraisemblance.py.
# A lancer une fois que tous les jobs de generation sont finis
# (verifier avec squeue -u $USER, la liste doit etre vide).
# Robin Pioch, stage M1 LIRMM, juillet 2026

# Usage : python lancer_analyse_ml_cluster.py --n_m 20 --n_l 20 --l_min 5 --l_max 100

# =============================================================================

MODULE_PYTHON = "python/3.11.2"

PYTHON = "python3"

DOSSIER_SCRIPTS = "/shared/projects/spatial_moran_migration/vraisemblance"

# meme dossier de travail que la generation, pour retrouver donnees/ et y ajouter resultats/
DOSSIER_TRAVAIL = "/shared/projects/spatial_moran_migration/resultats_cluster"

COMPTE_SLURM = "spatial_moran_migration"

PARTITION = "fast"
MEM_MO   = 100000
TEMPS_MAX = "24:00:00"

# =============================================================================

import argparse
import subprocess
import os
import glob


parser = argparse.ArgumentParser(
    description="Soumet un job SLURM par fichier de donnees deja genere, pour l'estimation conjointe (m,l)"
)
parser.add_argument("--n_m", type=int, default=20,
                    help="Nombre de valeurs de m testees (defaut : 20)")
parser.add_argument("--n_l", type=int, default=20,
                    help="Nombre de valeurs de l testees (defaut : 20)")
parser.add_argument("--l_min", type=int, default=5,
                    help="Plus petite valeur de l testee (defaut : 5)")
parser.add_argument("--l_max", type=int, default=200,
                    help="Plus grande valeur de l testee (defaut : 200)")
parser.add_argument("--cond", action="store_true",
                    help="Utilise proba_transition_exp_cond en vraisemblance")
parser.add_argument("--optimiser", action="store_true",
                    help="Utilise minimize (Nelder-Mead) sur (m,l) au lieu d'une grille")
args = parser.parse_args()

dossier_donnees   = os.path.join(DOSSIER_TRAVAIL, "donnees")
dossier_resultats = os.path.join(DOSSIER_TRAVAIL, "resultats")
dossier_logs      = os.path.join(DOSSIER_TRAVAIL, "logs")

os.makedirs(dossier_resultats, exist_ok=True)

chemin_vraisemblance_ml = os.path.join(DOSSIER_SCRIPTS, "vraisemblance_ml.py")

options_vrais = ""
if args.cond:
    options_vrais += " --cond"
if args.optimiser:
    options_vrais += " --optimiser"

# on cherche tous les fichiers de donnees presents, pas juste un par id_jeu,
# puisqu'un meme jeu peut avoir plusieurs tailles d'echantillon
fichiers_donnees = sorted(glob.glob(os.path.join(dossier_donnees, "donnees_jeu*.csv")))

if len(fichiers_donnees) == 0:
    print(f"Aucun fichier de donnees trouve dans {dossier_donnees}/")

n_soumis = 0

for chemin_donnees_jeu in fichiers_donnees:

    # nom_base sert a nommer le job et les logs, ex: "jeu3_n50"
    nom_fichier = os.path.splitext(os.path.basename(chemin_donnees_jeu))[0]
    nom_base = nom_fichier.replace("donnees_", "", 1)

    script_slurm = f"""#!/bin/bash
#SBATCH --job-name=anml_{nom_base}
#SBATCH --partition={PARTITION}
#SBATCH --account={COMPTE_SLURM}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem={MEM_MO}
#SBATCH --time={TEMPS_MAX}
#SBATCH --output={dossier_logs}/anml_{nom_base}_%j.out
#SBATCH --error={dossier_logs}/anml_{nom_base}_%j.err

module load {MODULE_PYTHON}

{PYTHON} {chemin_vraisemblance_ml} \\
    --entree {chemin_donnees_jeu} \\
    --n_m {args.n_m} --n_l {args.n_l} \\
    --l_min {args.l_min} --l_max {args.l_max} \\
    --dossier {dossier_resultats} \\
    --sauvegarder{options_vrais}
"""

    chemin_sh = os.path.join(dossier_logs, f"job_anml_{nom_base}.sh")
    with open(chemin_sh, "w") as f:
        f.write(script_slurm)

    subprocess.run(["sbatch", chemin_sh])
    print(f"analyse m,l {nom_base} soumise")
    n_soumis += 1

print(f"\n{n_soumis} jobs d'analyse soumis.")
print(f"Suivre avec : squeue -u $USER")
print(f"Resultats dans : {dossier_resultats}/")
