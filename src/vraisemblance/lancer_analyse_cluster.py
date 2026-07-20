# -*- coding: utf-8 -*-
# Calcule la vraisemblance sur le cluster IFB (un job SLURM par fichier de donnees).
# A lancer une fois que tous les jobs de generation sont finis
# (verifier avec squeue -u $USER, la liste doit etre vide).
# Un jeu peut avoir plusieurs fichiers de donnees (une taille d'echantillon
# differente chacun), on les traite tous, pas juste un par jeu.
# Robin Pioch, stage M1 LIRMM, juillet 2026

# Usage : python lancer_analyse_cluster.py --n_m 20

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
    description="Soumet un job SLURM par fichier de donnees deja genere"
)
parser.add_argument("--n_m", type=int, default=20,
                    help="Nombre de valeurs de m testees (defaut : 20)")
parser.add_argument("--cond", action="store_true",
                    help="Utilise proba_transition_exp_cond en vraisemblance")
parser.add_argument("--optimiser", action="store_true",
                    help="Utilise minimize_scalar (Brent) au lieu d'une grille de m")
args = parser.parse_args()

dossier_donnees   = os.path.join(DOSSIER_TRAVAIL, "donnees")
dossier_resultats = os.path.join(DOSSIER_TRAVAIL, "resultats")
dossier_logs      = os.path.join(DOSSIER_TRAVAIL, "logs")

os.makedirs(dossier_resultats, exist_ok=True)

chemin_vraisemblance = os.path.join(DOSSIER_SCRIPTS, "vraisemblance.py")

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
#SBATCH --job-name=an_{nom_base}
#SBATCH --partition={PARTITION}
#SBATCH --account={COMPTE_SLURM}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem={MEM_MO}
#SBATCH --time={TEMPS_MAX}
#SBATCH --output={dossier_logs}/an_{nom_base}_%j.out
#SBATCH --error={dossier_logs}/an_{nom_base}_%j.err

module load {MODULE_PYTHON}

{PYTHON} {chemin_vraisemblance} \\
    --entree {chemin_donnees_jeu} \\
    --n_m {args.n_m} \\
    --dossier {dossier_resultats} \\
    --sauvegarder{options_vrais}
"""

    chemin_sh = os.path.join(dossier_logs, f"job_an_{nom_base}.sh")
    with open(chemin_sh, "w") as f:
        f.write(script_slurm)

    subprocess.run(["sbatch", chemin_sh])
    print(f"analyse {nom_base} soumise")
    n_soumis += 1

print(f"\n{n_soumis} jobs d'analyse soumis.")
print(f"Suivre avec : squeue -u $USER")
print(f"Resultats dans : {dossier_resultats}/")
