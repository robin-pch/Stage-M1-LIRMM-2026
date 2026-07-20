# -*- coding: utf-8 -*-
# Genere les donnees sur le cluster IFB avec la methode par liste d'evenements.
# Chaque tache fait : ecrire_evenements -> generation_donnees_evenements
# (pour chaque taille d'echantillon demandee) -> suppression des evenements.
# C'est la "vraie" methode Moran (evenements sur toute la grille), a ne pas
# confondre avec generation_donnees_genealogie.py qui est l'approximation
# par random walk.
#
# Les taches tournent en job array SLURM throttle (--array=0-N%CONCURRENCE)
# pour ne pas avoir trop de fichiers d'evenements en meme temps sur le
# disque (chaque fichier fait ~1 Go pour l=50, si 100 tournent en meme
# temps ca sature vite le quota).
#
# Ne relance que les jeux dont le csv de donnees n'existe pas encore, donc
# on peut relancer ce script plusieurs fois sans repeter le travail deja fait.
# Robin Pioch, stage M1 LIRMM, juillet 2026

# Usage : python lancer_generation_evenements_cluster.py --n_jeux 100 --l_min 50 --l_max 50 --m_min 1.0 --m_max 1.0 --echantillons 50
# Pour fixer m ou l a une valeur precise, mettre min = max (ex: --m_min 1.0 --m_max 1.0)

# =============================================================================

MODULE_PYTHON = "python/3.11.2"

# Pas de venv pour l'instant, on utilise le python3 du module charge au dessus
PYTHON = "python3"

# Dossier contenant ecrire_evenements.py et generation_donnees_evenements.py
DOSSIER_SCRIPTS = "/shared/projects/spatial_moran_migration/vraisemblance"

# Dossier de travail : c'est la que seront crees donnees/, evenements/, logs/
DOSSIER_TRAVAIL = "/shared/projects/spatial_moran_migration/resultats_cluster"

# Compte SLURM (nom du projet IFB)
COMPTE_SLURM = "spatial_moran_migration"

# Parametres SLURM. La generation des evenements est bien plus lourde que la
# generation directe par genealogie (fichier d'evenements sur toute la
# grille), donc on garde de la memoire et du temps large.
PARTITION = "fast"
MEM_MO   = 12000
TEMPS_MAX = "24:00:00"

# =============================================================================

import argparse
import subprocess
import os
import csv
import numpy as np


parser = argparse.ArgumentParser(
    description="Soumet les jeux manquants comme un job array SLURM throttle (vrai Moran, liste d'evenements)"
)
parser.add_argument("--n_jeux", type=int, default=100,
                    help="Nombre total de jeux vises (defaut : 100)")
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
parser.add_argument("--echantillons", type=str, default="50",
                    help="Tailles d'echantillon a generer pour chaque jeu, "
                         "separees par des virgules (defaut : 50). Attention : "
                         "generation_donnees_evenements.py ecrit toujours donnees_jeuX.csv, "
                         "sans le n_echantillons dans le nom. Avec plusieurs tailles, "
                         "chaque appel ecrase le csv du precedent pour le meme jeu.")
parser.add_argument("--concurrence", type=int, default=10,
                    help="Nombre max de taches qui tournent en meme temps (defaut : 10). "
                         "A baisser si le quota disque sature, a monter si on a de la marge.")
args = parser.parse_args()

liste_echantillons = [int(x) for x in args.echantillons.split(",")]

dossier_donnees    = os.path.join(DOSSIER_TRAVAIL, "donnees")
dossier_evenements = os.path.join(DOSSIER_TRAVAIL, "evenements")
dossier_logs       = os.path.join(DOSSIER_TRAVAIL, "logs")

for d in [dossier_donnees, dossier_evenements, dossier_logs]:
    os.makedirs(d, exist_ok=True)

chemin_ecrire   = os.path.join(DOSSIER_SCRIPTS, "ecrire_evenements.py")
chemin_backward = os.path.join(DOSSIER_SCRIPTS, "generation_donnees_evenements.py")

# on ne relance que les jeux dont le csv de donnees n'existe pas encore
ids_manquants = []
for id_jeu in range(args.n_jeux):
    chemin_attendu = os.path.join(dossier_donnees, f"donnees_jeu{id_jeu}.csv")
    if not os.path.exists(chemin_attendu):
        ids_manquants.append(id_jeu)

print(f"{args.n_jeux - len(ids_manquants)} jeux deja termines.")
print(f"{len(ids_manquants)} jeux a lancer.")

if len(ids_manquants) == 0:
    print("Rien a faire, tous les jeux sont deja generes.")
    raise SystemExit(0)

# on tire m et l pour chaque jeu manquant, et on les ecrit dans un fichier
# recap que chaque tache du job array ira relire pour connaitre ses valeurs
# (si m_min = m_max, ou l_min = l_max, la valeur est fixe pour tous les jeux)
chemin_recap = os.path.join(dossier_logs, "recap_m_generation.csv")
fichier_recap = open(chemin_recap, "w", newline="")
ecrivain_recap = csv.writer(fichier_recap)
ecrivain_recap.writerow(["id_jeu", "m_vrai", "l_vrai"])

for id_jeu in ids_manquants:
    m_vrai = np.random.uniform(args.m_min, args.m_max)
    l_vrai = np.random.randint(args.l_min, args.l_max + 1)
    ecrivain_recap.writerow([id_jeu, m_vrai, l_vrai])

fichier_recap.close()

# un appel a generation_donnees_evenements.py par taille d'echantillon,
# tous reutilisant le meme fichier d'evenements
lignes_backward = ""
for n_ech in liste_echantillons:
    lignes_backward += f"""
{PYTHON} {chemin_backward} \\
    --entree ${{CHEMIN_EVENEMENTS}} \\
    --l ${{L_VRAI}} --m ${{M_VRAI}} --lam {args.lam} \\
    --n_echantillons {n_ech} --id_jeu ${{ID_JEU}} \\
    --dossier {dossier_donnees} --sauvegarder
"""

dernier_index = len(ids_manquants) - 1

script_slurm = f"""#!/bin/bash
#SBATCH --job-name=gen_moran
#SBATCH --partition={PARTITION}
#SBATCH --account={COMPTE_SLURM}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH -c 1
#SBATCH --mem={MEM_MO}
#SBATCH --time={TEMPS_MAX}
#SBATCH --array=0-{dernier_index}%{args.concurrence}
#SBATCH --output={dossier_logs}/gen_%a_%A.out
#SBATCH --error={dossier_logs}/gen_%a_%A.err

module load {MODULE_PYTHON}

# la tache numero SLURM_ARRAY_TASK_ID va lire sa ligne dans le recap
# (+2 car sed compte a partir de 1, et la ligne 1 est l'en-tete)
LIGNE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" {chemin_recap})
IFS=',' read ID_JEU M_VRAI L_VRAI <<< "$LIGNE"

NOM_EVENEMENTS=evenements_jeu${{ID_JEU}}.csv
CHEMIN_EVENEMENTS={dossier_evenements}/${{NOM_EVENEMENTS}}

{PYTHON} {chemin_ecrire} \\
    --l ${{L_VRAI}} --m ${{M_VRAI}} --lam {args.lam} \\
    --dossier {dossier_evenements} --nom ${{NOM_EVENEMENTS}} \\
    --sauvegarder
{lignes_backward}
# le fichier d'evenements est supprime tout de suite, c'est lui qui prend toute la place
rm -f ${{CHEMIN_EVENEMENTS}}
"""

chemin_sh = os.path.join(dossier_logs, "job_gen_moran.sh")
with open(chemin_sh, "w") as f:
    f.write(script_slurm)

subprocess.run(["sbatch", chemin_sh])

print(f"\nJob array soumis : {len(ids_manquants)} taches, {args.concurrence} en parallele max.")
print(f"Suivre avec : squeue -u $USER")
print(f"Recap des m et l tires : {chemin_recap}")
print(f"Espace disque a surveiller avec : du -sh {DOSSIER_TRAVAIL}")
