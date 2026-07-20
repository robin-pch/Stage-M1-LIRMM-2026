# -*- coding: utf-8 -*-
# Compare m reel et m estime (max de vraisemblance), tous jeux confondus.
# Lit donnees/ (m reel) et resultats/ (log-vraisemblance), trace m reel vs m estime.
# Robin Pioch, stage M1 LIRMM, juin 2026

# Usage : python graphique_vraisemblance.py --dossier_donnees donnees --dossier_resultats resultats --afficher

import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
import glob
import os
import sys


def lire_m_reel(chemin_csv):
    """Lit un fichier de donnees et renvoie (id_jeu, m_reel)."""
    with open(chemin_csv, newline="") as f:
        lecteur = csv.DictReader(f)
        for ligne in lecteur:
            return int(ligne["id_jeu"]), float(ligne["m"])
    return None, None


def lire_m_estime(chemin_csv):
    """Lit un fichier de resultats, renvoie (id_jeu, m qui maximise la log-vraisemblance)."""
    id_jeu = None
    m_vals = []
    log_vrais_vals = []

    with open(chemin_csv, newline="") as f:
        lecteur = csv.DictReader(f)
        for ligne in lecteur:
            id_jeu = int(ligne["id_jeu"])
            m_vals.append(float(ligne["m"]))
            log_vrais_vals.append(float(ligne["log_vraisemblance"]))

    m_vals = np.array(m_vals)
    log_vrais_vals = np.array(log_vrais_vals)
    m_estime = m_vals[np.argmax(log_vrais_vals)]

    return id_jeu, m_estime


# =============================================================================
# Lecture des arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Compare m reel et m estime par vraisemblance, tous jeux confondus"
)

parser.add_argument("--dossier_donnees", type=str, default="donnees",
                    help="Dossier contenant les csv de donnees (defaut : donnees)")

parser.add_argument("--dossier_resultats", type=str, default="resultats",
                    help="Dossier contenant les csv de resultats (defaut : resultats)")

parser.add_argument("--afficher", action="store_true",
                    help="Affiche le graphique")

parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde le graphique en .png")

args = parser.parse_args()


# =============================================================================
# Lancement
# =============================================================================

fichiers_donnees = glob.glob(os.path.join(args.dossier_donnees, "donnees_jeu*.csv"))
fichiers_resultats = glob.glob(os.path.join(args.dossier_resultats, "resultats_jeu*.csv"))

print(f"{len(fichiers_donnees)} fichiers de donnees trouves.")
print(f"{len(fichiers_resultats)} fichiers de resultats trouves.")

m_reel_par_jeu = {}
for chemin in fichiers_donnees:
    id_jeu, m_reel = lire_m_reel(chemin)
    if id_jeu is not None:
        m_reel_par_jeu[id_jeu] = m_reel

m_estime_par_jeu = {}
for chemin in fichiers_resultats:
    id_jeu, m_estime = lire_m_estime(chemin)
    if id_jeu is not None:
        m_estime_par_jeu[id_jeu] = m_estime

jeux_communs = sorted(set(m_reel_par_jeu.keys()) & set(m_estime_par_jeu.keys()))
print(f"{len(jeux_communs)} jeux communs aux deux dossiers.")

if len(jeux_communs) == 0:
    print("Aucun jeu commun, impossible de tracer le graphique.")
    sys.exit(0)

m_reel = np.array([m_reel_par_jeu[id_jeu] for id_jeu in jeux_communs])
m_estime = np.array([m_estime_par_jeu[id_jeu] for id_jeu in jeux_communs])

erreur_abs = np.abs(m_estime - m_reel)
print(f"Erreur absolue moyenne : {erreur_abs.mean():.4f}")

if not args.afficher and not args.sauvegarder:
    print("Rien a afficher (utiliser --afficher ou --sauvegarder).")
    sys.exit(0)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(m_reel, m_estime, s=20, alpha=0.7)
ax.plot([0, 1], [0, 1], color="gray", ls="--", label="y = x")
ax.set_xlabel("m reel")
ax.set_ylabel("m estime (max de vraisemblance)")
ax.set_title(f"m reel vs m estime - {len(jeux_communs)} jeux")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend()
plt.tight_layout()

if args.sauvegarder:
    nom = "comparaison_m_reel_estime.png"
    plt.savefig(nom, dpi=150)
    print(f"Graphique sauvegarde : {nom}")
if args.afficher:
    plt.show()
plt.close(fig)
