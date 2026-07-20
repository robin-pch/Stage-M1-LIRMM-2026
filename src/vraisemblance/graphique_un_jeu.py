# -*- coding: utf-8 -*-
# Trace le profil de vraisemblance d'un seul jeu (sortie de vraisemblance.py).
# A utiliser apres la generation des donnees et le calcul de vraisemblance sur un jeu.
# Robin Pioch, stage M1 LIRMM, juin 2026

# Usage : python graphique_un_jeu.py --donnees donnees/donnees_jeu0.csv --resultats resultats/resultats_jeu0.csv --afficher

import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv


parser = argparse.ArgumentParser(
    description="Profil de vraisemblance pour un seul jeu"
)

parser.add_argument("--donnees", type=str, required=True,
                    help="Csv de donnees du jeu (pour recuperer m_reel)")

parser.add_argument("--resultats", type=str, required=True,
                    help="Csv de resultats du jeu (m teste, log_vraisemblance)")

parser.add_argument("--afficher", action="store_true", help="Affiche le graphique")
parser.add_argument("--sauvegarder", action="store_true", help="Sauvegarde en .png")

args = parser.parse_args()

# m_reel : premiere ligne du csv de donnees
with open(args.donnees, newline="") as f:
    lecteur = csv.DictReader(f)
    ligne = next(lecteur)
    m_reel = float(ligne["m"])
    l = int(ligne["l"])

# courbe de vraisemblance : tout le csv de resultats
m_vals = []
log_vrais_vals = []
with open(args.resultats, newline="") as f:
    lecteur = csv.DictReader(f)
    for ligne in lecteur:
        m_vals.append(float(ligne["m"]))
        log_vrais_vals.append(float(ligne["log_vraisemblance"]))

m_vals = np.array(m_vals)
log_vrais_vals = np.array(log_vrais_vals)
m_estime = m_vals[np.argmax(log_vrais_vals)]

print(f"m_reel = {m_reel:.4f}")
print(f"m_estime (max de vraisemblance) = {m_estime:.4f}")

if not args.afficher and not args.sauvegarder:
    print("Rien a afficher (utiliser --afficher ou --sauvegarder).")
else:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(m_vals, log_vrais_vals, marker="o", markersize=3)
    ax.axvline(m_reel, color="darkgreen", ls="--", label=f"m_reel = {m_reel:.3f}")
    ax.axvline(m_estime, color="darkorange", ls=":", label=f"m_estime = {m_estime:.3f}")
    ax.set_xlabel("m")
    ax.set_ylabel("Log-vraisemblance")
    ax.set_title(f"Profil de vraisemblance - grille {l}x{l}")
    ax.legend()
    plt.tight_layout()

    if args.sauvegarder:
        nom = f"profil_vraisemblance_mreel{m_reel:.2f}.png"
        plt.savefig(nom, dpi=150)
        print(f"Graphique sauvegarde : {nom}")
    if args.afficher:
        plt.show()
    plt.close(fig)
