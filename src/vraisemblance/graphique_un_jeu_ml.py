# -*- coding: utf-8 -*-
# Trace le profil de vraisemblance 2D (m, l) d'un seul jeu (sortie de vraisemblance_ml.py).
# Meme principe que graphique_un_jeu.py, mais avec une carte de chaleur puisqu'on
# a deux parametres au lieu d'un seul.
# Robin Pioch, stage M1 LIRMM, juillet 2026

# Usage : python graphique_un_jeu_ml.py --donnees donnees/donnees_jeu0.csv --resultats resultats/resultats_ml_jeu0.csv --afficher

import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv


parser = argparse.ArgumentParser(
    description="Profil de vraisemblance 2D (m, l) pour un seul jeu"
)

parser.add_argument("--donnees", type=str, required=True,
                    help="Csv de donnees du jeu (pour recuperer m_reel et l_reel)")

parser.add_argument("--resultats", type=str, required=True,
                    help="Csv de resultats du jeu (m, l testes, log_vraisemblance)")

parser.add_argument("--afficher", action="store_true", help="Affiche le graphique")
parser.add_argument("--sauvegarder", action="store_true", help="Sauvegarde en .png")

args = parser.parse_args()

# m_reel et l_reel : premiere ligne du csv de donnees
with open(args.donnees, newline="") as f:
    lecteur = csv.DictReader(f)
    ligne = next(lecteur)
    m_reel = float(ligne["m"])
    l_reel = int(ligne["l"])

# lecture du csv de resultats : une ligne par couple (m, l) teste
m_vals = []
l_vals = []
log_vrais_vals = []
with open(args.resultats, newline="") as f:
    lecteur = csv.DictReader(f)
    for ligne in lecteur:
        m_vals.append(float(ligne["m"]))
        l_vals.append(float(ligne["l"]))
        log_vrais_vals.append(float(ligne["log_vraisemblance"]))

m_vals = np.array(m_vals)
l_vals = np.array(l_vals)
log_vrais_vals = np.array(log_vrais_vals)

# les tres petits m avec un grand l donnent des vraisemblances tellement
# minuscules que ca finit en -inf (underflow numerique). matplotlib ne sait
# pas colorer -inf, donc on remplace ces valeurs par un plancher, juste en
# dessous de la plus petite valeur finie du tableau
valeurs_finies = log_vrais_vals[np.isfinite(log_vrais_vals)]
plancher = valeurs_finies.min() - 1
log_vrais_vals_affichage = np.where(np.isfinite(log_vrais_vals), log_vrais_vals, plancher)

# on recupere la grille unique de m et de l pour reformer un tableau 2D
m_grille = np.unique(m_vals)
l_grille = np.unique(l_vals)

# tableau 2D : une ligne par l, une colonne par m
grille_log_vrais = np.zeros((len(l_grille), len(m_grille)))
for i in range(len(m_vals)):
    idx_m = np.where(m_grille == m_vals[i])[0][0]
    idx_l = np.where(l_grille == l_vals[i])[0][0]
    grille_log_vrais[idx_l, idx_m] = log_vrais_vals_affichage[i]

# couple (m,l) qui maximise la log-vraisemblance
idx_max = np.argmax(log_vrais_vals)
m_estime = m_vals[idx_max]
l_estime = l_vals[idx_max]

print(f"m_reel = {m_reel:.4f}, l_reel = {l_reel}")
print(f"m_estime = {m_estime:.4f}, l_estime = {int(l_estime)} (max de vraisemblance)")

if not args.afficher and not args.sauvegarder:
    print("Rien a afficher (utiliser --afficher ou --sauvegarder).")
else:
    fig, ax = plt.subplots(figsize=(7, 6))

    image = ax.pcolormesh(m_grille, l_grille, grille_log_vrais, shading="auto", cmap="viridis")
    fig.colorbar(image, ax=ax, label="Log-vraisemblance")

    ax.scatter(m_reel, l_reel, color="red", marker="x", s=100, label="valeur reelle")
    ax.scatter(m_estime, l_estime, color="white", marker="o", s=60,
               edgecolor="black", label="valeur estimee (max)")

    ax.set_xlabel("m")
    ax.set_ylabel("l")
    ax.set_title("Profil de vraisemblance 2D (m, l)")
    ax.legend()
    plt.tight_layout()

    if args.sauvegarder:
        nom = f"profil_vraisemblance_ml_mreel{m_reel:.2f}_lreel{l_reel}.png"
        plt.savefig(nom, dpi=150)
        print(f"Graphique sauvegarde : {nom}")
    if args.afficher:
        plt.show()
    plt.close(fig)
