# -*- coding: utf-8 -*-
# Log-vraisemblance de m pour un jeu de paires (z1,z2,t), l fixe a sa vraie valeur.
# t est deja en calendaire, fige a la generation, on le recalcule jamais ici.
# Robin Pioch, stage M1 LIRMM, juin 2026

# Usage : python vraisemblance.py --entree donnees/donnees_jeu0.csv --n_m 50 --sauvegarder
#         python vraisemblance.py --entree donnees/donnees_jeu0.csv --optimiser --sauvegarder

import numpy as np
import argparse
import csv
import os
from scipy.optimize import minimize_scalar


# =============================================================================
# Formules analytiques
# =============================================================================

def proba_transition_exp(xy0, xy1, m, l, t, lam):
    """Proba de transition en temps continu. t = temps cumule des 2 lignees, donc on passe 2*t."""
    x0r = xy0[0] + 1
    y0r = xy0[1] + 1
    x1r = xy1[0] + 1
    y1r = xy1[1] + 1
    n = l

    X = np.arange(1, n)
    Y = np.arange(1, n)

    terme_const = 1.0 / n**2

    poids_X = np.exp(-(m/2) * lam * t * (1 - np.cos(np.pi * X / n)))
    cos_x0 = np.cos(np.pi * X * (0.5 - x0r) / n)
    cos_x1 = np.cos(np.pi * X * (0.5 - x1r) / n)
    terme_X = 2.0 / n**2 * np.sum(poids_X * cos_x0 * cos_x1)

    poids_Y = np.exp(-(m/2) * lam * t * (1 - np.cos(np.pi * Y / n)))
    cos_y0 = np.cos(np.pi * Y * (0.5 - y0r) / n)
    cos_y1 = np.cos(np.pi * Y * (0.5 - y1r) / n)
    terme_Y = 2.0 / n**2 * np.sum(poids_Y * cos_y0 * cos_y1)

    KX, KY = np.meshgrid(X, Y, indexing="ij")
    poids_XY = np.exp(-(m/2) * lam * t * (2 - np.cos(np.pi * KX / n) - np.cos(np.pi * KY / n)))
    cos_kx0 = np.cos(np.pi * KX * (0.5 - x0r) / n)
    cos_kx1 = np.cos(np.pi * KX * (0.5 - x1r) / n)
    cos_ky0 = np.cos(np.pi * KY * (0.5 - y0r) / n)
    cos_ky1 = np.cos(np.pi * KY * (0.5 - y1r) / n)
    terme_XY = 4.0 / n**2 * np.sum(
        poids_XY * cos_kx0 * cos_kx1 * cos_ky0 * cos_ky1
    )

    return terme_const + terme_X + terme_Y + terme_XY


def densite_temps_coalescence(t, n, m, lam):
    """
    Densite expo du temps de coalescence calendaire.
    Parametre = m*lam/n :
    Pr(T'>t) = (1 - m/n^2)^(n*t/alpha) ~ exp(-m*t/(n*alpha)) = exp(-m*lam*t/n).
    """
    parametre = m * lam / n
    return parametre * np.exp(-parametre * t)


def proba_transition_exp_cond(z0, z1, m, l, t, lam):
    """
    Version conditionnelle de proba_transition_exp : met p(z0->z0, t) a zero
    et renormalise par 1 - p(z0->z0, t). Ca exclut les paires a distance zero,
    coherent avec le fait que deux lignees issues d'un meme evenement Moran
    ne peuvent pas etre au meme endroit.
    """
    if z1 == z0:
        return 0.0
    p_z1 = proba_transition_exp(z0, z1, m, l, t, lam)
    p_z0 = proba_transition_exp(z0, z0, m, l, t, lam)
    denom = 1.0 - p_z0
    if denom <= 0:
        return 0.0
    return p_z1 / denom


# =============================================================================
# Vraisemblance
# =============================================================================

def vraisemblance_tuple(z1, z2, t, m, l, lam, cond=False):
    """
    Vraisemblance d'une paire (z1,z2,t) pour un m candidat.
    Si cond=True, utilise proba_transition_exp_cond (exclut d=0 et renormalise).
    """
    n = l * l
    if cond:
        p_cond = proba_transition_exp_cond(z1, z2, m, l, 2 * t, lam) / n
    else:
        p_cond = proba_transition_exp(z1, z2, m, l, 2 * t, lam) / n
    p_t = densite_temps_coalescence(t, n, m, lam)
    return p_cond * p_t


def log_vraisemblance_jeu(paires, m, l, lam, cond=False):
    """Somme des log-vraisemblances sur toutes les paires (= log du produit, paires supposees indep)."""
    log_vrais = 0.0
    for (z1, z2, t) in paires:
        p = vraisemblance_tuple(z1, z2, t, m, l, lam, cond=cond)
        if p <= 0:
            log_vrais = log_vrais + (-np.inf)
        else:
            log_vrais = log_vrais + np.log(p)
    return log_vrais


# =============================================================================
# Lecture du fichier d'entree
# =============================================================================

def lire_jeu(chemin_csv):
    """
    Lit le csv d'un jeu, renvoie les paires (z1,z2,t) + l, lam, id_jeu.
    Les paires coalesce=0 (censurees, pas de t) sont ignorees pour l'instant.
    """
    paires = []
    l = None
    lam = None
    id_jeu = None
    n_ignorees = 0

    with open(chemin_csv, newline="") as f:
        lecteur = csv.DictReader(f)
        for ligne in lecteur:
            if "coalesce" in ligne and ligne["coalesce"] == "0":
                n_ignorees += 1
                continue
            x1 = int(ligne["x1"])
            y1 = int(ligne["y1"])
            x2 = int(ligne["x2"])
            y2 = int(ligne["y2"])
            t = float(ligne["t"])
            l = int(ligne["l"])
            lam = float(ligne["lam"])
            id_jeu = int(ligne["id_jeu"])
            paires.append(((x1, y1), (x2, y2), t))

    if n_ignorees > 0:
        print(f"{n_ignorees} paires non coalescees ignorees (pas encore geree en vraisemblance)")

    return paires, l, lam, id_jeu


# =============================================================================
# Lecture des arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Calcule la log-vraisemblance de m pour un jeu de paires"
)

parser.add_argument("--entree", type=str, required=True,
                    help="Chemin du csv de donnees (un seul jeu)")

parser.add_argument("--n_m", type=int, default=50,
                    help="Nombre de valeurs de m testees (defaut : 50)")

parser.add_argument("--cond", action="store_true",
                    help="Utilise proba_transition_exp_cond (exclut d=0 et renormalise)")

parser.add_argument("--optimiser", action="store_true",
                    help="Utilise minimize_scalar (Brent) au lieu d'une grille de m. "
                         "Plus rapide, mais ne produit qu'un seul point (m_estime) "
                         "sans la courbe complete.")

parser.add_argument("--dossier", type=str, default="resultats",
                    help="Dossier de sortie pour le csv de resultats (defaut : resultats)")

parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde le tableau de resultats en .csv")

args = parser.parse_args()


# =============================================================================
# Lancement
# =============================================================================

print(f"Lecture de {args.entree}...")
paires, l, lam, id_jeu = lire_jeu(args.entree)
print(f"jeu {id_jeu} : {len(paires)} paires, l={l}, lam={lam}")
if args.cond:
    print("Mode : proba_transition_exp_cond (d=0 exclu, renormalise)")

if args.optimiser:
    # minimize_scalar minimise, donc on lui passe le negatif de la log-vraisemblance
    def neg_log_vrais(m):
        return -log_vraisemblance_jeu(paires, m, l, lam, cond=args.cond)

    resultat = minimize_scalar(neg_log_vrais, bounds=(0.01, 1.0), method="bounded")
    m_estime = resultat.x
    n_evals = resultat.nfev
    print(f"m_estime = {m_estime:.4f} ({n_evals} evaluations de la vraisemblance)")
    lignes_resultats = [[id_jeu, m_estime, l, -resultat.fun]]

else:
    m_vals = np.linspace(0.01, 1.0, args.n_m)
    lignes_resultats = []
    for m_test in m_vals:
        log_vrais = log_vraisemblance_jeu(paires, m_test, l, lam, cond=args.cond)
        lignes_resultats.append([id_jeu, m_test, l, log_vrais])
    print(f"{len(lignes_resultats)} valeurs de m testees.")

if not args.sauvegarder:
    print("Rien a sauvegarder (utiliser --sauvegarder).")
else:
    os.makedirs(args.dossier, exist_ok=True)
    # nom de sortie base sur le fichier d'entree (pas juste id_jeu), pour ne pas
    # ecraser les resultats entre plusieurs tailles d'echantillon d'un meme jeu
    nom_base = os.path.splitext(os.path.basename(args.entree))[0]
    if nom_base.startswith("donnees"):
        nom_sortie = nom_base.replace("donnees", "resultats", 1)
    else:
        nom_sortie = f"resultats_{nom_base}"
    nom_csv = os.path.join(args.dossier, f"{nom_sortie}.csv")
    with open(nom_csv, "w", newline="") as f:
        ecrivain = csv.writer(f)
        ecrivain.writerow(["id_jeu", "m", "l", "log_vraisemblance"])
        for ligne in lignes_resultats:
            ecrivain.writerow(ligne)
    print(f"CSV exporte : {nom_csv}")
