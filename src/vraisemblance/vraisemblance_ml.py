# -*- coding: utf-8 -*-
# Log-vraisemblance de m et/ou l pour un jeu de paires (z1,z2,t).
# Trois modes : grille ou optimisation sur (m,l) ensemble, ou m fixe (seul l estime).
# Utilise les coordonnees relatives (x1r,y1r,...) plutot que les positions
# absolues, pour ne pas fuiter d'info sur l quand plusieurs l sont testes :
# une position absolue hors grille rendrait certains l testes invalides
# avant meme le calcul de vraisemblance, ce qui biaiserait l'estimation.
# Robin Pioch, stage M1 LIRMM, juillet 2026

# Usage : python vraisemblance_ml.py --entree donnees/donnees_jeu0.csv --n_m 20 --n_l 20 --sauvegarder
#         python vraisemblance_ml.py --entree donnees/donnees_jeu0.csv --optimiser --sauvegarder
#         python vraisemblance_ml.py --entree donnees/donnees_jeu0.csv --m_fixe --sauvegarder

import numpy as np
import argparse
import csv
import os
from scipy.optimize import minimize


# =============================================================================
# Formules analytiques (identiques a vraisemblance.py)
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

    # une serie de cosinus tronquee peut redescendre legerement sous 0
    # a cause d'arrondis (phenomene de Gibbs), on le bloque a 0
    resultat = terme_const + terme_X + terme_Y + terme_XY
    return max(resultat, 0.0)


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


def p_z1_z2(xy0, xy1, m, l, lam):
    """
    p(z1,z2) integree sur tous les temps possibles. Sert a corriger le biais
    introduit par un schema d'echantillonnage non uniforme (cercle, diagonale).
    Meme structure que proba_transition_exp, mais avec 1/(taux) a la place
    de exp(-taux*t) (integrale d'une exponentielle sur tous les temps).
    """
    x0r = xy0[0] + 1
    y0r = xy0[1] + 1
    x1r = xy1[0] + 1
    y1r = xy1[1] + 1
    n = l

    X = np.arange(1, n)
    Y = np.arange(1, n)

    terme_const = 1.0 / (n**2 * m * lam)

    poids_X = 1.0 / (m * lam * (1.0 / n**2 + 0.5 - 0.5 * np.cos(np.pi * X / n)))
    cos_x0 = np.cos(np.pi * X * (0.5 - x0r) / n)
    cos_x1 = np.cos(np.pi * X * (0.5 - x1r) / n)
    terme_X = 2.0 / n**4 * np.sum(poids_X * cos_x0 * cos_x1)

    poids_Y = 1.0 / (m * lam * (1.0 / n**2 + 0.5 - 0.5 * np.cos(np.pi * Y / n)))
    cos_y0 = np.cos(np.pi * Y * (0.5 - y0r) / n)
    cos_y1 = np.cos(np.pi * Y * (0.5 - y1r) / n)
    terme_Y = 2.0 / n**4 * np.sum(poids_Y * cos_y0 * cos_y1)

    KX, KY = np.meshgrid(X, Y, indexing="ij")
    poids_XY = 1.0 / (m * lam * (1.0 / n**2 + 1.0 - 0.5 * np.cos(np.pi * KX / n) - 0.5 * np.cos(np.pi * KY / n)))
    cos_kx0 = np.cos(np.pi * KX * (0.5 - x0r) / n)
    cos_kx1 = np.cos(np.pi * KX * (0.5 - x1r) / n)
    cos_ky0 = np.cos(np.pi * KY * (0.5 - y0r) / n)
    cos_ky1 = np.cos(np.pi * KY * (0.5 - y1r) / n)
    terme_XY = 4.0 / n**4 * np.sum(
        poids_XY * cos_kx0 * cos_kx1 * cos_ky0 * cos_ky1
    )

    return terme_const + terme_X + terme_Y + terme_XY


# =============================================================================
# Vraisemblance
# =============================================================================

def vraisemblance_tuple(z1r, z2r, t, m, l, lam, cond=False, echantillonnage=False):
    """
    Vraisemblance d'une paire (z1r,z2r,t) pour un (m,l) candidat.
    z1r, z2r sont les coordonnees relatives (entre 0 et 1) lues dans le csv.
    On reconstruit la position sur la grille l x l testee avec la case
    entiere la plus proche : round(x_relatif * l).

    Si cond=True, utilise proba_transition_exp_cond (exclut d=0 et renormalise).
    Si echantillonnage=True, divise en plus par p(z1,z2) pour corriger le
    biais d'un schema d'echantillonnage non uniforme.
    """
    n = l * l
    p_t = densite_temps_coalescence(t, n, m, lam)

    # position 1-indexee la plus proche (1 a l), puis -1 pour redonner a
    # proba_transition_exp son entree 0-indexee habituelle (elle fait +1 en interne)
    x1 = min(max(round(z1r[0] * l), 1), l) - 1
    y1 = min(max(round(z1r[1] * l), 1), l) - 1
    x2 = min(max(round(z2r[0] * l), 1), l) - 1
    y2 = min(max(round(z2r[1] * l), 1), l) - 1
    z1 = (x1, y1)
    z2 = (x2, y2)

    if cond:
        p_cond = proba_transition_exp_cond(z1, z2, m, l, 2 * t, lam) / n
    else:
        p_cond = proba_transition_exp(z1, z2, m, l, 2 * t, lam) / n

    if not echantillonnage:
        return p_cond * p_t

    denom = p_z1_z2(z1, z2, m, l, lam)
    if denom <= 0:
        return 0.0
    return (p_cond * p_t) / denom


def log_vraisemblance_jeu(paires, m, l, lam, cond=False, echantillonnage=False):
    """
    Somme des log-vraisemblances sur toutes les paires (= log du produit,
    paires supposees indep). On utilise un plancher numerique (1e-300) a la
    place de -inf quand p<=0, pour eviter qu'un seul cas limite (underflow
    de la serie de cosinus tronquee) ne bloque l'optimiseur.
    """
    log_vrais = 0.0
    for (z1, z2, t) in paires:
        p = vraisemblance_tuple(z1, z2, t, m, l, lam, cond=cond, echantillonnage=echantillonnage)
        p_plancher = max(p, 1e-300)
        log_vrais = log_vrais + np.log(p_plancher)
    return log_vrais


# =============================================================================
# Lecture du fichier d'entree
# =============================================================================

def lire_jeu(chemin_csv):
    """
    Lit le csv d'un jeu, renvoie les paires (z1r,z2r,t) en coordonnees
    relatives + m_vrai, l_vrai, lam, id_jeu.
    Les paires coalesce=0 (censurees, pas de t) sont ignorees pour l'instant.
    """
    paires = []
    m_vrai = None
    l_vrai = None
    lam = None
    id_jeu = None
    n_ignorees = 0

    with open(chemin_csv, newline="") as f:
        lecteur = csv.DictReader(f)
        for ligne in lecteur:
            if "coalesce" in ligne and ligne["coalesce"] == "0":
                n_ignorees += 1
                continue
            x1r = float(ligne["x1r"])
            y1r = float(ligne["y1r"])
            x2r = float(ligne["x2r"])
            y2r = float(ligne["y2r"])
            t = float(ligne["t"])
            m_vrai = float(ligne["m"])
            l_vrai = int(ligne["l"])
            lam = float(ligne["lam"])
            id_jeu = int(ligne["id_jeu"])
            paires.append(((x1r, y1r), (x2r, y2r), t))

    if n_ignorees > 0:
        print(f"{n_ignorees} paires non coalescees ignorees (pas encore geree en vraisemblance)")

    return paires, m_vrai, l_vrai, lam, id_jeu


# =============================================================================
# Lecture des arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Calcule la log-vraisemblance de m et/ou l pour un jeu de paires"
)

parser.add_argument("--entree", type=str, required=True,
                    help="Chemin du csv de donnees (un seul jeu)")

parser.add_argument("--n_m", type=int, default=20,
                    help="Nombre de valeurs de m testees (defaut : 20)")

parser.add_argument("--n_l", type=int, default=20,
                    help="Nombre de valeurs de l testees (defaut : 20)")

parser.add_argument("--l_min", type=int, default=5,
                    help="Plus petite valeur de l testee (defaut : 5)")

parser.add_argument("--l_max", type=int, default=200,
                    help="Plus grande valeur de l testee (defaut : 200)")

parser.add_argument("--cond", action="store_true",
                    help="Utilise proba_transition_exp_cond (exclut d=0 et renormalise)")

parser.add_argument("--echantillonnage", action="store_true",
                    help="Corrige la vraisemblance pour un schema d'echantillonnage non "
                         "uniforme (divise par p(z1,z2)).")

parser.add_argument("--m_fixe", action="store_true",
                    help="Fixe m a sa vraie valeur (lue dans le csv), estime seulement l.")

parser.add_argument("--optimiser", action="store_true",
                    help="Utilise minimize (Nelder-Mead) sur (m,l) au lieu d'une grille. "
                         "Plus rapide, mais ne produit qu'un seul point (m_estime, l_estime) "
                         "sans la surface complete.")

parser.add_argument("--dossier", type=str, default="resultats",
                    help="Dossier de sortie pour le csv de resultats (defaut : resultats)")

parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde le tableau de resultats en .csv")

args = parser.parse_args()


# =============================================================================
# Lancement
# =============================================================================

print(f"Lecture de {args.entree}...")
paires, m_vrai, l_vrai, lam, id_jeu = lire_jeu(args.entree)
print(f"jeu {id_jeu} : {len(paires)} paires, m_vrai={m_vrai}, l_vrai={l_vrai}, lam={lam}")
if args.cond:
    print("Mode : proba_transition_exp_cond (d=0 exclu, renormalise)")
if args.echantillonnage:
    print("Mode : correction pour echantillonnage non uniforme (division par p(z1,z2))")
if args.m_fixe:
    print(f"Mode : m fixe a sa vraie valeur ({m_vrai}), seul l est estime")

if args.m_fixe:

    # m est fixe, on n'estime que l
    if args.optimiser:
        def neg_log_vrais_l(l_param):
            l_test = int(round(l_param[0]))
            return -log_vraisemblance_jeu(paires, m_vrai, l_test, lam, cond=args.cond,
                                          echantillonnage=args.echantillonnage)

        resultat = minimize(neg_log_vrais_l, [(args.l_min + args.l_max) / 2.0],
                            method="Nelder-Mead", bounds=[(args.l_min, args.l_max)])
        l_estime = int(round(resultat.x[0]))
        n_evals = resultat.nfev
        print(f"l_estime = {l_estime} ({n_evals} evaluations de la vraisemblance)")
        lignes_resultats = [[id_jeu, m_vrai, l_estime, -resultat.fun]]
    else:
        l_vals = np.linspace(args.l_min, args.l_max, args.n_l)
        l_vals = sorted(set(int(round(l_test)) for l_test in l_vals))
        lignes_resultats = []
        for l_test in l_vals:
            log_vrais = log_vraisemblance_jeu(paires, m_vrai, l_test, lam, cond=args.cond,
                                              echantillonnage=args.echantillonnage)
            lignes_resultats.append([id_jeu, m_vrai, l_test, log_vrais])
        print(f"{len(lignes_resultats)} valeurs de l testees.")

elif args.optimiser:

    # minimize minimise, donc on lui passe le negatif de la log-vraisemblance
    # l doit etre un entier (nombre de cases par cote), on arrondit dedans la fonction
    def neg_log_vrais(params):
        m_test, l_test = params
        l_test = int(round(l_test))
        if l_test < args.l_min or m_test <= 0:
            return np.inf
        return -log_vraisemblance_jeu(paires, m_test, l_test, lam, cond=args.cond,
                                      echantillonnage=args.echantillonnage)

    # avec des coordonnees relatives, la reconstruction est toujours valide
    # quel que soit l_test, donc le milieu de l_min/l_max suffit comme depart
    point_depart = [0.5, (args.l_min + args.l_max) / 2.0]
    bornes = [(0.01, 1.0), (args.l_min, args.l_max)]

    resultat = minimize(neg_log_vrais, point_depart, method="Nelder-Mead", bounds=bornes)
    m_estime = resultat.x[0]
    l_estime = int(round(resultat.x[1]))
    n_evals = resultat.nfev
    print(f"m_estime = {m_estime:.4f}, l_estime = {l_estime} ({n_evals} evaluations de la vraisemblance)")
    lignes_resultats = [[id_jeu, m_estime, l_estime, -resultat.fun]]

else:
    m_vals = np.linspace(0.01, 1.0, args.n_m)
    l_vals = np.linspace(args.l_min, args.l_max, args.n_l)
    l_vals = sorted(set(int(round(l_test)) for l_test in l_vals))

    lignes_resultats = []
    for m_test in m_vals:
        for l_test in l_vals:
            log_vrais = log_vraisemblance_jeu(paires, m_test, l_test, lam, cond=args.cond,
                                              echantillonnage=args.echantillonnage)
            lignes_resultats.append([id_jeu, m_test, l_test, log_vrais])
    print(f"{len(lignes_resultats)} couples (m,l) testes ({len(m_vals)} valeurs de m x {len(l_vals)} valeurs de l).")

if not args.sauvegarder:
    print("Rien a sauvegarder (utiliser --sauvegarder).")
else:
    os.makedirs(args.dossier, exist_ok=True)
    # nom de sortie base sur le fichier d'entree (pas juste id_jeu), pour ne pas
    # ecraser les resultats entre plusieurs tailles d'echantillon d'un meme jeu
    nom_base = os.path.splitext(os.path.basename(args.entree))[0]
    if nom_base.startswith("donnees"):
        nom_sortie = nom_base.replace("donnees", "resultats_ml", 1)
    else:
        nom_sortie = f"resultats_ml_{nom_base}"
    if args.echantillonnage:
        nom_sortie = nom_sortie + "_ech"
    if args.m_fixe:
        nom_sortie = nom_sortie + "_mfixe"
    nom_csv = os.path.join(args.dossier, f"{nom_sortie}.csv")
    with open(nom_csv, "w", newline="") as f:
        ecrivain = csv.writer(f)
        ecrivain.writerow(["id_jeu", "m", "l", "log_vraisemblance"])
        for ligne in lignes_resultats:
            ecrivain.writerow(ligne)
    print(f"CSV exporte : {nom_csv}")
