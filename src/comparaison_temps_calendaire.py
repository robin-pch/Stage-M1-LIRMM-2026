# -*- coding: utf-8 -*-
# Compare le temps de coalescence en temps calendaire entre Moran et une
# marche aleatoire, a distance de depart fixee.
#
# Moran : on reprend generer_evenements et forward_uniforme tels quels
# (comparaison_analytique.py), qui donnent un tableau de toutes les paires
# (z1, z2, t, d). On filtre par distance d0, puis conversion calendaire :
#   t_cal = 2 * t_fwd * m / (n * lam)
# Facteur 2 car les deux lignees diffusent chacune (meme principe que
# proba_transition_exp).
#
# Marche aleatoire (une lignee tiree sur deux a chaque tour) : un temps dt
# tire dans une expo de parametre lam, une lignee tiree au hasard (proba
# 0.5), elle se deplace comme dans generer_evenements (proba k*m/4 par
# voisin, sinon reste sur place). On cumule les dt jusqu'a coalescence.
# Ce temps cumule est deja en calendaire, pas de conversion supplementaire.
#
# Usage :
#   python comparaison_temps_calendaire.py --l 7 --rep 500 --afficher
#   python comparaison_temps_calendaire.py --l 7 --distances 1 2 4 6 --rep 500 --sauvegarder

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys


# =============================================================================
# Partie Moran (reprise de comparaison_analytique.py)
# =============================================================================

class Noeud:
    """
    Represente un noeud dans l'arbre de descendance (forward).

    Chaque evenement Moran cree deux fils : un en A, un en B.
    Les feuilles vivantes a la fin sont celles encore presentes dans la grille.
    """

    def __init__(self, x, y, temps):
        self.x = x
        self.y = y
        self.temps = temps
        self.fils1 = None
        self.fils2 = None
        self.est_feuille = True
        self.desc1 = 0
        self.desc2 = 0
        self.feuilles1 = []
        self.feuilles2 = []


def distance(x1, y1, x2, y2):
    """Distance euclidienne entre deux cases."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def generer_evenements(l, T, m):
    """
    Genere exactement T evenements Moran.

    Les tirages aleatoires sont pre-generes en avance par blocs (plus rapide
    que T appels numpy individuels). Les evenements "rester sur place" (A = B)
    sont inclus : ils comptent comme un pas de temps ou rien ne se passe.

    Retourne une liste de T tuples ((xA, yA), (xB, yB)).
    """
    evenements = []
    # On tire 4*T valeurs d'un coup comme marge, au cas ou beaucoup tombent
    # sur "rester sur place"
    indices_B = np.random.randint(0, l * l, size=T * 4)
    tirages_r = np.random.rand(T * 4)
    i = 0

    while len(evenements) < T:
        if i >= len(indices_B):
            # si par malchance la marge n'etait pas suffisante, on regenere
            indices_B = np.random.randint(0, l * l, size=T)
            tirages_r = np.random.rand(T)
            i = 0

        xB, yB = indices_B[i] // l, indices_B[i] % l

        voisins = []
        if xB > 0:
            voisins.append((xB - 1, yB))
        if xB < l - 1:
            voisins.append((xB + 1, yB))
        if yB > 0:
            voisins.append((xB, yB - 1))
        if yB < l - 1:
            voisins.append((xB, yB + 1))

        k = len(voisins)
        p_rester = 1.0 - k * m / 4.0
        r = tirages_r[i]
        i = i + 1

        if r < p_rester:
            xA, yA = xB, yB  # rester sur place : A = B
        else:
            idx_voisin = int((r - p_rester) / (m / 4.0))
            idx_voisin = min(idx_voisin, k - 1)
            xA, yA = voisins[idx_voisin]
        evenements.append(((xA, yA), (xB, yB)))

    return evenements


def calculer_T(n, m):
    """
    Calcule T a partir de la courbe analytique (loi geometrique, p = m/n^2).

    On cherche T tel que 99.99% des coalescences ont eu lieu avant T,
    puis on multiplie par 1.3 pour avoir une petite marge de securite.
    On ne peut pas prendre seuil = 1 exactement car log(0) est indefini.
    """
    p = m / (n * n)
    seuil = 0.9999
    T = int(np.ceil(np.log(1.0 - seuil) / np.log(1.0 - p)) * 1.3)
    return T


def forward_uniforme(l, T, evenements):
    """
    Forward uniforme : toutes les cases, pas de zone restreinte.
    Retourne (temps_array, distances_array) en pas avant le present.
    """
    # initialisation : une case = un noeud feuille
    grille = [[None] * l for _ in range(l)]
    tous_les_noeuds = []

    for x in range(l):
        for y in range(l):
            noeud = Noeud(x=x, y=y, temps=0)
            grille[x][y] = noeud
            tous_les_noeuds.append(noeud)

    # avancer dans le temps : chaque evenement cree deux nouveaux noeuds
    # sauf si A=B (rester sur place) : dans ce cas rien ne change
    for t in range(T):
        A, B = evenements[t]
        xA, yA = A
        xB, yB = B

        if (xA, yA) == (xB, yB):
            continue  # pas de temps sans bifurcation

        pere = grille[xB][yB]

        fils_A = Noeud(x=xA, y=yA, temps=t + 1)
        fils_B = Noeud(x=xB, y=yB, temps=t + 1)

        pere.fils1 = fils_A
        pere.fils2 = fils_B
        pere.est_feuille = False
        pere.temps = t + 1

        grille[xA][yA].est_feuille = False
        grille[xA][yA] = fils_A
        grille[xB][yB] = fils_B

        tous_les_noeuds.append(fils_A)
        tous_les_noeuds.append(fils_B)

    # marquer les feuilles vivantes (celles encore dans la grille a T)
    for noeud in tous_les_noeuds:
        if noeud.est_feuille:
            noeud.desc1 = 1
            noeud.feuilles1 = [(noeud.x, noeud.y)]

    # post-ordre iteratif : les peres sont crees avant leurs fils,
    # donc parcourir a l'envers revient a traiter les fils en premier
    for i in range(len(tous_les_noeuds) - 1, -1, -1):
        noeud = tous_les_noeuds[i]
        if noeud.fils1 is not None:
            noeud.desc1 = noeud.fils1.desc1 + noeud.fils1.desc2
            noeud.desc2 = noeud.fils2.desc1 + noeud.fils2.desc2
            noeud.feuilles1 = noeud.fils1.feuilles1 + noeud.fils1.feuilles2
            noeud.feuilles2 = noeud.fils2.feuilles1 + noeud.fils2.feuilles2

    # collecte des noeuds valides : desc1 >= 1 et desc2 >= 1
    temps_liste = []
    distances_liste = []

    for noeud in tous_les_noeuds:
        if noeud.fils1 is not None:
            if noeud.desc1 >= 1 and noeud.desc2 >= 1:
                for (xa, ya) in noeud.feuilles1:
                    for (xb, yb) in noeud.feuilles2:
                        d = distance(xa, ya, xb, yb)
                        temps_liste.append(noeud.temps)
                        distances_liste.append(d)

    # conversion en "pas avant le present" : t=1 = dernier evenement
    temps_array = np.array(temps_liste)
    distances_array = np.array(distances_liste)
    if len(temps_array) > 0:
        temps_array = T - temps_array + 1

    return temps_array, distances_array


# =============================================================================
# Partie marche aleatoire en temps calendaire
# =============================================================================

def deplacer_lignee(x, y, m, l):
    """Deplace une lignee d'un pas : proba k*m/4 par voisin valide, sinon reste sur place."""
    voisins = []
    if x > 0:
        voisins.append((x - 1, y))
    if x < l - 1:
        voisins.append((x + 1, y))
    if y > 0:
        voisins.append((x, y - 1))
    if y < l - 1:
        voisins.append((x, y + 1))

    k = len(voisins)
    p_rester = 1.0 - k * m / 4.0
    r = np.random.rand()

    if r < p_rester:
        return x, y  # reste sur place

    idx_voisin = int((r - p_rester) / (m / 4.0))
    idx_voisin = min(idx_voisin, k - 1)
    return voisins[idx_voisin]


def backward_marche_aleatoire(l, m, lam, t_max):
    """
    Coalescence backward de deux lignees, une lignee tiree sur deux a chaque
    tour. Les deux lignees partent au hasard sur la grille. A chaque tour :
    dt tire dans une expo(lam), une lignee tiree au hasard (proba 0.5), elle
    se deplace avec deplacer_lignee. On cumule jusqu'a coalescence.

    Le temps cumule est deja calendaire, pas de conversion a faire ensuite.
    t_max evite une boucle infinie.

    Renvoie (temps_calendaire, distance_depart), ou (None, None) si pas
    coalesce avant t_max.
    """
    x1, y1 = np.random.randint(0, l), np.random.randint(0, l)
    x2, y2 = np.random.randint(0, l), np.random.randint(0, l)

    distance_depart = distance(x1, y1, x2, y2)

    temps_calendaire = 0.0

    while x1 != x2 or y1 != y2:

        dt = np.random.exponential(scale=1.0 / lam)
        temps_calendaire = temps_calendaire + dt

        if temps_calendaire > t_max:
            return None, None

        if np.random.rand() < 0.5:
            x1, y1 = deplacer_lignee(x1, y1, m, l)
        else:
            x2, y2 = deplacer_lignee(x2, y2, m, l)

    return temps_calendaire, distance_depart


# =============================================================================
# Affichage
# =============================================================================

def afficher_comparaison(resultats_par_distance, l, m, lam, afficher, sauvegarder):
    """
    Un sous-graphique par distance d0 : distribution du temps de
    coalescence calendaire, Moran et marche aleatoire superposes (en
    lignes plutot qu'en barres). Bins propres a chaque distribution.

    resultats_par_distance : {d0: {"moran": array, "marche": array}}
    """
    distances = sorted(resultats_par_distance.keys())
    n_distances = len(distances)

    n_colonnes = int(np.ceil(np.sqrt(n_distances)))
    n_lignes = int(np.ceil(n_distances / n_colonnes))

    fig, axes = plt.subplots(n_lignes, n_colonnes, figsize=(5 * n_colonnes, 4 * n_lignes))
    axes = np.array(axes).reshape(-1)

    n_bins = 30

    for i, d0 in enumerate(distances):
        ax = axes[i]
        temps_moran = resultats_par_distance[d0]["moran"]
        temps_marche = resultats_par_distance[d0]["marche"]

        if len(temps_moran) > 0:
            hauteurs, bords = np.histogram(temps_moran, bins=n_bins, density=True)
            centres = (bords[:-1] + bords[1:]) / 2
            ax.plot(centres, hauteurs, "-", color="#0072B2",
                    label=f"Moran (n={len(temps_moran)})")

        if len(temps_marche) > 0:
            hauteurs, bords = np.histogram(temps_marche, bins=n_bins, density=True)
            centres = (bords[:-1] + bords[1:]) / 2
            ax.plot(centres, hauteurs, "-", color="#D55E00",
                    label=f"Marche aleatoire (n={len(temps_marche)})")

        ax.set_xlabel("Temps de coalescence (calendaire)")
        ax.set_ylabel("Densite")
        ax.set_title(f"d0 = {d0}")
        ax.legend(fontsize=8)

    for j in range(n_distances, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"Temps de coalescence calendaire : Moran vs marche aleatoire "
                 f"(grille {l}x{l}, m={m}, lam={lam})", fontsize=11)
    plt.tight_layout()

    if sauvegarder:
        nom = f"comparaison_temps_calendaire_l{l}.png"
        plt.savefig(nom, dpi=150)
        print(f"  Graphique sauvegarde : {nom}")
    if afficher:
        plt.show()


# =============================================================================
# Lecture des arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Comparaison Moran / marche aleatoire en temps calendaire, a distance fixe"
)

parser.add_argument("--l", type=int, default=7,
                    help="Cote de la grille l x l (defaut : 7)")

parser.add_argument("--m", type=float, default=1.0,
                    help="Taux de migration (defaut : 1.0)")

parser.add_argument("--lam", type=float, default=1.0,
                    help="lambda = 1/temps de generation (defaut : 1.0)")

parser.add_argument("--distances", type=float, nargs="+", default=None,
                    help="Liste des distances d0 a comparer. Si absent, "
                         "calcule automatiquement 4 distances en coupant "
                         "la distance max atteignable (l-1) en 4 parts "
                         "egales.")

parser.add_argument("--rep", type=int, default=500,
                    help="Nombre de repetitions du forward Moran (defaut : 500). "
                         "Chaque repetition genere des milliers de paires d'un coup.")

parser.add_argument("--rep_marche", type=int, default=20000,
                    help="Nombre de simulations marche aleatoire (defaut : 20000). "
                         "Chaque simulation ne produit qu'une seule paire (les deux "
                         "lignees sont tirees au hasard sur la grille), donc il en "
                         "faut beaucoup plus que pour Moran.")

parser.add_argument("--T", type=int, default=None,
                    help="Nombre de pas de Moran. Si absent, calcule automatiquement.")

parser.add_argument("--afficher", action="store_true",
                    help="Affiche les graphiques")

parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde les graphiques en .png")

parser.add_argument("--seed", type=int, default=None,
                    help="Graine pour reproductibilite")

args = parser.parse_args()

if args.distances is None:
    # distance max en ligne droite sur la grille = l-1 (coords de 0 a l-1)
    # on coupe en 4 parts egales, arrondies a l'entier le plus proche
    d_max = args.l - 1
    args.distances = [round(d_max / 4), round(2 * d_max / 4),
                       round(3 * d_max / 4), round(d_max)]


# =============================================================================
# Lancement
# =============================================================================

n = args.l * args.l

if args.T is not None:
    T = args.T
else:
    T = calculer_T(n, args.m)
    print(f"T calcule automatiquement : T = {T}")

if args.seed is not None:
    np.random.seed(args.seed)
    print(f"Graine fixee : {args.seed}")

print(f"Grille {args.l}x{args.l} | n={n} | T={T} | m={args.m} | lam={args.lam}")
print(f"Distances comparees : {args.distances}")

# --- Moran : tableau (t_fwd, d_fwd) avec le forward, comme dans comparaison_analytique.py ---
print(f"\nSimulation forward Moran (T={T}, rep={args.rep})...")
t_fwd_total = []
d_fwd_total = []

for rep in range(args.rep):
    print(f"  repetition {rep + 1}/{args.rep}...", end="\r", flush=True)
    evenements = generer_evenements(args.l, T, args.m)
    t_rep, d_rep = forward_uniforme(args.l, T, evenements)
    t_fwd_total.extend(t_rep)
    d_fwd_total.extend(d_rep)

t_fwd = np.array(t_fwd_total)
d_fwd = np.array(d_fwd_total)
print(f"\nSimulation forward terminee : {len(t_fwd)} paires au total")

# conversion en calendaire : facteur 2 car les deux lignees diffusent
# chacune (meme convention que proba_transition_exp)
t_fwd_cal = 2 * t_fwd * args.m / (n * args.lam)

# --- Marche aleatoire : tableau (temps, distance) sur rep_marche paires tirees au hasard ---
print(f"\nSimulation marche aleatoire (rep_marche={args.rep_marche})...")
t_max_marche = (T * 5) / args.lam  # grosse marge de securite

t_marche_total = []
d_marche_total = []
n_non_coal = 0

for rep in range(args.rep_marche):
    if rep % 500 == 0:
        print(f"  repetition {rep + 1}/{args.rep_marche}...", end="\r", flush=True)
    t_marche, d_marche = backward_marche_aleatoire(args.l, args.m, args.lam, t_max_marche)
    if t_marche is None:
        n_non_coal = n_non_coal + 1
    else:
        t_marche_total.append(t_marche)
        d_marche_total.append(d_marche)

t_marche_cal = np.array(t_marche_total)
d_marche = np.array(d_marche_total)
print(f"\nSimulation marche aleatoire terminee : {len(t_marche_cal)} coalescences "
      f"({n_non_coal} non coalescees avant t_max)")

# --- filtrage des deux tableaux par distance demandee ---
resultats_par_distance = {}

for d0 in args.distances:

    masque_moran = np.abs(d_fwd - d0) < 1e-6
    temps_moran_cal = t_fwd_cal[masque_moran]

    masque_marche = np.abs(d_marche - d0) < 1e-6
    temps_marche_cal = t_marche_cal[masque_marche]

    if len(temps_moran_cal) == 0 or len(temps_marche_cal) == 0:
        print(f"  d0={d0} : pas assez de paires (Moran={len(temps_moran_cal)}, "
              f"marche aleatoire={len(temps_marche_cal)}), distance ignoree.")
        continue

    print(f"  d0={d0} : {len(temps_moran_cal)} paires Moran | "
          f"{len(temps_marche_cal)} paires marche aleatoire")

    resultats_par_distance[d0] = {
        "moran": temps_moran_cal,
        "marche": temps_marche_cal
    }

if len(resultats_par_distance) == 0:
    print("\nAucun resultat a afficher (aucune distance valide).")
    sys.exit(0)

if not args.afficher and not args.sauvegarder:
    print("Rien a afficher (utiliser --afficher ou --sauvegarder).")
    sys.exit(0)

afficher_comparaison(resultats_par_distance, args.l, args.m, args.lam,
                     args.afficher, args.sauvegarder)