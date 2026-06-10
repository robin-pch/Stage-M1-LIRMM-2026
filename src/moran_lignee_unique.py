# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Diffusion d'une lignée unique - Moran spatialisé
# ----------------------------------------------------------------------
# Auteur : Robin Pioch
# Stage M1 Bioinformatique, Université de Montpellier
# Encadrant : Stéphane Guindon (LIRMM)
# Juin 2026
#
# Même processus de Moran spatialisé que moran_spatialise.py : à chaque pas,
# B est tiré au hasard sur la grille et se propage sur A avec proba m/4 par
# direction (1 - k*m/4 de rester sur place).
#
# Au lieu de regarder toutes les paires de feuilles, on part d'un noeud de
# départ (x0, y0) et on tire UNE feuille vivante au hasard parmi ses
# descendants au temps T. Si la lignée est morte (écrasée à un moment),
# la répétition est comptée comme éteinte et on ne tire rien.
#
# Pour chaque répétition survivante, on enregistre les coordonnées (x, y)
# de la feuille tirée et sa distance à (x0, y0).
#
# On compare avec la formule analytique de la marche aléatoire réfléchie 2D.
# Pour faire correspondre simulation et analytique, on passe t = T/n à la
# formule (n = l*l), car un pas de Moran ne correspond pas à une génération.
#
# Usage :
#   python moran_lignee_unique.py --l 7 --T 5000 --rep 500 --m 1.0
#   python moran_lignee_unique.py --l 7 --T 5000 --rep 500 --m 1.0 --afficher
#   python moran_lignee_unique.py --l 7 --T 5000 --rep 500 --m 1.0 --sauvegarder
#   python moran_lignee_unique.py --l 7 --rep 500 --m 1.0 --afficher
#
# Options :
#   --l           : côté de la grille (défaut : 7), population = l*l
#   --T           : nombre de pas. Si absent, calculé automatiquement.
#   --rep         : nombre de répétitions (défaut : 500)
#   --m           : taux de migration (défaut : 1.0)
#   --x0          : coordonnée x de départ (défaut : centre de la grille)
#   --y0          : coordonnée y de départ (défaut : centre de la grille)
#   --afficher    : affiche les graphiques
#   --sauvegarder : sauvegarde les graphiques en .png
#   --analytique_3d : affiche uniquement la surface 3D analytique, sans simulation
# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import argparse
import sys
from math import comb


# =============================================================================
# Classe Noeud (reprise intégrale de moran_spatialise.py)
# =============================================================================

class Noeud:
    """
    Représente un noeud dans l'arbre de descendance (forward).

    Chaque événement Moran crée deux fils : un en A, un en B.
    Les feuilles vivantes à la fin sont celles encore présentes dans la grille.

    Attributs :
        x, y : position sur la grille
        temps : pas de création (absolu, depuis t=0)
        fils1, fils2 : fils gauche et droit (None si feuille)
        est_feuille : True tant que le noeud n'a pas bifurqué
        desc1, desc2 : nb de descendants vivants de chaque côté
        feuilles1, feuilles2 : coordonnées (x, y) des feuilles vivantes de chaque côté
        est_dans_zone : True si la feuille est dans la zone d'échantillonnage
        feuilles1_zone, feuilles2_zone : sous-ensemble de feuilles1/2 dans la zone
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
        self.est_dans_zone = False
        self.feuilles1_zone = []
        self.feuilles2_zone = []

    def afficher(self, profondeur=0):
        # pour le debug seulement
        indent = "  " * profondeur
        print(f"{indent}Noeud t={self.temps} ({self.x},{self.y}) "
              f"feuille={self.est_feuille} desc=({self.desc1},{self.desc2})")
        if self.fils1 is not None:
            self.fils1.afficher(profondeur + 1)
        if self.fils2 is not None:
            self.fils2.afficher(profondeur + 1)


# =============================================================================
# Fonctions utilitaires (reprises intégrales de moran_spatialise.py)
# =============================================================================

def generer_evenements(l, T, m):
    """
    Génère exactement T événements Moran.

    Les tirages sont pré-générés en avance par blocs. Les événements
    "rester sur place" (A = B) sont inclus : ils comptent comme un pas
    de temps mais ne modifient pas la grille.

    Retourne une liste de T tuples ((xA, yA), (xB, yB)).
    """
    evenements = []
    indices_B = np.random.randint(0, l * l, size=T * 4)
    tirages_r = np.random.rand(T * 4)
    i = 0

    while len(evenements) < T:
        if i >= len(indices_B):
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
            xA, yA = xB, yB  # rester sur place
        else:
            idx_voisin = int((r - p_rester) / (m / 4.0))
            idx_voisin = min(idx_voisin, k - 1)
            xA, yA = voisins[idx_voisin]
        evenements.append(((xA, yA), (xB, yB)))

    return evenements


def distance(x1, y1, x2, y2):
    """Distance euclidienne entre deux cases."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def calculer_T(n, m):
    """
    Calcule T à partir de la courbe analytique (loi géométrique, p = m/n^2).

    On cherche T tel que 99.99% des coalescences aient eu lieu, puis on
    multiplie par 1.3 pour avoir une marge de sécurité.
    """
    p = m / (n * n)
    seuil = 0.9999
    T = int(np.ceil(np.log(1.0 - seuil) / np.log(1.0 - p)) * 1.3)
    return T


# =============================================================================
# Construction de l'arbre et tirage d'un descendant
# =============================================================================

def forward_lignee_unique(l, T, evenements, x0, y0):
    """
    Construit l'arbre forward (identique à moran_spatialise.py) et tire
    une feuille vivante au hasard parmi les descendants de (x0, y0).

    On garde une référence sur noeud_depart = grille[x0][y0] avant de
    lancer la simulation. Après le post-ordre, feuilles1 + feuilles2 de
    ce noeud contient tous ses descendants vivants au temps T.
    Si la liste est vide (la lignée s'est fait écraser), on retourne None.

    Paramètres :
        l, T: côté de la grille, nombre de pas
        evenements: liste des T événements Moran
        x0, y0: case de départ

    Retourne :
        (x, y) si la lignée a survécu, None sinon
    """
    grille = [[None] * l for _ in range(l)]
    tous_les_noeuds = []

    for x in range(l):
        for y in range(l):
            noeud = Noeud(x=x, y=y, temps=0)
            grille[x][y] = noeud
            tous_les_noeuds.append(noeud)

    noeud_depart = grille[x0][y0]

    for t in range(T):
        A, B = evenements[t]
        xA, yA = A
        xB, yB = B

        if (xA, yA) == (xB, yB):
            continue  # rester sur place : pas de bifurcation

        pere = grille[xB][yB]

        fils_A = Noeud(x=xA, y=yA, temps=t + 1)
        fils_B = Noeud(x=xB, y=yB, temps=t + 1)

        pere.fils1 = fils_A
        pere.fils2 = fils_B
        pere.est_feuille = False
        pere.temps = t + 1

        # l'occupant de A meurt, que ce soit notre lignée ou non
        grille[xA][yA].est_feuille = False
        grille[xA][yA] = fils_A
        grille[xB][yB] = fils_B

        tous_les_noeuds.append(fils_A)
        tous_les_noeuds.append(fils_B)

    # marquer les feuilles vivantes
    for noeud in tous_les_noeuds:
        if noeud.est_feuille:
            noeud.desc1 = 1
            noeud.feuilles1 = [(noeud.x, noeud.y)]

    # post-ordre itératif : les pères sont créés avant les fils,
    # donc parcourir à l'envers revient à traiter les fils en premier
    for i in range(len(tous_les_noeuds) - 1, -1, -1):
        noeud = tous_les_noeuds[i]
        if noeud.fils1 is not None:
            noeud.desc1 = noeud.fils1.desc1 + noeud.fils1.desc2
            noeud.desc2 = noeud.fils2.desc1 + noeud.fils2.desc2
            noeud.feuilles1 = noeud.fils1.feuilles1 + noeud.fils1.feuilles2
            noeud.feuilles2 = noeud.fils2.feuilles1 + noeud.fils2.feuilles2

    descendants = noeud_depart.feuilles1 + noeud_depart.feuilles2

    if len(descendants) == 0:
        return None  # lignée éteinte

    idx = np.random.randint(len(descendants))
    return descendants[idx]


# =============================================================================
# Formule analytique (marche aléatoire réfléchie 2D)
# =============================================================================

def proba_transition_analytique(xy0, xy1, m, l, t):
    """
    P(xy1 | xy0, m, l, t) : probabilité de transition de xy0 vers xy1
    en t pas sur une grille l x l avec taux de migration m.

    Traduit depuis la formule R de Stéphane Guindon (rw.2d.reflected).
    Coordonnées passées en 0-indexé, converties en 1-indexé en interne.

    Paramètres :
        xy0: (x0, y0) position de départ (0-indexé)
        xy1: (x1, y1) position d'arrivée (0-indexé)
        m: taux de migration
        l: côté de la grille
        t: nombre de pas (en générations, pas en pas de Moran)

    Retourne :
        p: float
    """
    x0r = xy0[0] + 1
    y0r = xy0[1] + 1
    x1r = xy1[0] + 1
    y1r = xy1[1] + 1
    n = l

    X = np.arange(1, n + 1)
    Y = np.arange(1, n + 1)

    terme_const = 1.0 / n**2

    poids_X = (1 - m/2 + m/2 * np.cos(np.pi * X / n))**t
    cos_x0 = np.cos(np.pi * X * (0.5 - x0r) / n)
    cos_x1 = np.cos(np.pi * X * (0.5 - x1r) / n)
    terme_X = 2.0 / n**2 * np.sum(poids_X * cos_x0 * cos_x1)

    poids_Y = (1 - m/2 + m/2 * np.cos(np.pi * Y / n))**t
    cos_y0 = np.cos(np.pi * Y * (0.5 - y0r) / n)
    cos_y1 = np.cos(np.pi * Y * (0.5 - y1r) / n)
    terme_Y = 2.0 / n**2 * np.sum(poids_Y * cos_y0 * cos_y1)

    KX, KY = np.meshgrid(X, Y, indexing="ij")
    poids_XY = (1 - m + m/2 * (np.cos(np.pi * KX / n) + np.cos(np.pi * KY / n)))**t
    cos_kx0 = np.cos(np.pi * KX * (0.5 - x0r) / n)
    cos_kx1 = np.cos(np.pi * KX * (0.5 - x1r) / n)
    cos_ky0 = np.cos(np.pi * KY * (0.5 - y0r) / n)
    cos_ky1 = np.cos(np.pi * KY * (0.5 - y1r) / n)
    terme_XY = 4.0 / n**2 * np.sum(
        poids_XY * cos_kx0 * cos_kx1 * cos_ky0 * cos_ky1
    )

    return terme_const + terme_X + terme_Y + terme_XY


def heatmap_analytique(xy0, m, l, t):
    """
    Calcule P(xy1 | xy0, m, l, t) pour toutes les cases de la grille.

    On passe t = T/n en entrée pour faire correspondre les pas de Moran
    aux générations de la formule.

    Retourne un tableau l x l.
    """
    h = np.zeros((l, l))
    for x1 in range(l):
        for y1 in range(l):
            h[x1, y1] = proba_transition_analytique(xy0, (x1, y1), m, l, t)
    return h

def heatmap_analytique_melange(xy0, m, l, T):
    """
    Mélange analytique tenant compte du fait que
    la lignée est affectée un nombre aléatoire de fois.
    """

    n = l * l
    p = 1.0 / n

    h = np.zeros((l, l))

    for k in range(T + 1):

        poids = comb(T, k) * (p ** k) * ((1 - p) ** (T - k))

        if poids < 1e-12:
            continue

        h += poids * heatmap_analytique(xy0, m, l, k)

    return h

def distribution_analytique_distance(xy0, m, l, t):
    """
    Distribution analytique de la distance au départ au temps t.

    Pour chaque distance distincte d, somme les P(xy1|...) sur toutes
    les cases à cette distance de xy0.

    Retourne (d_vals, p_vals).
    """
    prob_par_dist = {}
    for x1 in range(l):
        for y1 in range(l):
            d = round(distance(xy0[0], xy0[1], x1, y1), 8)
            p = proba_transition_analytique(xy0, (x1, y1), m, l, t)
            if d not in prob_par_dist:
                prob_par_dist[d] = 0.0
            prob_par_dist[d] = prob_par_dist[d] + p
    d_vals = np.array(sorted(prob_par_dist.keys()))
    p_vals = np.array([prob_par_dist[d] for d in d_vals])
    return d_vals, p_vals

def distribution_nombre_pas(l, T):

    n = l * l
    p = 1.0 / n

    k_vals = np.arange(T + 1)

    p_vals = np.array([
        comb(T, k) * (p ** k) * ((1 - p) ** (T - k))
        for k in k_vals
    ])

    return k_vals, p_vals

# =============================================================================
# Simulation principale
# =============================================================================

def simuler(l, T, rep, x0, y0, m):
    """
    Lance rep répétitions et retourne les positions des feuilles tirées.

    Retourne :
        positions: liste de tuples (x, y), une par répétition survivante
        n_eteintes: nombre de répétitions où la lignée était éteinte
    """
    positions = []
    n_eteintes = 0

    for r in range(rep):
        print(f"  répétition {r + 1}/{rep}...", end="\r", flush=True)

        evenements = generer_evenements(l, T, m)
        resultat = forward_lignee_unique(l, T, evenements, x0, y0)

        if resultat is None:
            n_eteintes = n_eteintes + 1
        else:
            positions.append(resultat)

    print()
    return positions, n_eteintes


# =============================================================================
# Affichage des résultats
# =============================================================================

def afficher_resultats(positions, x0, y0, m, l, T, n, rep, n_eteintes,
                       afficher, sauvegarder):
    """
    Produit les figures de comparaison simulation / analytique.

    Pour l'analytique, on utilise t = T/n (conversion pas de Moran ->
    générations). Figure 1 : heatmaps simulation et analytique + distribution
    de la distance. Figure 2 : scatter plot (x, y) pondéré par les occurrences.
    """
    if len(positions) == 0:
        print("Aucune lignée survivante. Diminuer T ou augmenter rep.")
        return

    # t en générations pour la formule analytique
    t_ana = T / n

    distances_emp = np.array([distance(x0, y0, xi, yi) for (xi, yi) in positions])

    # --- Figure 1 : heatmaps simulation et analytique côte à côte ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(
        f"Lignée unique depuis ({x0},{y0}) - T={T} (t_ana={t_ana:.1f}) - "
        f"grille {l}×{l} - m={m} - "
        f"{len(positions)} rép. survivantes ({n_eteintes} éteintes)",
        fontsize=9
    )

    # heatmap empirique
    ax = axes[0]
    ax.set_title("Simulation")
    heatmap_sim = np.zeros((l, l))
    for (xi, yi) in positions:
        heatmap_sim[xi, yi] = heatmap_sim[xi, yi] + 1
    heatmap_sim = heatmap_sim / len(positions)

    # heatmap analytique (t = T/n) — calculée ici pour connaître le vmax commun
    h_ana = heatmap_analytique_melange((x0, y0), m, l, T)

    # vmax commun pour que les deux couleurs soient comparables
    vmax_commun = max(heatmap_sim.max(), h_ana.max())

    im = ax.imshow(heatmap_sim.T, origin="lower", cmap="Blues",
                   vmin=0, vmax=vmax_commun, aspect="equal")
    ax.scatter([x0], [y0], color="#D55E00", s=60, zorder=5,
               label=f"départ ({x0},{y0})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=7)
    plt.colorbar(im, ax=ax)

    # heatmap analytique
    ax = axes[1]
    ax.set_title("Analytique")
    im2 = ax.imshow(h_ana.T, origin="lower", cmap="Blues",
                    vmin=0, vmax=vmax_commun, aspect="equal")
    ax.scatter([x0], [y0], color="#D55E00", s=60, zorder=5,
               label=f"départ ({x0},{y0})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=7)
    plt.colorbar(im2, ax=ax)

    plt.tight_layout()
    if sauvegarder:
        nom = f"lignee_unique_l{l}_T{T}_m{m:.2f}.png"
        plt.savefig(nom, dpi=150)
        print(f"  Sauvegardé : {nom}")
    if afficher:
        plt.show()
    plt.close(fig)

    # --- Figure 2 : distribution jointe (x, y) simulation + analytique superposé ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title(
        f"Distribution jointe (x, y) - T={T} (t_ana={t_ana:.1f})\n"
        f"grille {l}×{l}, m={m}, {len(positions)} rép.",
        fontsize=9
    )

    # heatmap simulation en fond
    heatmap_sim2 = np.zeros((l, l))
    for (xi, yi) in positions:
        heatmap_sim2[xi, yi] = heatmap_sim2[xi, yi] + 1
    heatmap_sim2 = heatmap_sim2 / len(positions)
    ax.imshow(heatmap_sim2.T, origin="lower", cmap="Blues",
              vmin=0, vmax=heatmap_sim2.max(), aspect="equal",
              extent=(-0.5, l - 0.5, -0.5, l - 0.5))

    # analytique superposé : bulles proportionnelles à la proba
    # taille max = 200 pts² pour rester dans la case (environ 0.8 * taille d'une case)
    xs_ana = []
    ys_ana = []
    ps_ana = []
    for x1 in range(l):
        for y1 in range(l):
            xs_ana.append(x1)
            ys_ana.append(y1)
            ps_ana.append(h_ana[x1, y1])
    ps_ana = np.array(ps_ana)
    tailles = np.where(ps_ana > 0, ps_ana * 200 / ps_ana.max(), 0)
    ax.scatter(xs_ana, ys_ana,
               s=tailles,
               c=ps_ana, cmap="Oranges",
               alpha=0.6,
               zorder=3, edgecolors="none")

    # point de départ
    ax.scatter([x0], [y0], color="#D55E00", s=80, zorder=6, marker="*")

    ax.set_xlim(-0.5, l - 0.5)
    ax.set_ylim(-0.5, l - 0.5)
    ax.set_xticks(range(l))
    ax.set_yticks(range(l))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # légende en dehors du graphique pour ne pas cacher les données
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc="#ADD8E6", label="simulation"),
            plt.scatter([], [], c="orange", s=60, alpha=0.6, label="analytique"),
            plt.scatter([], [], marker="*", c="#D55E00", s=80, label=f"départ ({x0},{y0})")
        ],
        loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, framealpha=0.8
    )
    plt.tight_layout()

    if sauvegarder:
        nom = f"lignee_unique_l{l}_T{T}_m{m:.2f}_joint_xy.png"
        plt.savefig(nom, dpi=150)
        print(f"  Sauvegardé : {nom}")
    if afficher:
        plt.show()
    plt.close(fig)

    # --- Figure 3 : distribution de la distance ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title(
        f"Distance au point de départ - T={T} (t_ana={t_ana:.1f})\n"
        f"grille {l}×{l}, m={m}, {len(positions)} rép.",
        fontsize=9
    )
    d_max = distance(0, 0, l - 1, l - 1)
    bins = np.linspace(0, d_max + 0.5, 25)
    largeur_bin = bins[1] - bins[0]
    centres = (bins[:-1] + bins[1:]) / 2

    # crop au 99e percentile pour ne pas étirer l'axe x inutilement
    x_max = np.percentile(distances_emp, 99.5) + 1.0
    ax.set_xlim(0, x_max)

    ax.hist(distances_emp, bins=bins, density=True,
            color="#0072B2", alpha=0.6, label="simulation (densité)")
    ax.set_xlabel("Distance au départ")
    ax.set_ylabel("Densité (simulation)", color="#0072B2")

    # axe droit pour la probabilité analytique
    ax2 = ax.twinx()
    p_bins = np.zeros(len(centres))
    for x1 in range(l):
        for y1 in range(l):
            d = distance(x0, y0, x1, y1)
            p = proba_transition_analytique((x0, y0), (x1, y1), m, l, t_ana)
            idx = np.searchsorted(bins[1:], d)
            if idx < len(p_bins):
                p_bins[idx] = p_bins[idx] + p
    ax2.plot(centres, p_bins, color="#D55E00", lw=1.5,
             marker="o", markersize=4, label="analytique (proba)")
    ax2.set_ylabel("Probabilité (analytique)", color="#D55E00")

    # aligner les zéros des deux axes
    ax.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)

    lignes1, labels1 = ax.get_legend_handles_labels()
    lignes2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lignes1 + lignes2, labels1 + labels2, fontsize=8)
    plt.tight_layout()

    if sauvegarder:
        nom = f"lignee_unique_l{l}_T{T}_m{m:.2f}_distance.png"
        plt.savefig(nom, dpi=150)
        print(f"  Sauvegardé : {nom}")
    if afficher:
        plt.show()
    plt.close(fig)

        # --- Figure : distribution du nombre de pas ---

    fig, ax = plt.subplots(figsize=(6, 4))

    k_vals, p_vals = distribution_nombre_pas(l, T)

    ax.bar(k_vals, p_vals)
    ax.set_xlim(0, 20)

    ax.set_xlabel("Nombre de fois où la lignée est affectée")
    ax.set_ylabel("Probabilité")
    ax.set_title("Distribution analytique du nombre de pas")

    plt.tight_layout()

    if sauvegarder:
        nom = f"lignee_unique_l{l}_T{T}_m{m:.2f}_nb_pas.png"
        plt.savefig(nom, dpi=150)

    if afficher:
        plt.show()

    plt.close(fig)




# =============================================================================
# Lecture des arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Diffusion d'une lignée unique - Moran spatialisé"
)
parser.add_argument("--l", type=int, default=7,
                    help="Côté de la grille l x l (défaut : 7)")
parser.add_argument("--T", type=int, default=None,
                    help="Nombre de pas Moran. Si absent, calculé automatiquement.")
parser.add_argument("--rep", type=int, default=500,
                    help="Nombre de répétitions (défaut : 500)")
parser.add_argument("--m", type=float, default=1.0,
                    help="Taux de migration (défaut : 1.0)")
parser.add_argument("--x0", type=int, default=None,
                    help="Coordonnée x du point de départ (défaut : centre)")
parser.add_argument("--y0", type=int, default=None,
                    help="Coordonnée y du point de départ (défaut : centre)")
parser.add_argument("--afficher", action="store_true",
                    help="Affiche les graphiques")
parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde les graphiques en .png")
parser.add_argument("--analytique_3d", action="store_true",
                    help="Affiche uniquement la surface 3D analytique, sans simulation")

args = parser.parse_args()


# =============================================================================
# Lancement
# =============================================================================

l   = args.l
m   = args.m
rep = args.rep
n   = l * l

if args.T is not None:
    T = args.T
else:
    T = calculer_T(n, m)
    print(f"T calculé automatiquement : T = {T}")

x0 = args.x0 if args.x0 is not None else l // 2
y0 = args.y0 if args.y0 is not None else l // 2

print(f"Grille {l}×{l} | n={n} | T={T} | rep={rep} | m={m}")
print(f"Point de départ : ({x0}, {y0})")
print()

# mode surface 3D uniquement, pas de simulation
if args.analytique_3d:
    t_ana = T / n
    h_ana = heatmap_analytique((x0, y0), m, l, t_ana)
    xs, ys = np.meshgrid(np.arange(l), np.arange(l))
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection="3d")
    ax3d.plot_surface(xs, ys, h_ana.T, cmap="Blues")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("P(x,y)")
    ax3d.set_title(f"Analytique - t={t_ana:.1f}, grille {l}×{l}, m={m}")
    if args.sauvegarder:
        nom = f"lignee_unique_l{l}_T{T}_m{m:.2f}_analytique_3d.png"
        plt.savefig(nom, dpi=150)
        print(f"  Sauvegardé : {nom}")
    plt.show()
    sys.exit(0)

positions, n_eteintes = simuler(l, T, rep, x0, y0, m)

n_surv = len(positions)
print(f"Lignées survivantes : {n_surv}/{rep} "
      f"({n_eteintes} éteintes avant T={T})")

if n_surv > 0:
    distances_emp = [distance(x0, y0, xi, yi) for (xi, yi) in positions]
    print(f"Distance moyenne au départ : {np.mean(distances_emp):.3f}")
print()

if not args.afficher and not args.sauvegarder:
    print("Pas de sortie graphique demandée (utiliser --afficher ou --sauvegarder).")
    sys.exit(0)

afficher_resultats(positions, x0, y0, m, l, T, n, rep, n_eteintes,
                   args.afficher, args.sauvegarder)