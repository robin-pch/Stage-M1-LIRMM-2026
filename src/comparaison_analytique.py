# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Comparaison analytique vs simulation - distribution P(distance)
# ----------------------------------------------------------------------
# Auteur : Robin Pioch
# Stage M1 Bioinformatique, Université de Montpellier
# Encadrant : Stéphane Guindon (LIRMM)
# Juin 2026
#
# On compare ici la distribution P(distance) de la simulation forward
# avec deux versions analytiques :
#   - sans correction : proba_transition_exp, d=0 inclus
#   - avec correction : proba_transition_exp_cond, d=0 mis a zero puis renormalise
#
# Seul le forward uniforme est utilise (pas de backward, pas de schemas
# cercle/diagonale). Pour chaque tranche de temps on fait un density plot
# (KDE avec scipy.stats.gaussian_kde) de la simu, et on superpose les deux
# courbes analytiques. Pour ca, on tire des valeurs au hasard dans le
# tableau de probas analytique, pour avoir un nuage de points comparable
# a la simu et faisable en KDE.
#
# Usage :
#   python comparaison_analytique.py --l 7 --T 50000 --rep 200 --afficher
#   python comparaison_analytique.py --l 7 --rep 200 --sauvegarder
#   python comparaison_analytique.py --l 7 --rep 200 --afficher --seed 42
#
# Options :
#   --l          : cote de la grille (defaut : 7), population = l*l
#   --T          : nombre de pas. Si absent, calcule automatiquement.
#   --m          : taux de migration (defaut : 1.0)
#   --lam        : lambda = 1/temps de generation (defaut : 1.0)
#   --rep        : nombre de repetitions (defaut : 200)
#   --n_tirages  : nb de valeurs tirees dans l'analytique pour le KDE (defaut : 2000)
#   --afficher   : affiche les graphiques
#   --sauvegarder: sauvegarde les graphiques en .png
#   --seed       : graine pour reproductibilite
# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import argparse
import sys


# =============================================================================
# Classe Noeud (identique a moran_spatialise.py)
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
# Fonctions utilitaires (identiques a moran_spatialise.py)
# =============================================================================

def distance(x1, y1, x2, y2):
    """Distance euclidienne entre deux cases."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def generer_evenements(l, T, m):
    """
    Génère exactement T événements Moran.

    Schéma :
      - on choisit uniformément B parmi les n = l*l cases
      - si B a k voisins :
            u ~ U(0,1)
            si !(u > k*m/4) : migration vers un voisin choisi uniformément
            sinon           : A = B (pas de migration)

    Retourne une liste de T tuples ((xA, yA), (xB, yB)).
    """
    n = l * l
    evenements = []

    indices_B = np.random.randint(0, n, size=2 * T)
    tirages_u = np.random.rand(2 * T)

    i = 0

    while len(evenements) < T:

        if i >= len(indices_B):
            indices_B = np.random.randint(0, n, size=T)
            tirages_u = np.random.rand(T)
            i = 0

        xB, yB = divmod(indices_B[i], l)

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
        u = tirages_u[i]
        i += 1

        if not (u > k * m / 4):
            xA, yA = voisins[np.random.randint(k)]
        else:
            xA, yA = xB, yB

        evenements.append(((xA, yA), (xB, yB)))

    return evenements


def calculer_T(n, m):
    """
    Calcule T à partir de la courbe analytique (loi géométrique, p = m/n^2).

    On cherche T tel que 99.99% des coalescences ont eu lieu avant T,
    puis on multiplie par 1.3 pour avoir une petite marge de sécurité.
    On ne peut pas prendre seuil = 1 exactement car log(0) est indéfini.
    """
    p = m / (n * n)
    seuil = 0.9999
    T = int(np.ceil(np.log(1.0 - seuil) / np.log(1.0 - p)) * 1.3)
    return T


# =============================================================================
# Simulation forward uniforme uniquement
# =============================================================================

def forward_uniforme(l, T, evenements):
    """
    Forward sur toute la grille, sans zone restreinte.
    Retourne (temps_array, distances_array), en pas avant le present.
    """
    # au depart, chaque case est son propre noeud feuille
    grille = [[None] * l for _ in range(l)]
    tous_les_noeuds = []

    for x in range(l):
        for y in range(l):
            noeud = Noeud(x=x, y=y, temps=0)
            grille[x][y] = noeud
            tous_les_noeuds.append(noeud)

    # chaque evenement cree deux nouveaux noeuds, sauf si A=B (rien ne bouge)
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

    # on marque les feuilles encore vivantes a la fin (T)
    for noeud in tous_les_noeuds:
        if noeud.est_feuille:
            noeud.desc1 = 1
            noeud.feuilles1 = [(noeud.x, noeud.y)]

    # parcours a l'envers : les peres sont crees avant leurs fils,
    # donc en remontant la liste on traite toujours les fils avant le pere
    for i in range(len(tous_les_noeuds) - 1, -1, -1):
        noeud = tous_les_noeuds[i]
        if noeud.fils1 is not None:
            noeud.desc1 = noeud.fils1.desc1 + noeud.fils1.desc2
            noeud.desc2 = noeud.fils2.desc1 + noeud.fils2.desc2
            noeud.feuilles1 = noeud.fils1.feuilles1 + noeud.fils1.feuilles2
            noeud.feuilles2 = noeud.fils2.feuilles1 + noeud.fils2.feuilles2

    # on garde les noeuds qui ont au moins un descendant de chaque cote
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

    # on remet le temps "a l'envers" : t=1 correspond au dernier evenement
    temps_array = np.array(temps_liste)
    distances_array = np.array(distances_liste)
    if len(temps_array) > 0:
        temps_array = T - temps_array + 1

    return temps_array, distances_array


# =============================================================================
# Formules analytiques
# =============================================================================

def proba_transition_exp(xy0, xy1, m, l, t, lam):
    """Probabilite de transition a temps continu."""
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


def proba_transition_exp_cond(z0, z1, m, l, t, lam):
    """Proba conditionnelle : met p(z0,z0,t) a zero et renormalise."""
    if z1 == z0:
        return 0.0
    p_z1 = proba_transition_exp(z0, z1, m, l, t, lam)
    p_z0 = proba_transition_exp(z0, z0, m, l, t, lam)
    denom = 1.0 - p_z0
    if denom <= 0:
        return 0.0
    return p_z1 / denom


def densite_temps_coalescence(t, n, m, lam):
    """
    Densite exponentielle du temps de coalescence en temps calendaire.
    """
    parametre = lam / n
    return parametre * np.exp(-parametre * t)


def construire_table_complete(l, m, temps_cal, lam):
    """
    Une ligne par couple (z1, z2) : colonnes x1, y1, x2, y2, d, puis une
    colonne p_cond par temps (p(z1,z2|t)). p_temps contient p(t), une
    valeur par temps (pas par couple).

    temps_cal est le temps de coalescence d'une seule lignee, sans facteur 2.
    proba_transition_exp attend le temps de diffusion cumule des deux
    lignees, donc on lui passe 2*t. densite_temps_coalescence recoit t
    directement, sans ce facteur.

    Retourne (table, p_temps) :
        table : colonnes [x1, y1, x2, y2, d, p_cond_t0, p_cond_t1, ...]
        p_temps : p_temps[i] = p(temps_cal[i])
    """
    lignes = []
    for x1 in range(l):
        for y1 in range(l):
            for x2 in range(l):
                for y2 in range(l):
                    d = distance(x1, y1, x2, y2)
                    p_cond_par_temps = [
                        proba_transition_exp((x1, y1), (x2, y2), m, l, 2 * t, lam) / (l * l)
                        for t in temps_cal
                    ]
                    lignes.append([x1, y1, x2, y2, d] + p_cond_par_temps)

    table = np.array(lignes)
    p_temps = np.array([
        densite_temps_coalescence(t, l * l, m, lam) for t in temps_cal
    ])
    return table, p_temps


def construire_table_complete_cond(l, m, temps_cal, lam):
    """
    Pareil que construire_table_complete mais avec proba_transition_exp_cond.
    Meme histoire de facteur 2 : temps_cal est le temps simple, on passe
    2*t a la fonction de transition.
    """
    lignes = []
    for x1 in range(l):
        for y1 in range(l):
            for x2 in range(l):
                for y2 in range(l):
                    d = distance(x1, y1, x2, y2)
                    p_cond_par_temps = [
                        proba_transition_exp_cond((x1, y1), (x2, y2), m, l, 2 * t, lam) / (l * l)
                        for t in temps_cal
                    ]
                    lignes.append([x1, y1, x2, y2, d] + p_cond_par_temps)

    table = np.array(lignes)
    p_temps = np.array([
        densite_temps_coalescence(t, l * l, m, lam) for t in temps_cal
    ])
    return table, p_temps


def table_vers_Z(table, p_temps, n_temps):
    """
    Reconstruit la matrice compacte Z (pour les figures) a partir de la
    table complete : Z[i,j] = somme sur les couples a distance d_vals[j]
    de p_cond_colonne_i * p_temps[i].

    C'est juste un regroupement par distance, rien n'est perdu dans la
    table elle-meme.
    """
    d_col = table[:, 4]
    distances_possibles = np.array(sorted(set(np.round(d_col, 8))))
    index_d = {d: j for j, d in enumerate(distances_possibles)}

    Z = np.zeros((n_temps, len(distances_possibles)))
    for ligne in table:
        d = round(ligne[4], 8)
        j = index_d[d]
        for i in range(n_temps):
            p_cond = ligne[5 + i]
            Z[i, j] += p_cond * p_temps[i]

    return distances_possibles, Z


def tirer_distances_analytique(d_vals, p_vals, n_tirages):
    """
    Tire n_tirages distances selon la distribution p_vals sur d_vals.
    Sert a faire un KDE comparable a la simulation.
    """
    p_vals = np.clip(p_vals, 0, None)
    total = p_vals.sum()
    if total <= 0:
        return np.array([])
    probas = p_vals / total
    indices = np.random.choice(len(d_vals), size=n_tirages, p=probas)
    return d_vals[indices]


# =============================================================================
# Figures
# =============================================================================

def figure_distance(t_fwd, d_fwd, d_vals, Z_brute, Z_cond,
                    bins_t, centres_t, bins_retenus,
                    l, T, m, lam, afficher, sauvegarder):
    """
    Figure 1 : P(distance) par tranche de temps, courbes discretes.

    Pour chaque tranche on trace la simulation (pointille) et les deux
    courbes analytiques (trait plein), sur les distances exactes.
    """
    fig2, axes2 = plt.subplots(1, len(bins_retenus),
                               figsize=(4 * len(bins_retenus), 4),
                               sharey=False)
    if len(bins_retenus) == 1:
        axes2 = [axes2]
    fig2.suptitle(
        f"P(distance) selon le temps : analytique vs simulation - "
        f"grille {l}x{l}, m={m}, lam={lam}",
        fontsize=10
    )

    n = l * l
    t_fwd_cal = t_fwd * m / (n * lam)

    print("\nErreurs quadratiques (comparaison avec simulation) :")
    for k, i in enumerate(bins_retenus):
        ax2 = axes2[k]
        t_label = f"t={centres_t[i]:.1f}"

        # analytique : toutes les distances, y compris d=0
        masque_d = d_vals >= 0.0
        p_brute = Z_brute[i, masque_d].copy()
        p_cond = Z_cond[i, masque_d].copy()
        d_vals_plot = d_vals[masque_d]

        # simulation : histogramme sur les memes distances exactes
        masque_t = (t_fwd_cal >= bins_t[i]) & (t_fwd_cal < bins_t[i + 1])
        d_sim = d_fwd[masque_t]
        p_sim = np.zeros(len(d_vals_plot))
        for val in d_sim:
            val_r = round(val, 8)
            idx_d = np.searchsorted(d_vals_plot, val_r)
            if idx_d < len(d_vals_plot) and d_vals_plot[idx_d] == val_r:
                p_sim[idx_d] += 1
        # normalisation : ramener la simulation sur la meme echelle que l'analytique
        s_sim = p_sim.sum()
        s_brute = p_brute.sum()
        if s_sim > 0 and s_brute > 0:
            p_sim = p_sim / s_sim * s_brute

        # erreur quadratique (d=0 exclu pour comparaison equitable)
        masque_sans_zero = d_vals_plot >= 1.0
        err_brute = np.sum((p_brute[masque_sans_zero] - p_sim[masque_sans_zero]) ** 2)
        err_cond = np.sum((p_cond[masque_sans_zero] - p_sim[masque_sans_zero]) ** 2)
        print(f"  t={centres_t[i]:.1f} (n={masque_t.sum()} paires) : "
              f"sans correction = {err_brute:.6e}  |  avec correction = {err_cond:.6e}")

        ax2.plot(d_vals_plot, p_brute, color="steelblue", lw=2, marker="o",
                 markersize=3, label="ana. sans correction")
        ax2.plot(d_vals_plot, p_cond, color="forestgreen", lw=2, marker="o",
                 markersize=3, label="ana. avec correction")
        ax2.plot(d_vals_plot, p_sim, color="darkorange", lw=1.5, ls="--",
                 marker="o", markersize=3, label=f"sim (n={masque_t.sum()})")
        ax2.set_title(t_label, fontsize=9)
        ax2.set_xlabel("Distance")
        if k == 0:
            ax2.set_ylabel("Probabilité")
        ax2.legend(fontsize=7)

    plt.tight_layout()
    if sauvegarder:
        nom = f"comparaison_ana_l{l}_T{T}_distance.png"
        plt.savefig(nom, dpi=150)
        print(f"  Graphique sauvegarde : {nom}")
    if afficher:
        plt.show()
    plt.close(fig2)


def preparer_tirages(t_fwd, d_fwd, d_vals, Z_brute, Z_cond, p_temps,
                     bins_t, centres_t, n_tirages, l, m, lam):
    """
    Tire des distances analytiques (brute et cond) pour chaque tranche de
    temps. Le nombre total de tirages (n_tirages) est reparti entre les
    tranches selon p_temps : une tranche avec un p(t) faible recoit peu de
    tirages, une tranche avec un p(t) fort en recoit beaucoup. Avant, chaque
    tranche recevait le meme nombre de tirages, ce qui etalait le nuage
    analytique sur tout l'axe du temps alors que la masse de probabilite
    est surtout concentree au debut.

    Retourne les tableaux (t, d) pour simulation, brute et cond.
    """
    n = l * l
    t_fwd_cal = t_fwd * m / (n * lam)

    t_tire_brute = []
    d_tire_brute = []
    t_tire_cond = []
    d_tire_cond = []

    poids_temps = p_temps / p_temps.sum()
    n_tirer_par_tranche = np.round(poids_temps * n_tirages).astype(int)

    for i in range(len(centres_t)):
        n_tirer = n_tirer_par_tranche[i]
        if n_tirer == 0:
            continue

        d_b = tirer_distances_analytique(d_vals, Z_brute[i].copy(), n_tirer)
        d_c = tirer_distances_analytique(d_vals, Z_cond[i].copy(), n_tirer)

        # temps tirés uniformément dans la tranche pour éviter les colonnes discrètes
        t_b = np.random.uniform(bins_t[i], bins_t[i + 1], size=len(d_b))
        t_c = np.random.uniform(bins_t[i], bins_t[i + 1], size=len(d_c))

        t_tire_brute.extend(t_b)
        d_tire_brute.extend(d_b)
        t_tire_cond.extend(t_c)
        d_tire_cond.extend(d_c)

    return (
        t_fwd_cal,
        np.array(t_tire_brute), np.array(d_tire_brute),
        np.array(t_tire_cond),  np.array(d_tire_cond),
    )


def figure_density(t_fwd, d_fwd, d_vals, Z_brute, Z_cond, p_temps,
                   bins_t, centres_t, n_tirages,
                   l, T, m, lam, afficher, sauvegarder):
    """
    Figure 2 : scatter coloré par densite locale (vrai density chart).

    Pour chaque point (t, d) on estime la densite locale avec gaussian_kde
    et on colore le scatter en fonction. Trois panneaux : simulation,
    analytique sans correction, analytique avec correction.

    On sous-echantillonne a MAX_SCATTER points pour que le KDE reste rapide.
    """
    MAX_SCATTER = 10000

    n = l * l

    t_fwd_cal, t_tire_brute, d_tire_brute, t_tire_cond, d_tire_cond = \
        preparer_tirages(t_fwd, d_fwd, d_vals, Z_brute, Z_cond, p_temps,
                         bins_t, centres_t, n_tirages, l, m, lam)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Density chart (temps, distance) : simulation vs analytique - "
        f"grille {l}x{l}, m={m}, lam={lam}",
        fontsize=10
    )

    donnees = [
        (t_fwd_cal,    d_fwd,        "Oranges", f"Forward simulation (n={len(t_fwd)} paires)"),
        (t_tire_brute, d_tire_brute, "Blues",   "Analytique sans correction"),
        (t_tire_cond,  d_tire_cond,  "Greens",  "Analytique avec correction"),
    ]

    for ax, (dt, dd, cmap, titre) in zip(axes, donnees):
        if len(dt) > 1:
            # sous-échantillonnage si trop de points (KDE en O(n²))
            if len(dt) > MAX_SCATTER:
                idx = np.random.choice(len(dt), size=MAX_SCATTER, replace=False)
                dt_plot = dt[idx]
                dd_plot = dd[idx]
            else:
                dt_plot = dt
                dd_plot = dd

            # estimation de la densité locale en chaque point
            kde = gaussian_kde(np.vstack([dt_plot, dd_plot]))
            couleurs = kde(np.vstack([dt_plot, dd_plot]))
            # tri par densité croissante pour que les points denses soient au premier plan
            ordre = couleurs.argsort()
            ax.scatter(dt_plot[ordre], dd_plot[ordre], c=couleurs[ordre],
                       cmap=cmap, s=2, linewidths=0)
        ax.set_title(titre, fontsize=9)
        ax.set_xlabel("Temps calendaire (t1+t2)")
        ax.set_ylabel("Distance")

    plt.tight_layout()
    if sauvegarder:
        nom = f"comparaison_ana_l{l}_T{T}_density.png"
        plt.savefig(nom, dpi=150)
        print(f"  Graphique sauvegarde : {nom}")
    if afficher:
        plt.show()
    plt.close(fig)


def figure_density_seaborn(t_fwd, d_fwd, d_vals, Z_brute, Z_cond, p_temps,
                           bins_t, centres_t, n_tirages,
                           l, T, m, lam, afficher, sauvegarder):
    """
    Figure 3 : contours KDE 2D avec seaborn (sans fill, pour rester rapide).
    Meme sous-echantillonnage que figure_density, meme raison.
    """
    MAX_SCATTER = 10000

    n = l * l

    t_fwd_cal, t_tire_brute, d_tire_brute, t_tire_cond, d_tire_cond = \
        preparer_tirages(t_fwd, d_fwd, d_vals, Z_brute, Z_cond, p_temps,
                         bins_t, centres_t, n_tirages, l, m, lam)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Density chart seaborn (temps, distance) : simulation vs analytique - "
        f"grille {l}x{l}, m={m}, lam={lam}",
        fontsize=10
    )

    donnees = [
        (t_fwd_cal,    d_fwd,        "Oranges", f"Forward simulation (n={len(t_fwd)} paires)"),
        (t_tire_brute, d_tire_brute, "Blues",   "Analytique sans correction"),
        (t_tire_cond,  d_tire_cond,  "Greens",  "Analytique avec correction"),
    ]

    for ax, (dt, dd, cmap, titre) in zip(axes, donnees):
        if len(dt) > 1:
            if len(dt) > MAX_SCATTER:
                idx = np.random.choice(len(dt), size=MAX_SCATTER, replace=False)
                dt_plot = dt[idx]
                dd_plot = dd[idx]
            else:
                dt_plot = dt
                dd_plot = dd
            sns.kdeplot(x=dt_plot, y=dd_plot, ax=ax, cmap=cmap, fill=False, thresh=0.05)
        ax.set_title(titre, fontsize=9)
        ax.set_xlabel("Temps calendaire (t1+t2)")
        ax.set_ylabel("Distance")

    plt.tight_layout()
    if sauvegarder:
        nom = f"comparaison_ana_l{l}_T{T}_density_seaborn.png"
        plt.savefig(nom, dpi=150)
        print(f"  Graphique sauvegarde : {nom}")
    if afficher:
        plt.show()
    plt.close(fig)


def figure_heatmap(t_fwd, d_fwd, d_vals, Z_brute, Z_cond,
                   bins_t, centres_t,
                   l, T, m, lam, afficher, sauvegarder):
    """
    Heatmap (temps, distance) avec la vraie densite jointe
    p(z1,z2,t) = p(z1,z2|t) * p(t). On regroupe les distances exactes en
    bins entiers, par somme (pas par moyenne). 3 panneaux :
    simulation, analytique sans correction, analytique avec correction.

    On garde seulement les 3/4 de la plage de temps.
    """
    n = l * l
    t_fwd_cal = t_fwd * m / (n * lam)

    d_max_int = int(d_vals.max()) + 1
    bins_d = np.arange(0, d_max_int + 1, 1)  # commence a 0 : d=0 inclus

    # regroupement des distances exactes en bins entiers, par somme
    def regrouper_par_bin(Z):
        Z_int = np.zeros((Z.shape[0], len(bins_d) - 1))
        for j, d in enumerate(d_vals):
            bin_j = np.searchsorted(bins_d, d, side="right") - 1
            if 0 <= bin_j < len(bins_d) - 1:
                Z_int[:, bin_j] += Z[:, j]
        return Z_int

    Z_brute_int = regrouper_par_bin(Z_brute)
    Z_cond_int = regrouper_par_bin(Z_cond)

    # troncature douce : 3/4 de l'axe, pas la moitie
    n_garde = int(len(centres_t) * 0.75)
    bins_t_plot = bins_t[:n_garde + 1]
    Z_brute_int = Z_brute_int[:n_garde]
    Z_cond_int = Z_cond_int[:n_garde]
    t_max_plot = bins_t_plot[-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Heatmap (temps, distance) : simulation vs analytique - "
        f"grille {l}x{l}, m={m}, lam={lam}",
        fontsize=10
    )

    # panneau 1 : simulation (on filtre aussi la simulation sur la meme plage)
    ax = axes[0]
    masque_t_plot = t_fwd_cal <= t_max_plot
    H, _, _ = np.histogram2d(
        t_fwd_cal[masque_t_plot], d_fwd[masque_t_plot],
        bins=[bins_t_plot, bins_d], density=True
    )
    ax.pcolormesh(bins_t_plot, bins_d, H.T, cmap="Oranges", shading="auto")
    ax.set_xlabel("Temps calendaire (t1+t2)")
    ax.set_ylabel("Distance")
    ax.set_title(f"Forward simulation (n={masque_t_plot.sum()} paires)")

    # panneau 2 : analytique sans correction
    ax = axes[1]
    ax.pcolormesh(bins_t_plot, bins_d, Z_brute_int.T, cmap="Blues", shading="auto")
    ax.set_xlabel("Temps calendaire (t1+t2)")
    ax.set_ylabel("Distance")
    ax.set_title("Analytique sans correction")

    # panneau 3 : analytique avec correction
    ax = axes[2]
    ax.pcolormesh(bins_t_plot, bins_d, Z_cond_int.T, cmap="Greens", shading="auto")
    ax.set_xlabel("Temps calendaire (t1+t2)")
    ax.set_ylabel("Distance")
    ax.set_title("Analytique avec correction")

    plt.tight_layout()
    if sauvegarder:
        nom = f"comparaison_ana_l{l}_T{T}_heatmap.png"
        plt.savefig(nom, dpi=150)
        print(f"  Graphique sauvegarde : {nom}")
    if afficher:
        plt.show()
    plt.close(fig)


def afficher_comparaison(t_fwd, d_fwd, l, T, m, lam, n_tirages, n_temps,
                         afficher, sauvegarder):
    """Lance les figures."""
    n = l * l

    t_fwd_cal = t_fwd * m / (n * lam)

    temps_moran = np.array([T * k / n_temps for k in range(1, n_temps + 1)])
    temps_cal = temps_moran * m / (n * lam)

    bins_t = np.linspace(0, temps_cal.max(), n_temps + 1)
    centres_t = (bins_t[:-1] + bins_t[1:]) / 2

    print("Calcul de la table complete (sans correction)...")
    table_brute, p_temps = construire_table_complete(l, m, centres_t, lam)
    print("Calcul de la table complete (avec correction)...")
    table_cond, _ = construire_table_complete_cond(l, m, centres_t, lam)

    n_temps_reel = len(centres_t)
    d_vals, Z_brute = table_vers_Z(table_brute, p_temps, n_temps_reel)
    _, Z_cond = table_vers_Z(table_cond, p_temps, n_temps_reel)

    # selectionner au plus 4 tranches avec assez de paires en simulation
    bins_retenus = []
    for i in range(len(centres_t)):
        masque = (t_fwd_cal >= bins_t[i]) & (t_fwd_cal < bins_t[i + 1])
        if masque.sum() >= 100:
            bins_retenus.append(i)
    if len(bins_retenus) > 4:
        pas = len(bins_retenus) // 4
        bins_retenus = bins_retenus[::pas][:4]

    # figure_distance est mise de cote pour le moment : elle compare une
    # quantite conditionnelle (simulation) a une densite jointe (analytique,
    # depuis l'ajout de p(t)), donc plus vraiment comparable directement.
    # if len(bins_retenus) > 0:
    #     figure_distance(t_fwd, d_fwd, d_vals, Z_brute, Z_cond,
    #                     bins_t, centres_t, bins_retenus,
    #                     l, T, m, lam, afficher, sauvegarder)

    figure_heatmap(t_fwd, d_fwd, d_vals, Z_brute, Z_cond,
                   bins_t, centres_t,
                   l, T, m, lam, afficher, sauvegarder)

    figure_density(t_fwd, d_fwd, d_vals, Z_brute, Z_cond, p_temps,
                   bins_t, centres_t, n_tirages,
                   l, T, m, lam, afficher, sauvegarder)

    figure_density_seaborn(t_fwd, d_fwd, d_vals, Z_brute, Z_cond, p_temps,
                           bins_t, centres_t, n_tirages,
                           l, T, m, lam, afficher, sauvegarder)


# =============================================================================
# Lecture des arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Comparaison P(distance) : simulation forward vs analytique (avec/sans correction)"
)

parser.add_argument("--l", type=int, default=7,
                    help="Cote de la grille l x l (defaut : 7)")

parser.add_argument("--T", type=int, default=None,
                    help="Nombre de pas. Si absent, calcule automatiquement.")

parser.add_argument("--m", type=float, default=1.0,
                    help="Taux de migration (defaut : 1.0)")

parser.add_argument("--lam", type=float, default=1.0,
                    help="lambda = 1/temps de generation (defaut : 1.0)")

parser.add_argument("--rep", type=int, default=200,
                    help="Nombre de repetitions (defaut : 200)")

parser.add_argument("--n_tirages", type=int, default=2000,
                    help="Nb de valeurs tirees dans l'analytique pour le KDE (defaut : 2000)")

parser.add_argument("--n_temps", type=int, default=50,
                    help="Nb de tranches de temps pour le tableau analytique (defaut : 50)")

parser.add_argument("--afficher", action="store_true",
                    help="Affiche les graphiques")

parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde les graphiques en .png")

parser.add_argument("--seed", type=int, default=None,
                    help="Graine pour reproductibilite")

args = parser.parse_args()


# =============================================================================
# Lancement
# =============================================================================

n = args.l * args.l

if args.T is not None:
    T = args.T
else:
    T = calculer_T(n, args.m)
    print(f"T calcule automatiquement : T = {T}")

print(f"Grille {args.l}x{args.l} | n={n} | T={T} | m={args.m} | lam={args.lam}")
print(f"Simulation forward (T={T}, rep={args.rep})...")

if args.seed is not None:
    np.random.seed(args.seed)
    print(f"Graine fixee : {args.seed}")

# simulation : on accumule les temps et distances sur toutes les repetitions
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

print(f"\nSimulation terminee : {len(t_fwd)} paires au total")

if len(t_fwd) == 0:
    print("Aucune paire. Augmenter T ou rep.")
    sys.exit(0)

if not args.afficher and not args.sauvegarder:
    print("Rien a afficher (utiliser --afficher ou --sauvegarder).")
    sys.exit(0)

afficher_comparaison(
    t_fwd, d_fwd,
    args.l, T, args.m, args.lam, args.n_tirages, args.n_temps,
    args.afficher, args.sauvegarder
)