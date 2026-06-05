# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Processus de Moran spatialisé - comparaison forward / backward
# ----------------------------------------------------------------------
# Auteur : Robin Pioch
# Stage M1 Bioinformatique, Université de Montpellier
# Encadrant : Stéphane Guindon (LIRMM)
# Mai-Juin 2026
#
# On simule un processus de Moran spatialisé sur une grille l x l
# (population n = l*l). À chaque pas, on tire B au hasard sur la grille.
# B se propage sur A : B écrase A. La destination A est choisie avec
# probabilité m/4 par direction disponible, et 1 - k*m/4 de rester sur
# place (k = nb de voisins de B : 4 au centre, 3 sur un bord, 2 en angle).
# Les événements "rester sur place" (A = B) sont enregistrés comme les
# autres. Dans forward, ils ne créent pas de noeud (continue). Dans
# backward, ils ne déplacent aucune lignée.
#
# Pour chaque répétition, on génère les événements une seule fois, on
# fait le forward (qui donne n_paires), puis le backward trois fois
# (uniforme, cercle, diagonale) en tirant ce même nombre de paires.
#
# Usage :
#   python moran_spatialise.py --l 7 --T 50000 --rep 200 --afficher
#   python moran_spatialise.py --l 7 --rep 200 --afficher
#   python moran_spatialise.py --l 7 --rep 200 --sauvegarder
#
# Options :
#   --l          : côté de la grille (défaut : 7), population = l*l
#   --T          : nombre de pas. Si absent, calculé automatiquement.
#   --m          : taux de migration (défaut : 1.0)
#   --rep        : nombre de répétitions (défaut : 200)
#   --sigma      : écart-type pour le schéma diagonale (défaut : 1.0)
#   --rayon      : rayon pour le schéma cercle (défaut : l/4)
#   --quantile   : percentile pour le crop des histogrammes (défaut : 99)
#   --afficher   : affiche les graphiques
#   --sauvegarder: sauvegarde les graphiques en .png
# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys


# =============================================================================
# Classe Noeud
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
# Fonctions utilitaires
# =============================================================================

def generer_evenements(l, T, m):
    """
    Génère exactement T événements Moran.

    Les tirages aléatoires sont pré-générés en avance par blocs (plus rapide
    que T appels numpy individuels). Les événements "rester sur place" (A = B)
    sont inclus : ils comptent comme un pas de temps où rien ne se passe.

    Retourne une liste de T tuples ((xA, yA), (xB, yB)).
    """
    evenements = []
    # On tire 4*T valeurs d'un coup comme marge, au cas où beaucoup tombent
    # sur "rester sur place"
    indices_B = np.random.randint(0, l * l, size=T * 4)
    tirages_r = np.random.rand(T * 4)
    i = 0

    while len(evenements) < T:
        if i >= len(indices_B):
            # si par malchance la marge n'était pas suffisante, on régénère
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


def distance(x1, y1, x2, y2):
    """Distance euclidienne entre deux cases."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def trouver_case_plus_proche(l, px, py):
    """
    Projette un point continu (px, py) sur la grille en arrondissant
    et en clippant pour rester dans les bornes.
    """
    x = int(np.clip(round(px), 0, l - 1))
    y = int(np.clip(round(py), 0, l - 1))
    return x, y


def case_dans_zone(l, x, y, schema, sigma=1.0, rayon=None):
    """
    Retourne True si la case (x, y) est dans la zone du schéma.

    uniforme  : toutes les cases
    cercle    : distance au centre <= rayon
    diagonale : |y - x| <= sigma
    """
    if rayon is None:
        rayon = l / 4.0

    if schema == "uniforme":
        return True

    if schema == "cercle":
        cx, cy = (l - 1) / 2.0, (l - 1) / 2.0
        return distance(x, y, cx, cy) <= rayon

    if schema == "diagonale":
        return abs(y - x) <= sigma

    return True


def cases_dans_zone(l, schema, sigma=1.0, rayon=None):
    """
    Retourne la liste de toutes les cases dans la zone du schéma.

    Calculée une seule fois avant la boucle des répétitions, pour ne pas
    refaire ce travail à chaque paire tirée dans backward.
    """
    cases = []
    for x in range(l):
        for y in range(l):
            if case_dans_zone(l, x, y, schema, sigma=sigma, rayon=rayon):
                cases.append((x, y))
    return cases


def tirer_paire(cases_valides):
    """
    Tire deux cases distinctes au hasard dans la liste des cases valides.

    Le petit trick sur j évite d'obtenir i == j sans introduire de biais
    dans le tirage.
    """
    n = len(cases_valides)
    i = np.random.randint(0, n)
    j = np.random.randint(0, n - 1)
    if j >= i:
        j = j + 1
    x1, y1 = cases_valides[i]
    x2, y2 = cases_valides[j]
    return x1, y1, x2, y2


# =============================================================================
# Choix automatique de T
# =============================================================================

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
# Simulation FORWARD
# =============================================================================

def forward(l, T, evenements, schema="uniforme", sigma=1.0, rayon=None):
    """
    Calcule p(t) en forward en construisant un arbre de descendants.

    À chaque événement (B se propage sur A), l'occupant de B bifurque et
    devient un noeud interne avec deux fils (un en A, un en B).

    À la fin, on parcourt la liste à l'envers pour calculer le nombre de
    descendants vivants de chaque côté (post-ordre itératif — la version
    récursive provoquait des RecursionError pour T > ~500).

    Pour un schéma non-uniforme, seules les feuilles dans la zone comptent
    pour valider un noeud. Les distances ne sont calculées qu'entre feuilles
    dans la zone. n_in et n_total servent au facteur de normalisation.

    Les temps sont convertis depuis le repère absolu vers "pas avant le
    présent" (t=1 = dernier pas), pour cohérence avec backward.

    Retourne (temps_array, distances_array, n_paires, n_in, n_total).
    """
    # initialisation : une case = un noeud feuille
    grille = [[None] * l for _ in range(l)]
    tous_les_noeuds = []

    for x in range(l):
        for y in range(l):
            noeud = Noeud(x=x, y=y, temps=0)
            grille[x][y] = noeud
            tous_les_noeuds.append(noeud)

    # avancer dans le temps : chaque événement crée deux nouveaux noeuds
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

    # marquer les feuilles vivantes (celles encore dans la grille à T)
    for noeud in tous_les_noeuds:
        if noeud.est_feuille:
            noeud.desc1 = 1
            noeud.feuilles1 = [(noeud.x, noeud.y)]
            noeud.est_dans_zone = case_dans_zone(
                l, noeud.x, noeud.y, schema, sigma=sigma, rayon=rayon
            )
            if noeud.est_dans_zone:
                noeud.feuilles1_zone = [(noeud.x, noeud.y)]

    # post-ordre itératif : les pères sont créés avant leurs fils,
    # donc parcourir à l'envers revient à traiter les fils en premier
    for i in range(len(tous_les_noeuds) - 1, -1, -1):
        noeud = tous_les_noeuds[i]
        if noeud.fils1 is not None:
            noeud.desc1 = noeud.fils1.desc1 + noeud.fils1.desc2
            noeud.desc2 = noeud.fils2.desc1 + noeud.fils2.desc2
            noeud.feuilles1 = noeud.fils1.feuilles1 + noeud.fils1.feuilles2
            noeud.feuilles2 = noeud.fils2.feuilles1 + noeud.fils2.feuilles2
            noeud.feuilles1_zone = noeud.fils1.feuilles1_zone + noeud.fils1.feuilles2_zone
            noeud.feuilles2_zone = noeud.fils2.feuilles1_zone + noeud.fils2.feuilles2_zone

    # collecte des noeuds valides : desc1 >= 1 et desc2 >= 1,
    # avec au moins une feuille dans la zone de chaque côté
    temps_zone = []
    distances_zone = []
    n_in = 0
    n_total = 0

    for noeud in tous_les_noeuds:
        if noeud.fils1 is not None:
            if noeud.desc1 >= 1 and noeud.desc2 >= 1:
                for (xa, ya) in noeud.feuilles1:
                    for (xb, yb) in noeud.feuilles2:
                        d = distance(xa, ya, xb, yb)
                        in_zone = (
                            case_dans_zone(l, xa, ya, schema, sigma=sigma, rayon=rayon)
                            and
                            case_dans_zone(l, xb, yb, schema, sigma=sigma, rayon=rayon)
                        )
                        n_total = n_total + 1
                        if in_zone:
                            n_in = n_in + 1
                            temps_zone.append(noeud.temps)
                            distances_zone.append(d)

    n_paires = len(temps_zone)

    # conversion en "pas avant le présent" : t=1 = dernier événement
    temps_array = np.array(temps_zone)
    distances_array = np.array(distances_zone)
    if len(temps_array) > 0:
        temps_array = T - temps_array + 1

    return temps_array, distances_array, n_paires, n_in, n_total


# =============================================================================
# Simulation BACKWARD
# =============================================================================

def backward(l, T, n_paires, evenements, cases_valides, schema="uniforme",
             d0_min=0.0, d0_max=float("inf"), sigma=1.0, rayon=None):
    """
    Simule la coalescence de n_paires paires en remontant les événements.

    On tire deux cases au présent, puis on remonte : si une case A a été
    écrasée par B à un certain pas, la lignée en A vient de B. Quand les
    deux lignées se retrouvent sur la même case, elles ont coalescé.

    t=1 = dernier événement (1 pas avant le présent),
    t=T = premier événement.

    Retourne (temps_array, distances_array, n_non_coal).
    """
    temps_liste = []
    distances_liste = []
    n_non_coal = 0

    for _ in range(n_paires):

        x1, y1, x2, y2 = tirer_paire(cases_valides)
        d0 = distance(x1, y1, x2, y2)

        coalesce = False

        for k in range(T - 1, -1, -1):
            A, B = evenements[k]

            if A == (x1, y1):
                x1, y1 = B
            if A == (x2, y2):
                x2, y2 = B

            if (x1, y1) == (x2, y2):
                temps_liste.append(T - k)
                distances_liste.append(d0)
                coalesce = True
                break

        if not coalesce:
            n_non_coal = n_non_coal + 1

    return np.array(temps_liste), np.array(distances_liste), n_non_coal


# =============================================================================
# Affichage
# =============================================================================

def afficher_resultats(resultats_fwd, resultats_bwd, facteurs_norm,
                       l, T, m, quantile, afficher, sauvegarder, sigma=1.0, rayon=None):
    """
    Produit les figures de comparaison forward / backward.

    Figure 1 : forward uniforme vs backward uniforme + courbe analytique.
    Figure 2 : distributions jointes (temps, distance) pour l'uniforme.
    Figures 3+ : pour cercle et diagonale :
        A) forward schéma vs forward uniforme (gris)
        B) backward schéma vs backward uniforme (gris)
        C+D) distributions jointes forward et backward côte à côte

    L'axe des temps des histogrammes est coupé au percentile `quantile`
    pour ne pas écraser la zone dense. L'axe des distances est commun
    à toutes les figures jointes pour faciliter la comparaison.
    """
    Q = quantile
    n = l * l

    t_fwd_unif = resultats_fwd["uniforme"]["temps"]
    d_fwd_unif = resultats_fwd["uniforme"]["distances"]
    t_bwd_unif = resultats_bwd["uniforme"]["temps"]
    d_bwd_unif = resultats_bwd["uniforme"]["distances"]

    # bins communs pour les marginales, coupés au percentile Q
    all_temps = []
    for schema in resultats_bwd:
        if len(resultats_bwd[schema]["temps"]) > 0:
            all_temps.extend(resultats_bwd[schema]["temps"])
    for schema in resultats_fwd:
        if len(resultats_fwd[schema]["temps"]) > 0:
            all_temps.extend(resultats_fwd[schema]["temps"])
    t_max = np.percentile(all_temps, Q) if len(all_temps) > 0 else T
    bins = np.linspace(0, t_max, 50)
    centres_bins = (bins[:-1] + bins[1:]) / 2
    largeur_bin = bins[1] - bins[0]

    # axe distance commun pour toutes les jointes
    d_max_joint_global = 0.0
    for schema in ["uniforme", "cercle", "diagonale"]:
        for res in [resultats_fwd, resultats_bwd]:
            d = res[schema]["distances"]
            if len(d) > 0:
                q = np.percentile(d, Q)
                if q > d_max_joint_global:
                    d_max_joint_global = q
    d_max_joint_global = d_max_joint_global + 0.5

    # --- Figure 1 : forward uniforme vs backward uniforme ---
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    fig1.suptitle(
        f"Forward vs backward uniforme - grille {l}x{l} (n={n}), T={T}",
        fontsize=11
    )

    if len(t_fwd_unif) > 0:
        ax1.hist(t_fwd_unif, bins=bins, density=True, alpha=0.5,
                 color="#D55E00", edgecolor="white",
                 label=f"Forward uniforme (n={len(t_fwd_unif)})")
        ax1.axvline(t_fwd_unif.mean(), color="#D55E00", lw=1.5, ls="--",
                    label=f"moy. forward uniforme = {t_fwd_unif.mean():.0f}")

    if len(t_bwd_unif) > 0:
        ax1.hist(t_bwd_unif, bins=bins, density=True, alpha=0.5,
                 color="#0072B2", edgecolor="white",
                 label=f"Backward uniforme (n={len(t_bwd_unif)})")
        ax1.axvline(t_bwd_unif.mean(), color="#0072B2", lw=1.5, ls="--",
                    label=f"moy. bwd uniforme = {t_bwd_unif.mean():.0f}")

    # courbe analytique : p(t) = (1 - m/n^2)^(t-1) * m/n^2
    p_coal = m / n**2
    p_t_unif = (1 - p_coal) ** (centres_bins - 1) * p_coal
    p_t_unif = p_t_unif / (p_t_unif.sum() * largeur_bin)
    ax1.plot(centres_bins, p_t_unif, color="darkgreen", lw=1.5, ls="-.",
             label=f"analytique : $m/n^2={p_coal:.2e}$")

    ax1.legend(fontsize=9, framealpha=0.5)
    ax1.set_xlabel("Temps (pas de Moran, t=1 = présent)")
    ax1.set_ylabel("Densité")

    print(f"  Référence analytique : m/n^2 = {p_coal:.2e} (n={n}, grille {l}x{l})")
    if len(t_fwd_unif) > 0:
        print(f"  Forward uniforme : moy = {t_fwd_unif.mean():.1f}, écart-type = {t_fwd_unif.std():.1f}")
    if len(t_bwd_unif) > 0:
        print(f"  Bwd uniforme     : moy = {t_bwd_unif.mean():.1f}, écart-type = {t_bwd_unif.std():.1f}")

    plt.tight_layout()
    if sauvegarder:
        nom = f"moran_l{l}_T{T}_uniforme.png"
        plt.savefig(nom, dpi=150)
        print(f"  Graphique sauvegardé : {nom}")
    if afficher:
        plt.show()
    plt.close(fig1)

    # --- Figure 2 : jointes uniforme ---
    n_sous_figures = 0
    if len(t_bwd_unif) > 0 and len(d_bwd_unif) > 0:
        n_sous_figures = n_sous_figures + 1
    if len(t_fwd_unif) > 0 and len(d_fwd_unif) > 0:
        n_sous_figures = n_sous_figures + 1

    if n_sous_figures > 0:
        fig2, axes = plt.subplots(1, n_sous_figures, figsize=(6 * n_sous_figures, 5))
        if n_sous_figures == 1:
            axes = [axes]

        fig2.suptitle(
            f"Distributions jointes uniforme - grille {l}x{l} (n={n}), T={T}",
            fontsize=11
        )
        idx = 0

        if len(t_bwd_unif) > 0 and len(d_bwd_unif) > 0:
            ax = axes[idx]
            idx = idx + 1
            t_max_joint = np.percentile(t_bwd_unif, Q)
            bins_t = np.linspace(0, t_max_joint, 40)
            bins_d = np.linspace(0, d_max_joint_global, 30)
            ax.hist2d(t_bwd_unif, d_bwd_unif, bins=[bins_t, bins_d],
                      cmap="Blues", density=True)
            ax.set_xlabel("Temps de coalescence t")
            ax.set_ylabel("Distance initiale d0")
            ax.set_title(f"Backward uniforme (n={len(t_bwd_unif)} points)")
            ax.axhline(d_bwd_unif.mean(), color="#0072B2", lw=1.2, ls="--",
                       label=f"moy. d0 = {d_bwd_unif.mean():.2f}")
            ax.legend(fontsize=8)

        if len(t_fwd_unif) > 0 and len(d_fwd_unif) > 0:
            ax = axes[idx]
            t_max_joint = np.percentile(t_fwd_unif, Q)
            bins_t = np.linspace(0, t_max_joint, 40)
            bins_d = np.linspace(0, d_max_joint_global, 30)
            ax.hist2d(t_fwd_unif, d_fwd_unif, bins=[bins_t, bins_d],
                      cmap="Oranges", density=True)
            ax.set_xlabel("Temps de divergence t")
            ax.set_ylabel("Distance entre feuilles (gauche x droite)")
            ax.set_title(f"Forward uniforme (n={len(t_fwd_unif)} paires)")
            ax.axhline(d_fwd_unif.mean(), color="#D55E00", lw=1.2, ls="--",
                       label=f"moy. dist = {d_fwd_unif.mean():.2f}")
            ax.legend(fontsize=8)

        plt.tight_layout()
        if sauvegarder:
            nom = f"moran_l{l}_T{T}_uniforme_joint.png"
            plt.savefig(nom, dpi=150)
            print(f"  Graphique sauvegardé : {nom}")
        if afficher:
            plt.show()
        plt.close(fig2)

    # --- Figures 3+ : cercle et diagonale ---
    for schema in ["cercle", "diagonale"]:
        t_bwd = resultats_bwd[schema]["temps"]
        d_bwd = resultats_bwd[schema]["distances"]
        t_fwd = resultats_fwd[schema]["temps"]
        d_fwd = resultats_fwd[schema]["distances"]

        if len(t_bwd) == 0 and len(t_fwd) == 0:
            continue

        facteur = facteurs_norm.get(schema, None)
        facteur_str = f"{facteur:.3f}" if facteur is not None else "N/A"

        if len(t_bwd) > 0:
            print(f"  Bwd {schema} : moy = {t_bwd.mean():.1f}, écart-type = {t_bwd.std():.1f}")
        if len(t_fwd) > 0:
            print(f"  Fwd {schema} : moy = {t_fwd.mean():.1f}, écart-type = {t_fwd.std():.1f}")
        print(f"  Facteur de normalisation ({schema}) = {facteur_str}")

        # A : forward schéma vs forward uniforme
        if len(t_fwd) > 0:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.suptitle(
                f"Forward {schema} vs forward uniforme - "
                f"grille {l}x{l} (n={n}), T={T} | norm.={facteur_str}",
                fontsize=10
            )
            if len(t_fwd_unif) > 0:
                ax.hist(t_fwd_unif, bins=bins, density=True, alpha=0.3,
                        color="gray", edgecolor="white",
                        label=f"Forward uniforme (référence, n={len(t_fwd_unif)})")
                ax.axvline(t_fwd_unif.mean(), color="gray", lw=1.2, ls="--",
                           label=f"moy. fwd uniforme = {t_fwd_unif.mean():.0f}")
            ax.hist(t_fwd, bins=bins, density=True, alpha=0.5,
                    color="#D55E00", edgecolor="white",
                    label=f"Forward {schema} (n={len(t_fwd)})")
            ax.axvline(t_fwd.mean(), color="#D55E00", lw=1.5, ls="--",
                       label=f"moy. fwd {schema} = {t_fwd.mean():.0f}")
            ax.legend(fontsize=9, framealpha=0.5)
            ax.set_xlabel("Temps (pas de Moran, t=1 = présent)")
            ax.set_ylabel("Densité")
            plt.tight_layout()
            if sauvegarder:
                nom = f"moran_l{l}_T{T}_fwd_{schema}.png"
                plt.savefig(nom, dpi=150)
                print(f"  Graphique sauvegardé : {nom}")
            if afficher:
                plt.show()
            plt.close(fig)

        # B : backward schéma vs backward uniforme
        if len(t_bwd) > 0:
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.suptitle(
                f"Backward {schema} vs backward uniforme - "
                f"grille {l}x{l} (n={n}), T={T}",
                fontsize=10
            )
            if len(t_bwd_unif) > 0:
                ax.hist(t_bwd_unif, bins=bins, density=True, alpha=0.3,
                        color="gray", edgecolor="white",
                        label=f"Backward uniforme (référence, n={len(t_bwd_unif)})")
                ax.axvline(t_bwd_unif.mean(), color="gray", lw=1.2, ls="--",
                           label=f"moy. bwd uniforme = {t_bwd_unif.mean():.0f}")
            ax.hist(t_bwd, bins=bins, density=True, alpha=0.5,
                    color="#0072B2", edgecolor="white",
                    label=f"Backward {schema} (n={len(t_bwd)})")
            ax.axvline(t_bwd.mean(), color="#0072B2", lw=1.5, ls="--",
                       label=f"moy. bwd {schema} = {t_bwd.mean():.0f}")
            ax.legend(fontsize=9, framealpha=0.5)
            ax.set_xlabel("Temps (pas de Moran, t=1 = présent)")
            ax.set_ylabel("Densité")
            plt.tight_layout()
            if sauvegarder:
                nom = f"moran_l{l}_T{T}_bwd_{schema}.png"
                plt.savefig(nom, dpi=150)
                print(f"  Graphique sauvegardé : {nom}")
            if afficher:
                plt.show()
            plt.close(fig)

        # C+D : jointes forward et backward côte à côte
        if (len(t_fwd) > 0 and len(d_fwd) > 0) or (len(t_bwd) > 0 and len(d_bwd) > 0):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(
                f"Distributions jointes {schema} - grille {l}x{l} (n={n}), T={T}",
                fontsize=10
            )

            # axe temporel commun aux deux panneaux
            t_all = []
            if len(t_fwd) > 0:
                t_all.extend(t_fwd)
            if len(t_bwd) > 0:
                t_all.extend(t_bwd)
            t_max_joint = np.percentile(t_all, Q)
            bins_t = np.linspace(0, t_max_joint, 40)
            bins_d = np.linspace(0, d_max_joint_global, 30)

            ax = axes[0]
            if len(t_fwd) > 0 and len(d_fwd) > 0:
                ax.hist2d(t_fwd, d_fwd, bins=[bins_t, bins_d],
                          cmap="Oranges", density=True)
                ax.axhline(d_fwd.mean(), color="#D55E00", lw=1.2, ls="--",
                           label=f"moy. dist = {d_fwd.mean():.2f}")
                ax.legend(fontsize=8)
            ax.set_xlabel("Temps de divergence t")
            ax.set_ylabel("Distance entre feuilles (gauche x droite)")
            ax.set_title(f"Forward {schema} (n={len(t_fwd)} paires)")

            ax = axes[1]
            if len(t_bwd) > 0 and len(d_bwd) > 0:
                ax.hist2d(t_bwd, d_bwd, bins=[bins_t, bins_d],
                          cmap="Blues", density=True)
                ax.axhline(d_bwd.mean(), color="#0072B2", lw=1.2, ls="--",
                           label=f"moy. d0 = {d_bwd.mean():.2f}")
                ax.legend(fontsize=8)
            ax.set_xlabel("Temps de coalescence t")
            ax.set_ylabel("Distance initiale d0")
            ax.set_title(f"Backward {schema} (n={len(t_bwd)} points)")

            plt.tight_layout()
            if sauvegarder:
                nom = f"moran_l{l}_T{T}_{schema}_joint.png"
                plt.savefig(nom, dpi=150)
                print(f"  Graphique sauvegardé : {nom}")
            if afficher:
                plt.show()
            plt.close(fig)


# =============================================================================
# Lecture des arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Moran spatialisé : forward puis backward (3 schémas) par répétition"
)

parser.add_argument("--l", type=int, default=7,
                    help="Côté de la grille l x l (défaut : 7), population = l*l")

parser.add_argument("--T", type=int, default=None,
                    help="Nombre de pas de Moran. Si absent, calculé automatiquement.")

parser.add_argument("--m", type=float, default=1.0,
                    help="Taux de migration (défaut : 1.0)")

parser.add_argument("--rep", type=int, default=200,
                    help="Nombre de répétitions (défaut : 200)")

parser.add_argument("--mode", type=str, default="compare",
                    choices=["estimer_T", "compare"],
                    help="estimer_T : estime le T_mrca | compare : simule forward + backward")

parser.add_argument("--sigma", type=float, default=1.0,
                    help="Écart-type pour le schéma diagonale (défaut : 1.0)")

parser.add_argument("--rayon", type=float, default=None,
                    help="Rayon pour le schéma cercle (défaut : l/4)")

parser.add_argument("--quantile", type=float, default=99,
                    help="Percentile pour le crop des histogrammes (défaut : 99)")

parser.add_argument("--afficher", action="store_true",
                    help="Affiche les graphiques à l'écran")

parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde les graphiques en .png")

args = parser.parse_args()


# =============================================================================
# Lancement
# =============================================================================

n = args.l * args.l
rayon_effectif = args.rayon if args.rayon is not None else args.l / 4.0

if args.T is not None:
    T = args.T
else:
    T = calculer_T(n, args.m)
    print(f"T calculé automatiquement : T = {T}")

print(f"Grille {args.l}x{args.l} | population n={n} | "
      f"T={T} | m={args.m} | "
      f"sigma={args.sigma} | rayon={rayon_effectif:.1f}")

print(f"\nSimulation (T={T}, rep={args.rep})...")

# précalcul des cases valides par schéma (une seule fois avant la boucle)
cases_par_schema = {}
for schema in ["uniforme", "cercle", "diagonale"]:
    cases_par_schema[schema] = cases_dans_zone(
        args.l, schema, sigma=args.sigma, rayon=args.rayon
    )

temps_fwd_total = {"uniforme": [], "cercle": [], "diagonale": []}
distances_fwd_total = {"uniforme": [], "cercle": [], "diagonale": []}
temps_bwd_total = {"uniforme": [], "cercle": [], "diagonale": []}
distances_bwd_total = {"uniforme": [], "cercle": [], "diagonale": []}
n_nc_bwd = {"uniforme": 0, "cercle": 0, "diagonale": 0}
n_zero_fwd = {"uniforme": 0, "cercle": 0, "diagonale": 0}
n_in_total = {"cercle": 0, "diagonale": 0}
n_total_norm = {"cercle": 0, "diagonale": 0}

for rep in range(args.rep):

    print(f"  répétition {rep + 1}/{args.rep}...", end="\r", flush=True)

    evenements = generer_evenements(args.l, T, args.m)

    for schema in ["uniforme", "cercle", "diagonale"]:
        t_fwd, d_fwd, n_paires, n_in, n_total = forward(
            args.l, T, evenements,
            schema=schema, sigma=args.sigma, rayon=args.rayon
        )

        if n_paires == 0:
            n_zero_fwd[schema] = n_zero_fwd[schema] + 1
            continue

        temps_fwd_total[schema].extend(t_fwd)
        distances_fwd_total[schema].extend(d_fwd)

        if schema in n_in_total:
            n_in_total[schema] = n_in_total[schema] + n_in
            n_total_norm[schema] = n_total_norm[schema] + n_total

        t_bwd, d_bwd, n_nc = backward(
            args.l, T, n_paires, evenements,
            cases_par_schema[schema],
            schema=schema, sigma=args.sigma, rayon=args.rayon
        )
        temps_bwd_total[schema].extend(t_bwd)
        distances_bwd_total[schema].extend(d_bwd)
        n_nc_bwd[schema] = n_nc_bwd[schema] + n_nc

# conversion en arrays numpy
resultats_fwd = {}
resultats_bwd = {}
for schema in ["uniforme", "cercle", "diagonale"]:
    resultats_fwd[schema] = {
        "temps": np.array(temps_fwd_total[schema]),
        "distances": np.array(distances_fwd_total[schema])
    }
    resultats_bwd[schema] = {
        "temps": np.array(temps_bwd_total[schema]),
        "distances": np.array(distances_bwd_total[schema])
    }

facteurs_norm = {}
for schema in ["cercle", "diagonale"]:
    if n_total_norm[schema] > 0:
        facteurs_norm[schema] = n_in_total[schema] / n_total_norm[schema]
    else:
        facteurs_norm[schema] = 0.0

print(f"\nRésultats :")
for schema in ["uniforme", "cercle", "diagonale"]:
    t_fwd = resultats_fwd[schema]["temps"]
    t_bwd = resultats_bwd[schema]["temps"]
    n_total_bwd = len(t_bwd) + n_nc_bwd[schema]
    taux_nc = n_nc_bwd[schema] / n_total_bwd * 100 if n_total_bwd > 0 else 0
    msg_nc = " -- augmenter T !" if taux_nc > 5 else ""
    print(f"  [{schema}] fwd={len(t_fwd)} paires | "
          f"bwd={len(t_bwd)} coal. ({taux_nc:.1f}% non coal.{msg_nc}) | "
          f"rep sans paires : {n_zero_fwd[schema]}")

for schema in ["cercle", "diagonale"]:
    print(f"  Facteur normalisation {schema} = {facteurs_norm[schema]:.3f} "
          f"({n_in_total[schema]} in / {n_total_norm[schema]} total)")

if len(resultats_fwd["uniforme"]["temps"]) == 0:
    print("\nAucun résultat à afficher. Augmenter T.")
    sys.exit(0)

# resultats_fwd et resultats_bwd sont le tableau principal.
# La colonne proba_analytique sera ajoutée ici plus tard.
afficher_resultats(
    resultats_fwd, resultats_bwd, facteurs_norm,
    args.l, T, args.m, args.quantile,
    args.afficher, args.sauvegarder,
    sigma=args.sigma, rayon=args.rayon
)