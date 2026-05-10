# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Processus de Moran spatialisé - comparaison forward / backward
# ----------------------------------------------------------------------
# Auteur : Robin Pioch
# Stage M1 Bioinformatique, Université de Montpellier
# Encadrant : Stéphane Guindon (LIRMM)
# Mai 2026
#
# Ce que fait ce script :
#
#   On simule un processus de Moran spatialisé sur une grille n x n.
#   Règle du jeu : à chaque pas, on choisit au hasard deux cases voisines
#   A et B. L'individu en A meurt. L'individu en B se reproduit deux fois :
#   un enfant reste en B, l'autre va en A. La grille est toujours pleine.
#
#   Le but est de comparer deux approches :
#
#   FORWARD : on simule toute l'histoire de la grille sur T pas.
#     On tire deux individus au présent et on remonte leur arbre
#     généalogique pour trouver quand leurs lignées se rejoignent.
#
#   BACKWARD : on tire deux individus au présent et on remonte
#     dans le passé en tirant des événements Moran un par un.
#     On s'arrête quand les deux lignées coalescent.
#
#   SURVIE : approche de la première génération proposée par Stéphane.
#     Au premier pas, deux lignées naissent (A et B). On simule T pas
#     et on regarde si les deux ont encore des descendants au présent.
#     S(t) = proba que les deux survivent jusqu'au présent.
#     p(t) = S(t-1) - S(t) : densité de coalescence forward.
#
#   Si les approches décrivent le même processus,
#   les distributions jointes (temps, distance) doivent coïncider.
#
# Usage :
#   python moran_spatialise.py --n 10 --mode estimer_T --rep 30
#   python moran_spatialise.py --n 10 --T 131000 --rep 500 --mode compare --afficher
#   python moran_spatialise.py --n 10 --T 131000 --rep 500 --mode survie --afficher
#
# Options :
#   --n : taille de la grille (défaut : 10)
#   --T : nombre de pas de Moran (défaut : 5000)
#   --rep : nombre de répétitions (défaut : 1000)
#   --mode : estimer_T / compare / survie
#   --afficher : affiche les graphiques
#   --sauvegarder : sauvegarde les graphiques en .png
# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

# -----------------------------------------------------------------------------
# Fonctions utilitaires
# -----------------------------------------------------------------------------

def construire_paires_voisins(n):
    """
    Liste toutes les paires de cases adjacentes sur la grille n x n.

    Deux cases sont voisines si elles sont côte à côte horizontalement
    ou verticalement (pas les diagonales). Les bords sont réfléchissants :
    une case en bord n'a pas de voisin de l'autre côté.

    Paramètres :
        n : taille de la grille (n x n)

    Retourne :
        paires : liste de tuples ((x1, y1), (x2, y2))
    """
    paires = []
    for x in range(n):
        for y in range(n):
            if x + 1 < n: # bords réfléchissants
                paires.append(((x, y), (x + 1, y)))
            if y + 1 < n:
                paires.append(((x, y), (x, y + 1)))
    return paires


def distance(x1, y1, x2, y2):
    """
    Distance euclidienne entre deux cases de la grille.

    Paramètres :
        x1, y1 : coordonnées de la première case
        x2, y2 : coordonnées de la deuxième case

    Retourne :
        distance : float
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# -----------------------------------------------------------------------------
# Estimation de T_mrca
# -----------------------------------------------------------------------------

def estimer_T_mrca(n, n_essais):
    """
    Estime le nombre de pas nécessaires pour que toute la grille ait
    un seul ancêtre commun (le MRCA).

    On fait tourner le processus en backward : chaque case démarre avec
    son propre identifiant de lignée. À chaque événement (A <- B), toutes
    les cases qui ont la lignée de A prennent la lignée de B. On compte
    jusqu'à ce qu'il ne reste plus qu'une seule lignée.

    Paramètres :
        n : taille de la grille
        n_essais : nombre de répétitions pour estimer la moyenne

    Retourne :
        t_moyen, t_max : (float, int) en pas de Moran
    """
    paires = construire_paires_voisins(n)
    resultats = []

    for _ in range(n_essais):

        # Chaque case part avec un identifiant unique
        lignee = np.arange(n * n).reshape(n, n)
        t = 0

        while np.unique(lignee).size > 1:
            A, B = paires[np.random.randint(len(paires))]
            if np.random.rand() < 0.5: # Façon de rendre aléatoire qui est A et qui est B
                A, B = B, A
            # Toutes les cases avec la lignée de A rejoignent la lignée de B
            lignee[lignee == lignee[A]] = lignee[B]
            t += 1

        resultats.append(t)

    return np.mean(resultats), np.max(resultats)


# -----------------------------------------------------------------------------
# Simulation FORWARD
# -----------------------------------------------------------------------------

def simuler_forward(n, T, n_repetitions):
    """
    Simule le processus de Moran en forward et remonte les généalogies.

    Pour chaque répétition, on génère T événements Moran indépendants,
    on tire deux individus au présent, puis on remonte leur arbre
    généalogique en lisant les événements à l'envers jusqu'à trouver
    leur ancêtre commun.

    On enregistre l'âge de l'ancêtre commun (T - k) et la distance
    entre les deux individus au présent. On mesure la distance au présent
    et pas à la coalescence parce qu'à la coalescence les deux lignées
    sont forcément voisines (distance = 1 toujours).

    On génère un historique différent pour chaque paire.
    Si on réutilisait le même, tous les temps tomberaient sur les mêmes
    pas de temps et ça créerait un artefact dans la distribution.

    Paramètres :
        n : taille de la grille
        T : nombre de pas de Moran simulés
        n_repetitions : nombre de paires tirées

    Retourne :
        temps_liste : array des âges des ancêtres communs trouvés
        distances_liste : array des distances au présent correspondantes
        n_non_coal : nombre de paires sans ancêtre trouvé dans [0, T]
    """
    paires = construire_paires_voisins(n)
    temps_liste = []
    distances_liste = []
    n_non_coal = 0

    for _ in range(n_repetitions):

        # Stockage de l'historique des événements : liste de tuples (A, B) pour chaque pas
        evenements = []
        for _ in range(T):
            A, B = paires[np.random.randint(len(paires))]
            if np.random.rand() < 0.5:
                A, B = B, A
            evenements.append((A, B))

        # Deux cases distinctes tirées au hasard au présent
        while True:
            xi, yi = np.random.randint(0, n), np.random.randint(0, n)
            xj, yj = np.random.randint(0, n), np.random.randint(0, n)
            if (xi, yi) != (xj, yj):
                break

        d0 = distance(xi, yi, xj, yj) # distance au présent
        
        # Initialisation des ancêtres : au départ chaque individu est son propre ancêtre
        anc_i = (xi, yi)
        anc_j = (xj, yj)
        coalesce = False

        # On lit les événements à l'envers
        for k in range(T - 1, -1, -1):
            A, B = evenements[k]
            if anc_i == A:
                anc_i = B
            if anc_j == A:
                anc_j = B
            if anc_i == anc_j:
                temps_liste.append(T - k)
                distances_liste.append(d0)
                coalesce = True
                break

        # Si on n'a pas trouvé d'ancêtre commun dans les T pas on compte comme non coalescé
        if not coalesce:
            n_non_coal += 1

    return np.array(temps_liste), np.array(distances_liste), n_non_coal


# -----------------------------------------------------------------------------
# Simulation BACKWARD
# -----------------------------------------------------------------------------

def simuler_backward(n, T, n_repetitions):
    """
    Simule la coalescence de deux lignées en backward sous Moran.

    On tire deux individus au présent et on remonte dans le passé en
    tirant des événements Moran un par un. À chaque événement (A <- B) :
        - Si une lignée est en A et l'autre en B : coalescence.
        - Si une lignée est en A (et l'autre ailleurs) : elle recule vers B.
        - Si aucune des deux n'est touchée : on passe au suivant.

    On enregistre le temps de coalescence et la distance initiale
    (entre les deux individus au présent, avant toute remontée).

    Paramètres :
        n : taille de la grille
        T : nombre de pas maximum avant d'abandonner
        n_repetitions : nombre de paires tirées

    Retourne :
        temps_liste : array des temps de coalescence
        distances_liste : array des distances initiales correspondantes
        n_non_coal : nombre de paires non coalescées dans [0, T]
    """
    paires = construire_paires_voisins(n)
    temps_liste = []
    distances_liste = []
    n_non_coal = 0

    for _ in range(n_repetitions):

        # Deux cases distinctes tirées au hasard
        while True:
            x1, y1 = np.random.randint(0, n), np.random.randint(0, n)
            x2, y2 = np.random.randint(0, n), np.random.randint(0, n)
            if (x1, y1) != (x2, y2):
                break

        d0 = distance(x1, y1, x2, y2)
        coalesce = False

        for t in range(1, T + 1):
            A, B = paires[np.random.randint(len(paires))]
            if np.random.rand() < 0.5:
                A, B = B, A

            # Les deux lignées sont sur A et B : coalescence
            if (A == (x1, y1) and B == (x2, y2)) or \
               (A == (x2, y2) and B == (x1, y1)):
                temps_liste.append(t)
                distances_liste.append(d0)
                coalesce = True
                break

            # Une seule lignée est touchée : elle recule vers B
            if A == (x1, y1):
                x1, y1 = B
            elif A == (x2, y2):
                x2, y2 = B

        if not coalesce:
            n_non_coal += 1

    return np.array(temps_liste), np.array(distances_liste), n_non_coal


# -----------------------------------------------------------------------------
# Simulation SURVIE
# -----------------------------------------------------------------------------

def simuler_survie(n, T, n_simulations):
    """
    Estime S(t) et p(t) à partir de l'approche de la première génération.

    Au premier pas de chaque simulation, un nœud B donne deux enfants :
    un sur B, un sur A. Ce sont les deux lignées qu'on suit.
    On veut savoir si ces deux lignées ont encore des descendants au
    présent après t générations.

    S(t) = proportion des simulations où les deux lignées ont toutes
    les deux au moins un descendant au présent après t pas.

    p(t) = S(t-1) - S(t) : c'est la densité de probabilité que la
    première perte d'une lignée se produise exactement au pas t.
    p(t) devrait être en O(1/n^2).

    Comment on sait si une lignée a encore des descendants ?
    On remonte depuis le présent : si au moins un individu au présent
    a pour ancêtre la case A (ou B) au pas 0, la lignée A (ou B) est
    encore vivante.

    Paramètres :
        n : taille de la grille
        T : nombre de pas simulés
        n_simulations : nombre de simulations indépendantes

    Retourne :
        t_valeurs : array des valeurs de t (1 à T)
        S : array de S(t) pour chaque t
        p : array de p(t) = S(t-1) - S(t)
    """
    paires = construire_paires_voisins(n)

    # Pour chaque t, on compte combien de simulations ont les deux lignées encore vivantes
    survie_comptes = np.zeros(T + 1)

    for _ in range(n_simulations):

        # Génération des T événements
        evenements = []
        for _ in range(T):
            A, B = paires[np.random.randint(len(paires))]
            if np.random.rand() < 0.5:
                A, B = B, A
            evenements.append((A, B))

        # Les deux lignées naissent au pas 0 : une sur A0, une sur B0
        A0, B0 = evenements[0]

        # ancetre_x[x, y] et ancetre_y[x, y] = ancêtre de la case (x, y)
        # au pas courant qu'on remonte. Au départ (au présent) chaque
        # case est son propre ancêtre.
        ancetre_x = np.zeros((n, n), dtype=np.int16)
        for x in range(n):
            for y in range(n):
                ancetre_x[x, y] = x
                
        ancetre_y = np.zeros((n, n), dtype=np.int16)
        for x in range(n):
            for y in range(n):
                ancetre_y[x, y] = y

        # On remonte les événements de T-1 vers 0
        for k in range(T - 1, -1, -1):
            Ak, Bk = evenements[k]
            # La case Ak hérite de Bk : son ancêtre devient l'ancêtre de Bk
            ancetre_x[Ak] = ancetre_x[Bk]
            ancetre_y[Ak] = ancetre_y[Bk]

            t = T - k  # nombre de générations dans le passé

            # Est-ce qu'au moins un individu au présent descend de A0 ?
            lignee_A = np.any((ancetre_x == A0[0]) & (ancetre_y == A0[1]))
            # Est-ce qu'au moins un individu au présent descend de B0 ?
            lignee_B = np.any((ancetre_x == B0[0]) & (ancetre_y == B0[1]))

            if lignee_A and lignee_B:
                survie_comptes[t] += 1

    # S(t) = proportion des simulations où les deux survivent jusqu'à t
    S = survie_comptes[1:] / n_simulations

    # p(t) = S(t-1) - S(t)
    # S(0) = 1 par définition (les deux lignées viennent juste de naître)
    S_avec_zero = np.concatenate([[1.0], S])
    p = S_avec_zero[:-1] - S_avec_zero[1:]

    t_valeurs = np.arange(1, T + 1)

    return t_valeurs, S, p


def afficher_survie(t_valeurs, S, p, n, T, afficher, sauvegarder):
    """
    Affiche S(t) et p(t).

    Paramètres :
        t_valeurs : array des valeurs de t
        S : array de S(t)
        p : array de p(t)
        n, T : paramètres de la simulation
        afficher : bool
        sauvegarder : bool
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Survie des deux lignées sous Moran spatialisé\n"
        f"(grille {n}x{n}, T={T})",
        fontsize=13
    )

    axes[0].plot(t_valeurs, S, color="steelblue", lw=1.5)
    axes[0].set_xlabel("t (pas de Moran)")
    axes[0].set_ylabel("S(t)")
    axes[0].set_title("Probabilité que les deux lignées survivent jusqu'à t")

    axes[1].plot(t_valeurs, p, color="#D55E00", lw=1.5)
    axes[1].set_xlabel("t (pas de Moran)")
    axes[1].set_ylabel("p(t) = S(t-1) - S(t)")
    axes[1].set_title("p(t) : densité de coalescence forward")

    # Temps moyen estimé
    somme_p = np.sum(p)
    t_moyen = np.sum(t_valeurs * p) / somme_p if somme_p > 0 else 0
    print(f"  Somme de p(t) sur [1, T] : {somme_p:.4f}")
    print(f"  Temps moyen estimé depuis p(t) : {t_moyen:.1f} pas")
    print(f"  Référence : n^2 = {n**2},  n^4 = {n**4}")

    plt.tight_layout()
    if sauvegarder:
        plt.savefig("moran_survie.png", dpi=150)
        print("  Graphique sauvegardé : moran_survie.png")
    if afficher:
        plt.show()
    plt.close()


# -----------------------------------------------------------------------------
# Affichage des résultats forward / backward
# -----------------------------------------------------------------------------

def afficher_resultats(temps_fwd, dist_fwd, temps_bwd, dist_bwd, n, T,
                        afficher, sauvegarder):
    """
    Produit deux graphiques : distributions marginales et densité jointe.

    Le premier graphique compare les distributions du temps et de la
    distance entre forward et backward, avec les moyennes, écarts-types,
    et la courbe analytique backward proposée par Stéphane.

    Le deuxième graphique compare les densités jointes (temps, distance)
    sous forme d'histogrammes 2D. Si les deux approches sont équivalentes,
    les deux histogrammes doivent se ressembler.

    Paramètres :
        temps_fwd, dist_fwd : résultats du forward
        temps_bwd, dist_bwd : résultats du backward
        n : taille de la grille
        T : nombre de pas utilisés
        afficher : bool, affiche les graphiques si True
        sauvegarder : bool, sauvegarde en .png si True
    """
    t_max = max(temps_fwd.max(), temps_bwd.max())
    d_max = max(dist_fwd.max(), dist_bwd.max())
    bins_t = np.linspace(0, t_max, 40)
    bins_d = np.linspace(0, d_max + 0.5, 40)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Moran spatialisé : comparaison forward / backward\n"
        f"(grille {n}x{n}, T={T})",
        fontsize=13
    )

    # Distribution du temps
    # En forward : âge de l'ancêtre commun
    # En backward : temps de coalescence
    axes[0].hist(temps_fwd, bins=bins_t, density=True, alpha=0.6,
                 color="steelblue", edgecolor="white",
                 label=f"Forward (n={len(temps_fwd)})")
    axes[0].hist(temps_bwd, bins=bins_t, density=True, alpha=0.6,
                 color="#D55E00", edgecolor="white",
                 label=f"Backward (n={len(temps_bwd)})")
    axes[0].axvline(temps_fwd.mean(), color="steelblue", lw=1.5, ls="--",
                    label=f"moy. forward = {temps_fwd.mean():.0f}")
    axes[0].axvline(temps_fwd.mean() + temps_fwd.std(), color="steelblue", lw=0.8, ls=":",
                    label=f"± σ forward = {temps_fwd.std():.0f}")
    axes[0].axvline(temps_fwd.mean() - temps_fwd.std(), color="steelblue", lw=0.8, ls=":")
    axes[0].axvline(temps_bwd.mean(), color="#D55E00", lw=1.5, ls="--",
                    label=f"moy. backward = {temps_bwd.mean():.0f}")
    axes[0].axvline(temps_bwd.mean() + temps_bwd.std(), color="#D55E00", lw=0.8, ls=":",
                    label=f"± σ backward = {temps_bwd.std():.0f}")
    axes[0].axvline(temps_bwd.mean() - temps_bwd.std(), color="#D55E00", lw=0.8, ls=":")

    # Courbe analytique backward : (1 - 1/n^4)^(t-1) * 1/n^4
    # temps moyen = n^4
    # Tracée sur un axe Y séparé car les échelles sont incompatibles
    t_analytique = np.linspace(1, t_max, 10000)
    p_t_backward = (1 - 1 / n**4) ** (t_analytique - 1) * (1 / n**4)
    dt = t_analytique[1] - t_analytique[0]
    p_t_backward = p_t_backward / (p_t_backward.sum() * dt)

    ax_twin = axes[0].twinx()
    ax_twin.plot(t_analytique, p_t_backward, color="darkgreen", lw=1.5, ls="-.",
                 label=f"$(1 - 1/n^4)^{{t-1}} \\cdot 1/n^4$ (backward)")
    ax_twin.set_ylabel("Densité analytique", fontsize=8)
    ax_twin.set_ylim(0, p_t_backward.max() * 1.1)

    # Fusion des légendes des deux axes en une seule
    lignes_gauche, labels_gauche = axes[0].get_legend_handles_labels()
    lignes_droite, labels_droite = ax_twin.get_legend_handles_labels()
    axes[0].legend(lignes_gauche + lignes_droite,
                   labels_gauche + labels_droite,
                   fontsize=8, framealpha=0.5)

    print(f"  Temps moyen analytique backward (1-1/n^4)^t : {n**4} pas")
    print(f"  Simulations : fwd={temps_fwd.mean():.0f}, bwd={temps_bwd.mean():.0f}")

    axes[0].set_xlabel("Forward : âge de l'ancêtre commun (pas)\n"
                       "Backward : temps de coalescence (pas)")
    axes[0].set_ylabel("Densité")
    axes[0].set_title("Distribution du temps")

    # Distribution de la distance
    axes[1].hist(dist_fwd, bins=bins_d, density=True, alpha=0.6,
                 color="steelblue", edgecolor="white",
                 label=f"Forward (n={len(dist_fwd)})")
    axes[1].hist(dist_bwd, bins=bins_d, density=True, alpha=0.6,
                 color="#D55E00", edgecolor="white",
                 label=f"Backward (n={len(dist_bwd)})")
    axes[1].axvline(dist_fwd.mean(), color="steelblue", lw=1.5, ls="--",
                    label=f"moy. forward = {dist_fwd.mean():.1f}")
    axes[1].axvline(dist_fwd.mean() + dist_fwd.std(), color="steelblue", lw=0.8, ls=":",
                    label=f"± σ forward = {dist_fwd.std():.1f}")
    axes[1].axvline(dist_fwd.mean() - dist_fwd.std(), color="steelblue", lw=0.8, ls=":")
    axes[1].axvline(dist_bwd.mean(), color="#D55E00", lw=1.5, ls="--",
                    label=f"moy. backward = {dist_bwd.mean():.1f}")
    axes[1].axvline(dist_bwd.mean() + dist_bwd.std(), color="#D55E00", lw=0.8, ls=":",
                    label=f"± σ backward = {dist_bwd.std():.1f}")
    axes[1].axvline(dist_bwd.mean() - dist_bwd.std(), color="#D55E00", lw=0.8, ls=":")
    axes[1].set_xlabel("Distance euclidienne entre les deux individus au présent")
    axes[1].set_ylabel("Densité")
    axes[1].set_title("Distribution de la distance")
    axes[1].legend(fontsize=8, framealpha=0.5)

    plt.tight_layout()
    if sauvegarder:
        plt.savefig("moran_marginal.png", dpi=150)
        print("  Graphique sauvegardé : moran_marginal.png")
    if afficher:
        plt.show()
    plt.close()

    # Densité jointe
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5))
    fig2.suptitle(
        f"Moran spatialisé : densité jointe (temps, distance)\n"
        f"(grille {n}x{n}, T={T})",
        fontsize=13
    )

    h1 = axes2[0].hist2d(temps_fwd, dist_fwd, bins=[bins_t, bins_d],
                          density=True, cmap="Blues")
    fig2.colorbar(h1[3], ax=axes2[0], label="Densité")
    axes2[0].set_xlabel("Âge de l'ancêtre commun (pas)")
    axes2[0].set_ylabel("Distance entre les deux individus au présent")
    axes2[0].set_title(f"Forward (n={len(temps_fwd)})")

    h2 = axes2[1].hist2d(temps_bwd, dist_bwd, bins=[bins_t, bins_d],
                          density=True, cmap="Greens")
    fig2.colorbar(h2[3], ax=axes2[1], label="Densité")
    axes2[1].set_xlabel("Temps de coalescence (pas)")
    axes2[1].set_ylabel("Distance entre les deux individus au présent")
    axes2[1].set_title(f"Backward (n={len(temps_bwd)})")

    plt.tight_layout()
    if sauvegarder:
        plt.savefig("moran_joint.png", dpi=150)
        print("  Graphique sauvegardé : moran_joint.png")
    if afficher:
        plt.show()
    plt.close()


# -----------------------------------------------------------------------------
# Lecture des arguments en ligne de commande
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Processus de Moran spatialisé : comparaison forward / backward"
)

parser.add_argument("--n", type=int, default=10,
                    help="Taille de la grille n x n (défaut : 10)")

parser.add_argument("--T", type=int, default=5000,
                    help="Nombre de pas de Moran simulés (défaut : 5000). "
                         "Doit être bien plus grand que T_mrca. "
                         "Utiliser --mode estimer_T pour calibrer.")

parser.add_argument("--rep", type=int, default=1000,
                    help="Nombre de répétitions (défaut : 1000)")

parser.add_argument("--mode", type=str, default="compare",
                    choices=["estimer_T", "compare", "survie"],
                    help=(
                        "estimer_T : estime le T_mrca pour la grille choisie. "
                        "compare   : simule forward et backward et compare (défaut). "
                        "survie    : estime S(t) et p(t) = S(t-1) - S(t) "
                        "            depuis la première génération."
                    ))

parser.add_argument("--afficher", action="store_true",
                    help="Affiche les graphiques à l'écran")

parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde les graphiques en .png")

args = parser.parse_args()


# -----------------------------------------------------------------------------
# Lancement
# -----------------------------------------------------------------------------

paires = construire_paires_voisins(args.n)
print(f"Grille {args.n}x{args.n}  |  {args.n**2} noeuds  |  "
      f"{len(paires)} paires voisines  |  T={args.T}  |  mode={args.mode}")


if args.mode == "estimer_T":
    print(f"\nEstimation de T_mrca sur {args.rep} essais...")
    t_moy, t_max = estimer_T_mrca(args.n, n_essais=args.rep)
    print(f"  T_mrca moyen : {t_moy:.0f} pas")
    print(f"  T_mrca max   : {t_max} pas")
    print(f"  Pour être sûr, utiliser --T {int(t_max * 200)}")
    sys.exit(0)


if args.mode == "compare":
    print(f"\nSimulation forward (T={args.T}, rep={args.rep})...")
    t_fwd, d_fwd, n_nc_fwd = simuler_forward(args.n, args.T, args.rep)
    taux_nc = n_nc_fwd / args.rep * 100
    print(f"  Ancêtre commun trouvé : {len(t_fwd)} / {args.rep}  "
          f"({n_nc_fwd} non trouvés, soit {taux_nc:.1f}%"
          f"{' -- augmenter T !' if taux_nc > 5 else ''})")
    if len(t_fwd) > 0:
        print(f"  Âge moyen de l'ancêtre commun : {t_fwd.mean():.1f} +/- {t_fwd.std():.1f} pas")
        print(f"  Distance moy. au présent      : {d_fwd.mean():.2f} +/- {d_fwd.std():.2f}")

    print(f"\nSimulation backward (T={args.T}, rep={args.rep})...")
    t_bwd, d_bwd, n_nc_bwd = simuler_backward(args.n, args.T, args.rep)
    taux_nc = n_nc_bwd / args.rep * 100
    print(f"  Coalescences trouvées : {len(t_bwd)} / {args.rep}  "
          f"({n_nc_bwd} non coalescées, soit {taux_nc:.1f}%"
          f"{' -- augmenter T !' if taux_nc > 5 else ''})")
    if len(t_bwd) > 0:
        print(f"  Temps de coalescence moyen : {t_bwd.mean():.1f} +/- {t_bwd.std():.1f} pas")
        print(f"  Distance moy. au présent   : {d_bwd.mean():.2f} +/- {d_bwd.std():.2f}")

    if len(t_fwd) == 0 or len(t_bwd) == 0:
        print("\nPas assez de coalescences pour comparer. Augmenter T.")
    else:
        afficher_resultats(t_fwd, d_fwd, t_bwd, d_bwd,
                           args.n, args.T,
                           args.afficher, args.sauvegarder)


if args.mode == "survie":
    print(f"\nSimulation survie (T={args.T}, rep={args.rep})...")
    t_valeurs, S, p = simuler_survie(args.n, args.T, args.rep)
    afficher_survie(t_valeurs, S, p, args.n, args.T,
                    args.afficher, args.sauvegarder)