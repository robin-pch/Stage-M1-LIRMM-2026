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
#   Si les deux approches décrivent le même processus,
#   les distributions jointes (temps, distance) doivent coïncider.
#
# Usage :
#   python moran_spatialise.py --n 10 --mode estimer_T --rep 30
#   python moran_spatialise.py --n 10 --T 131000 --rep 500 --mode compare --afficher
#
# Options :
#   --n : taille de la grille (défaut : 10)
#   --T : nombre de pas de Moran (défaut : 5000)
#   --rep : nombre de répétitions (défaut : 1000)
#   --mode : estimer_T / compare
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
# Affichage des résultats
# -----------------------------------------------------------------------------

def afficher_resultats(temps_fwd, dist_fwd, temps_bwd, dist_bwd, n, T,
                        afficher, sauvegarder):
    """
    Produit deux graphiques : distributions marginales et densité jointe.

    Le premier graphique compare les distributions du temps et de la
    distance entre forward et backward, avec les moyennes, écarts-types,
    et la courbe analytique proposée par Stéphane.

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

    # Premier graphique : marginales
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        f"Moran spatialisé : comparaison forward / backward\n"
        f"(grille {n}x{n}, T={T})",
        fontsize=13
    )

    # Distribution du temps
    # En forward : âge de l'ancêtre commun (combien de générations depuis la divergence)
    # En backward : temps de coalescence (combien de pas on remonte)
    # Les deux mesurent la même quantité, juste le sens de lecture est inversé.
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

    # Courbe analytique : p(t) = (1 - 1/n²)^(2t)
    # à chaque pas, la proba qu'une lignée soit éliminée est ~1/n².
    # Pour deux lignées, la proba qu'aucune ne soit touchée est (1 - 1/n²)².
    # Sur t pas indépendants : (1 - 1/n²)^(2t).
    # On la trace sur un axe Y séparé à droite parce qu'elle décroît
    # beaucoup plus vite que les histogrammes et les échelles sont incompatibles.
    t_analytique = np.linspace(0, t_max, 10000)
    p_t = (1 - 1 / n**2) ** (2 * t_analytique)

    ax_twin = axes[0].twinx()
    ax_twin.plot(t_analytique, p_t, color="black", lw=1.5, ls="-",
                 label=f"$(1 - 1/n^2)^{{2t}}$ analytique")
    ax_twin.set_ylabel("$(1 - 1/n^2)^{2t}$", fontsize=8)
    ax_twin.set_ylim(0, 1.1)

    # Fusion des légendes des deux axes en une seule
    lignes_gauche, labels_gauche = axes[0].get_legend_handles_labels()
    lignes_droite, labels_droite = ax_twin.get_legend_handles_labels()
    axes[0].legend(lignes_gauche + lignes_droite,
                   labels_gauche + labels_droite,
                   fontsize=8, framealpha=0.5)

    # On affiche l'écart entre la prédiction analytique et les simulations
    t_moyen_analytique = 1 / (2 * (-np.log(1 - 1 / n**2)))
    print(f"  Temps moyen analytique (1-1/n²)^2t : {t_moyen_analytique:.1f} pas"
          f"  (vs simulations : fwd={temps_fwd.mean():.0f}, bwd={temps_bwd.mean():.0f})")

    axes[0].set_xlabel("Forward : âge de l'ancêtre commun (pas)\n"
                       "Backward : temps de coalescence (pas)")
    axes[0].set_ylabel("Densité")
    axes[0].set_title("Distribution du temps")

    # Distribution de la distance
    # C'est la même chose dans les deux cas : distance entre les deux individus
    # au présent, avant toute remontée.
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

    # Deuxième graphique : densité jointe (temps, distance) en 2D
    # Si les deux approches sont équivalentes, les deux histogrammes doivent se ressembler.
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
                    help="Nombre de paires tirées (défaut : 1000)")

parser.add_argument("--mode", type=str, default="compare",
                    choices=["estimer_T", "compare"],
                    help=(
                        "estimer_T : estime le T_mrca pour la grille choisie, à faire en premier. "
                        "compare   : simule forward et backward et compare (défaut)."
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
    # T_mrca c'est le temps pour que toute la grille ait un ancêtre commun.
    # Mais deux individus au hasard peuvent coalescer bien plus tard,
    # surtout s'ils sont loin l'un de l'autre. D'où le x200.
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