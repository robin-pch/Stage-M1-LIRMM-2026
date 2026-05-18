# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Processus de Moran spatialisé - comparaison forward / backward (v2)
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
#   Le tableau d'événements est d'abord généré, puis exploité par deux
#   approches distinctes :
#
#   BACKWARD : on tire deux individus au présent et on remonte leur
#     généalogie dans le tableau. On cherche quand leurs lignées
#     coalescent (ancêtre commun).
#
#   FORWARD : on construit un arbre de descendants en avançant dans le temps.
#     À chaque événement (A <- B), l'occupant de B devient un noeud interne
#     avec deux fils : un en A, un en B. À la fin (temps T), on identifie
#     les noeuds dont les deux fils ont chacun au moins un descendant vivant.
#     Ces noeuds sont valides et on note leur temps dans p(t).
#
#   Le but est de vérifier que les deux approches donnent la même
#   distribution de temps (et plus tard, la même distribution jointe
#   avec la distance).
#
# Usage :
#   python moran_spatialise_v2.py --n 10 --T 5000 --rep 1000 --mode compare --afficher
#   python moran_spatialise_v2.py --n 10 --mode estimer_T --rep 30
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
            if x + 1 < n:
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
# Génération du tableau d'événements
# -----------------------------------------------------------------------------

def generer_evenements(n, T, paires):
    """
    Génère une séquence de T événements Moran indépendants sur la grille n x n.

    À chaque pas, on tire une paire voisine au hasard puis on décide
    aléatoirement quel nœud est A (qui meurt) et quel nœud est B
    (qui se reproduit). L'événement est noté (A, B) : A <- B.

    Ce tableau sert de base commune au backward et au forward.

    Paramètres :
        n : taille de la grille
        T : nombre de pas à générer
        paires : liste des paires voisines (produite par construire_paires_voisins)

    Retourne :
        evenements : liste de T tuples ((xA, yA), (xB, yB))
    """
    evenements = []
    for _ in range(T):
        A, B = paires[np.random.randint(len(paires))]
        if np.random.rand() < 0.5:
            A, B = B, A
        evenements.append((A, B))
    return evenements


# -----------------------------------------------------------------------------
# Estimation de T_mrca
# -----------------------------------------------------------------------------

def estimer_T_mrca(n, n_essais):
    """
    Estime le nombre de pas nécessaires pour que toute la grille ait
    un seul ancêtre commun (le MRCA).

    On fait tourner le processus en backward : chaque case démarre avec
    son propre identifiant de lignée. À chaque événement (A <- B), toutes
    les cases qui avaient la lignée de A prennent la lignée de B. On compte
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
            if np.random.rand() < 0.5:
                A, B = B, A
            lignee[lignee == lignee[A]] = lignee[B]
            t += 1

        resultats.append(t)

    return np.mean(resultats), np.max(resultats)


# -----------------------------------------------------------------------------
# Simulation BACKWARD
# -----------------------------------------------------------------------------

def simuler_backward(n, T, n_repetitions, paires):
    """
    Simule la coalescence de deux individus en remontant un tableau d'événements.

    Pour chaque répétition, on génère un tableau de T événements indépendant,
    on tire deux cases distinctes au présent, puis on lit le tableau à l'envers.
    À chaque événement (A <- B) :
        - Si une lignée est en A : elle recule vers B.
        - Si les deux lignées se retrouvent sur la même case : coalescence.

    On génère un tableau différent par répétition pour éviter que tous
    les temps tombent sur les mêmes pas (artefact de distribution).

    Paramètres :
        n : taille de la grille
        T : nombre de pas par simulation
        n_repetitions : nombre de paires tirées
        paires : liste des paires voisines

    Retourne :
        temps_liste : array des temps de coalescence trouvés
        distances_liste : array des distances initiales correspondantes
        n_non_coal : nombre de paires non coalescées dans [0, T]
    """
    temps_liste = []
    distances_liste = []
    n_non_coal = 0

    for _ in range(n_repetitions):
        evenements = generer_evenements(n, T, paires)

        # Deux cases distinctes au présent
        while True:
            x1, y1 = np.random.randint(0, n), np.random.randint(0, n)
            x2, y2 = np.random.randint(0, n), np.random.randint(0, n)
            if (x1, y1) != (x2, y2):
                break

        d0 = distance(x1, y1, x2, y2)
        coalesce = False

        # Lecture à l'envers
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
            n_non_coal += 1

    return np.array(temps_liste), np.array(distances_liste), n_non_coal


# -----------------------------------------------------------------------------
# Simulation FORWARD 
# -----------------------------------------------------------------------------

def simuler_forward(n, T, n_repetitions, paires):
    """
    Estime p(t) en forward en construisant un arbre de descendants.

    Pour chaque répétition :
      1. On génère un tableau de T événements.
      2. On initialise la grille : chaque case est un noeud initial.
         grille[x][y] contient l'indice du noeud courant en (x, y).
      3. On avance dans le temps. À chaque événement (A <- B) au pas t :
           - L'occupant courant de B (le père) se reproduit.
           - On crée deux nouveaux noeuds fils : un en A, un en B.
           - On note le père de chaque fils.
           - On met à jour grille[A] et grille[B].
      4. À la fin, on calcule en un seul passage le nombre de descendants
         vivants de chaque noeud, en remontant des feuilles vers les noeuds
         anciens. Les noeuds sont créés dans l'ordre chronologique, donc
         parcourir les indices en sens inverse revient à remonter l'arbre.
      5. Pour chaque noeud bifurcation, si ses deux fils ont chacun au moins
         un descendant vivant au temps T : noeud valide, on note son temps.

    Un noeud peut avoir plus de deux descendants vivants au total si ses
    lignées ont bifurqué plusieurs fois. Ce qui compte c'est que ses deux
    fils directs aient chacun au moins un descendant vivant.

    Paramètres :
        n : taille de la grille
        T : nombre de pas par simulation
        n_repetitions : nombre de répétitions (tableaux indépendants)
        paires : liste des paires voisines

    Retourne :
        temps_liste : array de tous les temps valides collectés
        n_zero : nombre de répétitions sans aucun noeud valide
    """
    temps_liste = []
    n_zero = 0

    for _ in range(n_repetitions):
        evenements = generer_evenements(n, T, paires)

        # --- Initialisation de l'arbre ---
        # n^2 noeuds initiaux + 2 nouveaux noeuds par evenement au maximum.
        # Ces noeuds initiaux représentent les occupants de chaque case au
        # temps 0, tous issus d'une même racine théorique commune qu'on ne
        # modélise pas explicitement.
        nb_noeuds_max = n * n + 2 * T

        # fils_gauche[i] et fils_droit[i] = indices des deux fils directs
        # -1 si le noeud n'a pas de fils (feuille)
        fils_gauche = [-1] * nb_noeuds_max
        fils_droit  = [-1] * nb_noeuds_max

        # temps_noeud[i] = pas de Moran auquel le noeud i a été créé
        temps_noeud = [0] * nb_noeuds_max

        nb_noeuds = 0

        # grille[x][y] = indice du noeud courant en (x, y)
        grille = [[0] * n for _ in range(n)]

        # Chaque case démarre avec son propre noeud initial (temps 0)
        for x in range(n):
            for y in range(n):
                grille[x][y] = nb_noeuds
                nb_noeuds = nb_noeuds + 1

        # Avancer dans le temps 
        for t in range(T):
            A, B = evenements[t]
            xA, yA = A
            xB, yB = B

            # L'occupant courant de B se reproduit
            p = grille[xB][yB]

            # Nouveau noeud fils en A
            fa = nb_noeuds
            temps_noeud[fa] = t + 1
            nb_noeuds = nb_noeuds + 1

            # Nouveau noeud fils en B
            fb = nb_noeuds
            temps_noeud[fb] = t + 1
            nb_noeuds = nb_noeuds + 1

            # Le pere gagne ses deux fils
            fils_gauche[p] = fa
            fils_droit[p]  = fb

            # Mise a jour de la grille
            grille[xA][yA] = fa
            grille[xB][yB] = fb

        # --- Calculer les descendants vivants de chaque noeud ---
        # On commence par marquer les nœuds vivants au temps T :
        # ce sont uniquement les nœuds encore présents dans la grille.
        # Les autres feuilles sont des nœuds morts (touchés par un A).
        nb_desc = [0] * nb_noeuds

        for x in range(n):
            for y in range(n):
                nb_desc[grille[x][y]] = 1

        # On remonte des feuilles vers les noeuds anciens en un seul passage.
        for i in range(nb_noeuds - 1, -1, -1):
            if fils_gauche[i] != -1:
                nb_desc[i] = nb_desc[fils_gauche[i]] + nb_desc[fils_droit[i]]

        # --- Identifier les noeuds valides ---
        # Un noeud bifurcation a deux fils (fils_gauche != -1).
        # Il est valide si chaque fils a au moins un descendant vivant.
        n_valides = 0

        for i in range(nb_noeuds):
            if fils_gauche[i] != -1:
                if nb_desc[fils_gauche[i]] >= 1 and nb_desc[fils_droit[i]] >= 1:
                    temps_liste.append(temps_noeud[i])
                    n_valides = n_valides + 1

        if n_valides == 0:
            n_zero = n_zero + 1

    return np.array(temps_liste), n_zero


# -----------------------------------------------------------------------------
# Affichage
# -----------------------------------------------------------------------------

def afficher_resultats(temps_bwd, temps_fwd, n, T, afficher, sauvegarder):
    """
    Compare les distributions des temps backward (coalescence) et
    forward (divergence).

    Paramètres :
        temps_bwd : array des temps de coalescence (backward)
        temps_fwd : array des temps de divergence (forward)
        n : taille de la grille
        T : nombre de pas utilisés
        afficher : bool
        sauvegarder : bool
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"Moran spatialisé : backward (coalescence) vs forward (divergence)\n"
        f"(grille {n}x{n}, T={T})",
        fontsize=12
    )

    t_max = max(temps_bwd.max() if len(temps_bwd) > 0 else T,
                temps_fwd.max() if len(temps_fwd) > 0 else T)
    bins = np.linspace(0, t_max, 50)

    if len(temps_bwd) > 0:
        ax.hist(temps_bwd, bins=bins, density=True, alpha=0.6,
                color="#0072B2", edgecolor="white",
                label=f"Backward - coalescence (n={len(temps_bwd)})")
        ax.axvline(temps_bwd.mean(), color="#0072B2", lw=1.5, ls="--",
                   label=f"moy. backward = {temps_bwd.mean():.0f}")

    if len(temps_fwd) > 0:
        ax.hist(temps_fwd, bins=bins, density=True, alpha=0.6,
                color="#D55E00", edgecolor="white",
                label=f"Forward - divergence (n={len(temps_fwd)})")
        ax.axvline(temps_fwd.mean(), color="#D55E00", lw=1.5, ls="--",
                   label=f"moy. forward = {temps_fwd.mean():.0f}")

    # Courbe analytique backward : loi géométrique de paramètre 1/n^4.
    # Calculée sur les centres des bins pour être sur la même échelle
    # de densité que l'histogramme backward — plus besoin d'axe séparé.
    centres_bins = (bins[:-1] + bins[1:]) / 2
    largeur_bin = bins[1] - bins[0]
    p_t = (1 - 1 / n**4) ** (centres_bins - 1) * (1 / n**4)
    p_t = p_t / (p_t.sum() * largeur_bin)

    ax.plot(centres_bins, p_t, color="darkgreen", lw=1.5, ls="-.",
            label=f"$(1 - 1/n^4)^{{t-1}} \\cdot 1/n^4$ (backward analytique)")

    ax.legend(fontsize=8, framealpha=0.5)
    ax.set_xlabel("Temps (pas de Moran)")
    ax.set_ylabel("Densité")

    print(f"  Référence analytique : n^4 = {n**4} pas")
    if len(temps_bwd) > 0:
        print(f"  Backward : moy = {temps_bwd.mean():.1f}, écart-type = {temps_bwd.std():.1f}")
    if len(temps_fwd) > 0:
        print(f"  Forward  : moy = {temps_fwd.mean():.1f}, écart-type = {temps_fwd.std():.1f}")

    plt.tight_layout()
    if sauvegarder:
        nom = f"moran_v2_n{n}_T{T}.png"
        plt.savefig(nom, dpi=150)
        print(f"  Graphique sauvegardé : {nom}")
    if afficher:
        plt.show()
    plt.close()


# -----------------------------------------------------------------------------
# Lecture des arguments en ligne de commande
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Processus de Moran spatialisé : backward (coalescence) et forward (divergence)"
)

parser.add_argument("--n", type=int, default=10,
                    help="Taille de la grille n x n (défaut : 10)")

parser.add_argument("--T", type=int, default=5000,
                    help="Nombre de pas de Moran simulés (défaut : 5000)")

parser.add_argument("--rep", type=int, default=1000,
                    help="Nombre de répétitions (défaut : 1000)")

parser.add_argument("--mode", type=str, default="compare",
                    choices=["estimer_T", "compare"],
                    help=(
                        "estimer_T : estime le T_mrca pour calibrer --T. "
                        "compare   : simule backward et forward et compare (défaut)."
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

    print(f"\nSimulation backward (T={args.T}, rep={args.rep})...")
    t_bwd, d_bwd, n_nc_bwd = simuler_backward(args.n, args.T, args.rep, paires)
    taux_nc = n_nc_bwd / args.rep * 100
    print(f"  Coalescences trouvées : {len(t_bwd)} / {args.rep}  "
          f"({n_nc_bwd} non coalescées, soit {taux_nc:.1f}%"
          f"{' -- augmenter T !' if taux_nc > 5 else ''})")

    print(f"\nSimulation forward (T={args.T}, rep={args.rep})...")
    t_fwd, n_zero = simuler_forward(args.n, args.T, args.rep, paires)
    print(f"  Temps valides collectés : {len(t_fwd)} au total sur {args.rep} répétitions "
        f"(soit {len(t_fwd)/args.rep:.1f} en moyenne par répétition) "
        f"| répétitions sans aucun noeud valide : {n_zero}")

    if len(t_bwd) == 0 and len(t_fwd) == 0:
        print("\nAucun résultat à afficher. Augmenter T.")
        sys.exit(0)

    afficher_resultats(t_bwd, t_fwd, args.n, args.T,
                       args.afficher, args.sauvegarder)