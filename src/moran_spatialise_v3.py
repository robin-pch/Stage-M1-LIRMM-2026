# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Processus de Moran spatialisé - comparaison forward / backward (v3)
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
#   Par rapport à v2, cette version utilise une représentation objet
#   (classe Noeud) inspirée du code de S. Guindon (ibd.py).
#   Sur chaque noeud interne on stocke le temps de bifurcation du père,
#   et les coordonnées du fils (où il se trouve sur la grille).
#   Le comptage de descendants se fait en post-ordre itératif :
#   on parcourt la liste de tous les noeuds à l'envers.
#
#   La racine globale n'est pas modélisée explicitement : chaque case
#   au temps 0 est la racine de son propre sous-arbre.
#
# Usage :
#   python moran_spatialise_v3.py --n 7 --T 50000 --rep 200 --mode compare --afficher
#   python moran_spatialise_v3.py --n 7 --mode estimer_T --rep 30
#
# Options :
#   --n : taille de la grille (défaut : 7)
#   --T : nombre de pas de Moran (défaut : 50000)
#   --rep : nombre de répétitions (défaut : 200)
#   --mode : estimer_T / compare
#   --afficher : affiche les graphiques
#   --sauvegarder : sauvegarde les graphiques en .png
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

    Attributs :
        x : coordonnée x du fils sur la grille
        y : coordonnée y du fils sur la grille
        temps : pas de Moran auquel ce noeud a bifurqué (temps absolu depuis t=0)
        fils1 : référence vers le fils gauche (Noeud ou None si feuille)
        fils2 : référence vers le fils droit (Noeud ou None si feuille)
        est_feuille : True si noeud terminal (pas encore bifurqué)
        desc1 : nombre de descendants vivants via fils1 (calculé en post-ordre)
        desc2 : nombre de descendants vivants via fils2 (calculé en post-ordre)
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

    def afficher(self, profondeur=0):
        # Affiche le noeud indenté (debug seulement, récursif)
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

def construire_paires_voisins(n):
    """
    Liste toutes les paires de cases adjacentes sur la grille n x n.

    Paramètres :
        n : taille de la grille

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
    Distance euclidienne entre deux cases.

    Paramètres :
        x1 : coordonnée x de la première case
        y1 : coordonnée y de la première case
        x2 : coordonnée x de la deuxième case
        y2 : coordonnée y de la deuxième case

    Retourne :
        float
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def generer_evenements(n, T, paires):
    """
    Génère une séquence de T événements Moran.

    Paramètres :
        n : taille de la grille
        T : nombre de pas
        paires : liste des paires voisines

    Retourne :
        evenements : liste de T tuples ((xA, yA), (xB, yB))
    """
    evenements = []
    for _ in range(T):
        A, B = paires[np.random.randint(len(paires))]
        # On tire au hasard quel individu meurt (A ou B)
        if np.random.rand() < 0.5:
            A, B = B, A
        evenements.append((A, B))
    return evenements


# =============================================================================
# Estimation de T_mrca
# =============================================================================

def estimer_T_mrca(n, n_essais):
    """
    Estime le nombre de pas pour que toute la grille ait un seul ancêtre commun.

    Paramètres :
        n : taille de la grille
        n_essais : nombre de répétitions

    Retourne :
        t_moyen : float
        t_max : int
    """
    paires = construire_paires_voisins(n)
    resultats = []

    for _ in range(n_essais):
        # Chaque case a un identifiant de lignée distinct au départ
        lignee = np.arange(n * n).reshape(n, n)
        t = 0
        while np.unique(lignee).size > 1:
            A, B = paires[np.random.randint(len(paires))]
            if np.random.rand() < 0.5:
                A, B = B, A
            # Toutes les cases avec la lignée de A prennent la lignée de B
            lignee[lignee == lignee[A]] = lignee[B]
            t = t + 1
        resultats.append(t)

    return np.mean(resultats), np.max(resultats)


# =============================================================================
# Simulation BACKWARD
# =============================================================================

def simuler_backward(n, T, n_repetitions, paires):
    """
    Simule la coalescence de deux individus en remontant un tableau d'événements.

    On tire deux cases distinctes au présent, puis on remonte les événements
    un par un. Si une case A a été remplacée lors d'un événement (A <- B),
    sa lignée vient de B : on remplace sa position par celle de B.
    Quand les deux cases se retrouvent au même endroit, elles ont coalescé.

    Convention de temps : t=1 = 1 pas avant le présent (dernier événement),
    t=T = T pas avant le présent (premier événement).

    Paramètres :
        n : taille de la grille
        T : nombre de pas par simulation
        n_repetitions : nombre de paires tirées
        paires : liste des paires voisines

    Retourne :
        temps_liste : array des temps de coalescence
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

        # On remonte les événements du plus récent (k=T-1) au plus ancien (k=0)
        for k in range(T - 1, -1, -1):
            A, B = evenements[k]

            # Si une des deux lignées était en A, elle vient en fait de B
            if A == (x1, y1):
                x1, y1 = B
            if A == (x2, y2):
                x2, y2 = B

            if (x1, y1) == (x2, y2):
                # t = T - k : nombre de pas remontés depuis le présent
                temps_liste.append(T - k)
                distances_liste.append(d0)
                coalesce = True
                break

        if not coalesce:
            n_non_coal = n_non_coal + 1

    return np.array(temps_liste), np.array(distances_liste), n_non_coal


# =============================================================================
# Simulation FORWARD (version objet, post-ordre itératif)
# =============================================================================

def simuler_forward(n, T, n_repetitions, paires):
    """
    Estime p(t) en forward en construisant un arbre de descendants.

    À chaque événement (A <- B) au pas t, l'occupant courant de B bifurque :
    il cesse d'être une feuille et se voit attribuer deux fils (un en A, un en B).
    On garde tous les noeuds dans une liste dans l'ordre de création.

    À la fin de la simulation, on parcourt cette liste à l'envers pour calculer
    le nombre de descendants vivants de chaque côté (post-ordre itératif).
    Un noeud est valide si desc1 >= 1 et desc2 >= 1.
    Il peut y avoir plusieurs noeuds valides par répétition.

    Convention de temps (même que backward) :
        t=1 = 1 pas avant le présent, t=T = T pas avant le présent.
        Les noeuds ont un temps absolu depuis t=0, converti en fin de fonction.

    Note sur la récursion :
        Une première version utilisait une récursion post-ordre. Elle provoquait
        un RecursionError dès T >  env.500 (profondeur de l'arbre env. T, limite Python
        env. 1000). Le parcours itératif à l'envers donne le même résultat.

    Paramètres :
        n : taille de la grille
        T : nombre de pas par simulation
        n_repetitions : nombre de répétitions
        paires : liste des paires voisines

    Retourne :
        temps_array : array de tous les temps valides (plusieurs par répétition)
        n_zero : nombre de répétitions sans aucun noeud valide
    """
    temps_liste = []
    n_zero = 0

    for _ in range(n_repetitions):
        evenements = generer_evenements(n, T, paires)

        # --- Initialisation de la grille ---
        # grille[x][y] pointe vers le Noeud actuellement en (x, y).
        # tous_les_noeuds garde tous les noeuds dans l'ordre de création :
        # utile pour le parcours itératif à l'envers.
        grille = [[None] * n for _ in range(n)]
        tous_les_noeuds = []

        for x in range(n):
            for y in range(n):
                noeud = Noeud(x=x, y=y, temps=0)
                grille[x][y] = noeud
                tous_les_noeuds.append(noeud)

        # --- Avancer dans le temps ---
        for t in range(T):
            A, B = evenements[t]
            xA, yA = A
            xB, yB = B

            # Le père est l'occupant courant de B
            pere = grille[xB][yB]

            # Deux fils : un va en A, un reste en B
            fils_A = Noeud(x=xA, y=yA, temps=t + 1)
            fils_B = Noeud(x=xB, y=yB, temps=t + 1)

            pere.fils1 = fils_A
            pere.fils2 = fils_B
            pere.est_feuille = False
            # On note le temps de la bifurcation sur le père
            pere.temps = t + 1

            # L'ancien occupant de A "meurt" : on le marque 
            grille[xA][yA].est_feuille = False
            grille[xA][yA] = fils_A
            grille[xB][yB] = fils_B

            tous_les_noeuds.append(fils_A)
            tous_les_noeuds.append(fils_B)

        # --- Marquer les feuilles vivantes ---
        # est_feuille=True signifie exactement "vivant au temps T" :
        # les feuilles mortes ont été marquées est_feuille=False au moment
        # où elles ont été écrasées par un événement A.
        for noeud in tous_les_noeuds:
            if noeud.est_feuille:
                noeud.desc1 = 1

        # --- Post-ordre itératif ---
        # Les pères sont toujours créés avant leurs fils dans tous_les_noeuds.
        # Parcourir à l'envers = traiter les fils avant leurs pères = post-ordre.
        for i in range(len(tous_les_noeuds) - 1, -1, -1):
            noeud = tous_les_noeuds[i]
            if noeud.fils1 is not None:  # noeud interne (a vraiment bifurqué)
                noeud.desc1 = noeud.fils1.desc1 + noeud.fils1.desc2
                noeud.desc2 = noeud.fils2.desc1 + noeud.fils2.desc2

        # --- Collecte des noeuds valides ---
        # On répète le temps desc1 * desc2 fois pour pondérer :
        # un vieux noeud avec beaucoup de descendants des deux côtés
        # a plus de chances d'être "vu" en backward, on compense ici.
        taille_avant = len(temps_liste)

        for noeud in tous_les_noeuds:
            if noeud.fils1 is not None:  # noeud interne (a vraiment bifurqué)
                if noeud.desc1 >= 1 and noeud.desc2 >= 1:
                    poids = noeud.desc1 * noeud.desc2
                    temps_liste.extend([noeud.temps] * poids)

        if len(temps_liste) == taille_avant:
            n_zero = n_zero + 1

    # --- Conversion en temps depuis le présent ---
    # noeud.temps est absolu (1 = premier événement = T pas avant le présent).
    # Conversion : t_depuis_present = T - noeud.temps + 1
    # Après conversion, t=1 = 1 pas avant le présent, comme en backward.
    temps_array = np.array(temps_liste)
    if len(temps_array) > 0:
        temps_array = T - temps_array + 1

    return temps_array, n_zero


# =============================================================================
# Affichage
# =============================================================================

def afficher_resultats(temps_bwd, temps_fwd, n, T, afficher, sauvegarder):
    """
    Compare les distributions des temps backward et forward.

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
        f"grille {n}x{n}, T={T}",
        fontsize=12
    )

    # Bornes communes pour les deux histogrammes
    t_max = T
    if len(temps_bwd) > 0 and temps_bwd.max() > t_max:
        t_max = temps_bwd.max()
    if len(temps_fwd) > 0 and temps_fwd.max() > t_max:
        t_max = temps_fwd.max()
    bins = np.linspace(0, t_max, 50)

    if len(temps_bwd) > 0:
        ax.hist(temps_bwd, bins=bins, density=True, alpha=0.6,
                color="#0072B2", edgecolor="white",
                label=f"Backward coalescence (n={len(temps_bwd)})")
        ax.axvline(temps_bwd.mean(), color="#0072B2", lw=1.5, ls="--",
                   label=f"moy. backward = {temps_bwd.mean():.0f}")

    if len(temps_fwd) > 0:
        ax.hist(temps_fwd, bins=bins, density=True, alpha=0.6,
                color="#D55E00", edgecolor="white",
                label=f"Forward divergence (n={len(temps_fwd)})")
        ax.axvline(temps_fwd.mean(), color="#D55E00", lw=1.5, ls="--",
                   label=f"moy. forward = {temps_fwd.mean():.0f}")

    # Courbe analytique backward : loi géométrique de paramètre 1/n^4
    centres_bins = (bins[:-1] + bins[1:]) / 2
    largeur_bin = bins[1] - bins[0]
    p_t = (1 - 1 / n**4) ** (centres_bins - 1) * (1 / n**4)
    p_t = p_t / (p_t.sum() * largeur_bin)

    ax.plot(centres_bins, p_t, color="darkgreen", lw=1.5, ls="-.",
            label=f"$(1 - 1/n^4)^{{t-1}} \\cdot 1/n^4$ analytique")

    ax.legend(fontsize=8, framealpha=0.5)
    ax.set_xlabel("Temps (pas de Moran, t=1 = présent)")
    ax.set_ylabel("Densité")

    print(f"  Référence analytique : n^4 = {n**4} pas")
    if len(temps_bwd) > 0:
        print(f"  Backward : moy = {temps_bwd.mean():.1f}, écart-type = {temps_bwd.std():.1f}")
    if len(temps_fwd) > 0:
        print(f"  Forward  : moy = {temps_fwd.mean():.1f}, écart-type = {temps_fwd.std():.1f}")

    plt.tight_layout()
    if sauvegarder:
        nom = f"moran_v3_n{n}_T{T}_rep{len(temps_bwd)}.png"
        plt.savefig(nom, dpi=150)
        print(f"  Graphique sauvegardé : {nom}")
    if afficher:
        plt.show()
    plt.close()


# =============================================================================
# Lecture des arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Moran spatialisé v3 : backward / forward (version objet)"
)

parser.add_argument("--n", type=int, default=7,
                    help="Taille de la grille n x n (défaut : 7)")

parser.add_argument("--T", type=int, default=50000,
                    help="Nombre de pas de Moran (défaut : 50000)")

parser.add_argument("--rep", type=int, default=200,
                    help="Nombre de répétitions (défaut : 200)")

parser.add_argument("--mode", type=str, default="compare",
                    choices=["estimer_T", "compare"],
                    help="estimer_T : estime le T_mrca | compare : simule backward et forward")

parser.add_argument("--afficher", action="store_true",
                    help="Affiche les graphiques à l'écran")

parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde les graphiques en .png")

args = parser.parse_args()


# =============================================================================
# Lancement
# =============================================================================

paires = construire_paires_voisins(args.n)
print(f"Grille {args.n}x{args.n} | {args.n**2} noeuds | "
      f"{len(paires)} paires voisines | T={args.T} | mode={args.mode}")

if args.mode == "estimer_T":
    print(f"\nEstimation de T_mrca sur {args.rep} essais...")
    t_moy, t_max = estimer_T_mrca(args.n, n_essais=args.rep)
    print(f"  T_mrca moyen : {t_moy:.0f} pas")
    print(f"  T_mrca max   : {t_max} pas")
    print(f"  Suggestion : utiliser --T {int(t_max * 3)}")
    sys.exit(0)

if args.mode == "compare":

    print(f"\nSimulation backward (T={args.T}, rep={args.rep})...")
    t_bwd, d_bwd, n_nc_bwd = simuler_backward(args.n, args.T, args.rep, paires)
    taux_nc = n_nc_bwd / args.rep * 100
    msg_nc = " -- augmenter T !" if taux_nc > 5 else ""
    print(f"  Coalescences trouvées : {len(t_bwd)} / {args.rep} "
          f"({n_nc_bwd} non coalescées, {taux_nc:.1f}%{msg_nc})")

    print(f"\nSimulation forward (T={args.T}, rep={args.rep})...")
    t_fwd, n_zero = simuler_forward(args.n, args.T, args.rep, paires)
    print(f"  Temps valides : {len(t_fwd)} au total "
          f"({len(t_fwd)/args.rep:.1f} en moyenne par répétition) "
          f"| répétitions sans valide : {n_zero}")

    if len(t_bwd) == 0 and len(t_fwd) == 0:
        print("\nAucun résultat à afficher. Augmenter T.")
        sys.exit(0)

    afficher_resultats(t_bwd, t_fwd, args.n, args.T,
                       args.afficher, args.sauvegarder)