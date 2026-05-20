# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Processus de Moran spatialisé - comparaison forward / backward (v4)
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
#   Par rapport à v3 :
#   - ajout de schémas d'échantillonnage pour le backward (inspiré de
#     Guindon & De Maio 2021) : uniforme, cluster, diagonale, bord, etc.
#   - ajout d'un filtre sur la distance initiale d0 entre les deux individus
#     tirés au présent (paramètres --d0_min et --d0_max).
#
# Usage :
#   python moran_spatialise_v4.py --n 7 --T 50000 --rep 200 --mode compare --afficher
#   python moran_spatialise_v4.py --n 7 --T 50000 --rep 200 --schema diagonale --afficher
#   python moran_spatialise_v4.py --n 7 --T 50000 --rep 200 --d0_min 1 --d0_max 1 --afficher
#
# Options :
#   --n : taille de la grille (défaut : 7)
#   --T : nombre de pas de Moran (défaut : 50000)
#   --rep : nombre de répétitions (défaut : 200)
#   --mode : estimer_T / compare
#   --schema : uniforme (défaut) | cluster_coin | cluster_centre |
#               diagonale | bord_droit | deux_clusters | surdisperse
#   --d0_min : distance initiale minimale entre les deux individus (défaut : 0)
#   --d0_max : distance initiale maximale (défaut : inf)
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
        x1, y1 : coordonnées de la première case
        x2, y2 : coordonnées de la deuxième case

    Retourne :
        float
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def trouver_case_plus_proche(n, px, py):
    """
    Retourne les coordonnées de la case de la grille la plus proche de (px, py).

    Utile pour projeter un point continu (tiré selon une loi quelconque)
    sur la grille entière n x n.

    Paramètres :
        n      : taille de la grille
        px, py : coordonnées du point continu (float)

    Retourne :
        (x, y) : case la plus proche (int, int)
    """
    x = int(np.clip(round(px), 0, n - 1)) #clip pour ne pas sortir de la grille et round pour prendre la case la plus proche
    y = int(np.clip(round(py), 0, n - 1))
    return x, y


def tirer_paire(n, schema, sigma=1.0, rayon=None):
    """
    Tire deux cases distinctes sur la grille selon un schéma d'échantillonnage.

    Schémas disponibles :
      uniforme  : tirage uniforme sur toutes les cases (défaut)
      cercle    : tirage dans un cercle de rayon `rayon` autour du centre
                  de la grille. Angle theta uniforme sur [0, 2*pi],
                  distance u uniforme sur [0, rayon] (distribution uniforme
                  sur le disque via u = rayon * sqrt(U) avec U ~ Unif[0,1]).
                  La case la plus proche est retenue via trouver_case_plus_proche.
      diagonale : tirage le long de la diagonale principale. On tire x
                  uniformément dans [0, n-1], puis y = x + epsilon avec
                  epsilon ~ N(0, sigma²). La case la plus proche est retenue.

    Paramètres :
        n : taille de la grille
        schema : nom du schéma (str)
        sigma : écart-type du bruit pour le schéma diagonale (défaut : 1.0)
        rayon : rayon du cercle pour le schéma cercle (défaut : n/4)

    Retourne :
        (x1, y1, x2, y2) : coordonnées des deux cases
    """
    if rayon is None:
        rayon = n / 4.0

    def tirer_un_point():
        if schema == "uniforme":
            return np.random.randint(0, n), np.random.randint(0, n)

        if schema == "cercle":
            cx, cy = (n - 1) / 2.0, (n - 1) / 2.0
            theta = np.random.uniform(0, 2 * np.pi)
            # sqrt pour distribution uniforme sur le disque 
            u = rayon * np.sqrt(np.random.uniform(0, 1))
            px = cx + np.cos(theta) * u
            py = cy + np.sin(theta) * u
            return trouver_case_plus_proche(n, px, py)

        if schema == "diagonale":
            px = np.random.uniform(0, n - 1)
            py = px + np.random.normal(0, sigma)
            return trouver_case_plus_proche(n, px, py)

        raise ValueError(f"Schéma inconnu : {schema}")

    # On tire deux points distincts
    for _ in range(10000):
        x1, y1 = tirer_un_point()
        x2, y2 = tirer_un_point()
        if (x1, y1) != (x2, y2):
            return x1, y1, x2, y2

    # Fallback si la zone est trop petite (ex. rayon très petit)
    return tirer_paire(n, "uniforme")


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

def simuler_backward(n, T, n_repetitions, paires,
                     schema="uniforme", d0_min=0.0, d0_max=float("inf"),
                     sigma=1.0, rayon=None):
    """
    Simule la coalescence de deux individus en remontant un tableau d'événements.

    On tire deux cases au présent selon le schéma d'échantillonnage choisi,
    avec filtre optionnel sur leur distance initiale d0.
    On remonte les événements un par un : si une case A a été remplacée
    (A <- B), sa lignée vient de B. Quand les deux cases coïncident,
    elles ont coalescé.

    Convention de temps : t=1 = 1 pas avant le présent (dernier événement),
    t=T = T pas avant le présent (premier événement).

    Paramètres :
        n : taille de la grille
        T : nombre de pas par simulation
        n_repetitions : nombre de paires tirées
        paires : liste des paires voisines
        schema : schéma d'échantillonnage (voir tirer_paire)
        d0_min : distance initiale minimale acceptée (défaut : 0)
        d0_max : distance initiale maximale acceptée (défaut : inf)
        sigma : écart-type pour le schéma diagonale (défaut : 1.0)
        rayon : rayon pour le schéma cercle (défaut : n/4)

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

        # Deux cases au présent selon le schéma, avec filtre sur d0
        for _ in range(10000):
            x1, y1, x2, y2 = tirer_paire(n, schema, sigma=sigma, rayon=rayon)
            d0 = distance(x1, y1, x2, y2)
            if d0_min <= d0 <= d0_max:
                break
        else:
            # Impossible de satisfaire le filtre : on saute cette répétition
            n_non_coal = n_non_coal + 1
            continue

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

def afficher_resultats(temps_bwd, temps_fwd, n, T, afficher, sauvegarder,
                       schema="uniforme", d0_min=0.0, d0_max=float("inf")):
    """
    Compare les distributions des temps backward et forward.

    Paramètres :
        temps_bwd : array des temps de coalescence (backward)
        temps_fwd : array des temps de divergence (forward, peut être vide)
        n : taille de la grille
        T : nombre de pas utilisés
        afficher : bool
        sauvegarder : bool
        schema : schéma d'échantillonnage utilisé (pour le titre)
        d0_min : filtre distance minimale (pour le titre)
        d0_max : filtre distance maximale (pour le titre)
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    # Titre : on ajoute le schéma et le filtre d0 si non-défaut
    titre = (f"Moran spatialisé : backward (coalescence)\n"
             f"grille {n}x{n}, T={T}, schéma={schema}")
    if d0_min > 0 or d0_max < float("inf"):
        d0_max_str = f"{d0_max:.1f}" if d0_max < float("inf") else "inf"
        titre = titre + f", d0 dans [{d0_min:.1f}, {d0_max_str}]"
    if len(temps_fwd) > 0:
        titre = titre.replace("backward (coalescence)", "backward vs forward")
    fig.suptitle(titre, fontsize=11)

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

    # Courbe analytique backward (valable uniquement pour schéma uniforme sans filtre d0)
    centres_bins = (bins[:-1] + bins[1:]) / 2
    largeur_bin = bins[1] - bins[0]
    p_t = (1 - 1 / n**4) ** (centres_bins - 1) * (1 / n**4)
    p_t = p_t / (p_t.sum() * largeur_bin)
    label_ana = f"$(1 - 1/n^4)^{{t-1}} \\cdot 1/n^4$ (uniforme)"
    ax.plot(centres_bins, p_t, color="darkgreen", lw=1.5, ls="-.",
            label=label_ana)

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
        d0_max_str = f"{d0_max:.0f}" if d0_max < float("inf") else "inf"
        nom = (f"moran_v4_{schema}_d0{d0_min:.0f}-{d0_max_str}"
               f"_n{n}_T{T}_rep{len(temps_bwd)}.png")
        plt.savefig(nom, dpi=150)
        print(f"  Graphique sauvegardé : {nom}")
    if afficher:
        plt.show()
    plt.close()


# =============================================================================
# Lecture des arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Moran spatialisé v4 : backward avec schémas d'échantillonnage"
)

parser.add_argument("--n", type=int, default=7,
                    help="Taille de la grille n x n (défaut : 7)")

parser.add_argument("--T", type=int, default=50000,
                    help="Nombre de pas de Moran (défaut : 50000)")

parser.add_argument("--rep", type=int, default=200,
                    help="Nombre de répétitions (défaut : 200)")

parser.add_argument("--mode", type=str, default="compare",
                    choices=["estimer_T", "compare"],
                    help="estimer_T : estime le T_mrca | compare : simule backward (et forward si pas de schéma)")

parser.add_argument("--schema", type=str, default="uniforme",
                    choices=["uniforme", "cercle", "diagonale"],
                    help="Schéma d'échantillonnage pour le tirage de la paire (défaut : uniforme)")

parser.add_argument("--sigma", type=float, default=1.0,
                    help="Écart-type du bruit pour le schéma diagonale (défaut : 1.0)")

parser.add_argument("--rayon", type=float, default=None,
                    help="Rayon du cercle pour le schéma cercle (défaut : n/4)")

parser.add_argument("--d0_min", type=float, default=0.0,
                    help="Distance initiale minimale entre les deux individus (défaut : 0)")

parser.add_argument("--d0_max", type=float, default=float("inf"),
                    help="Distance initiale maximale entre les deux individus (défaut : inf)")

parser.add_argument("--afficher", action="store_true",
                    help="Affiche les graphiques à l'écran")

parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde les graphiques en .png")

args = parser.parse_args()


# =============================================================================
# Lancement
# =============================================================================

paires = construire_paires_voisins(args.n)
rayon_effectif = args.rayon if args.rayon is not None else args.n / 4.0
print(f"Grille {args.n}x{args.n} | {args.n**2} noeuds | "
      f"{len(paires)} paires voisines | T={args.T} | mode={args.mode} | "
      f"schéma={args.schema} | sigma={args.sigma} | rayon={rayon_effectif:.1f} | "
      f"d0=[{args.d0_min}, "
      f"{'inf' if args.d0_max == float('inf') else args.d0_max}]")

if args.mode == "estimer_T":
    print(f"\nEstimation de T_mrca sur {args.rep} essais...")
    t_moy, t_max = estimer_T_mrca(args.n, n_essais=args.rep)
    print(f"  T_mrca moyen : {t_moy:.0f} pas")
    print(f"  T_mrca max   : {t_max} pas")
    print(f"  Suggestion : utiliser --T {int(t_max * 3)}")
    sys.exit(0)

if args.mode == "compare":

    print(f"\nSimulation backward (T={args.T}, rep={args.rep}, "
          f"schéma={args.schema})...")
    t_bwd, d_bwd, n_nc_bwd = simuler_backward(
        args.n, args.T, args.rep, paires,
        schema=args.schema, d0_min=args.d0_min, d0_max=args.d0_max,
        sigma=args.sigma, rayon=args.rayon
    )
    taux_nc = n_nc_bwd / args.rep * 100
    msg_nc = " -- augmenter T !" if taux_nc > 5 else ""
    print(f"  Coalescences trouvées : {len(t_bwd)} / {args.rep} "
          f"({n_nc_bwd} non coalescées, {taux_nc:.1f}%{msg_nc})")

    # Forward uniquement si schéma uniforme et pas de filtre d0
    # (le forward n'a pas de notion de schéma d'échantillonnage)
    t_fwd = np.array([])
    if args.schema == "uniforme" and args.d0_min == 0.0 and args.d0_max == float("inf"):
        print(f"\nSimulation forward (T={args.T}, rep={args.rep})...")
        t_fwd, n_zero = simuler_forward(args.n, args.T, args.rep, paires)
        print(f"  Temps valides : {len(t_fwd)} au total "
              f"({len(t_fwd)/args.rep:.1f} en moyenne par répétition) "
              f"| répétitions sans valide : {n_zero}")

    if len(t_bwd) == 0 and len(t_fwd) == 0:
        print("\nAucun résultat à afficher. Augmenter T.")
        sys.exit(0)

    afficher_resultats(t_bwd, t_fwd, args.n, args.T,
                       args.afficher, args.sauvegarder,
                       schema=args.schema,
                       d0_min=args.d0_min, d0_max=args.d0_max)