# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Processus de Moran spatialisé - comparaison forward / backward (v6)
# ----------------------------------------------------------------------
# Auteur : Robin Pioch
# Stage M1 Bioinformatique, Université de Montpellier
# Encadrant : Stéphane Guindon (LIRMM)
# Mai 2026
#
# Ce que fait ce script :
#
#   On simule un processus de Moran spatialisé sur une grille l x l.
#   La population est n = l*l individus (un par case).
#   Règle du jeu : à chaque pas, on choisit au hasard deux cases voisines
#   A et B. L'individu en A meurt. L'individu en B se reproduit deux fois :
#   un enfant reste en B, l'autre va en A. La grille est toujours pleine.
#
#   Par rapport à v5 :
#   - l est maintenant le côté de la grille, n = l*l la population.
#     Cela évite la confusion dans la courbe analytique (n^4 = (l*l)^2,
#     pas l^4).
#   - La boucle principale est réorganisée : pour chaque répétition, on
#     génère les événements Moran une seule fois, on fait d'abord le
#     forward (qui donne un nombre de paires valides), puis on fait le
#     backward trois fois (uniforme, cercle, diagonale) en tirant à chaque
#     fois ce même nombre de paires.
#   - simuler_forward et simuler_backward sont devenus forward et backward
#     (la simulation proprement dite, c'est la génération des événements).
#   - calculer_n_eff : n_eff = nombre de cases dans la zone (entier),
#     directement comparable à n (population effective).
#   - Figures séparées : une pour forward vs backward uniforme, puis une
#     par schéma non-uniforme (cercle, diagonale) avec le backward uniforme
#     en référence grisée.
#   - Progression : une seule ligne qui s'écrase dans le terminal (\r).
#
# Usage :
#   python moran_spatialise_v6.py --l 7 --T 50000 --rep 200 --afficher
#   python moran_spatialise_v6.py --l 7 --T 50000 --rep 200 --sauvegarder
#
# Options :
#   --l : côté de la grille (défaut : 7), population = l*l
#   --T : nombre de pas de Moran (défaut : 50000)
#   --rep : nombre de répétitions (défaut : 200)
#   --mode : estimer_T / compare
#   --sigma : écart-type du bruit pour le schéma diagonale (défaut : 1.0)
#   --rayon : rayon du cercle pour le schéma cercle (défaut : l/4)
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
    Un noeud dans l'arbre de descendance (forward).

    Attributs :
        x, y : position sur la grille
        temps : pas absolu de création (depuis t=0)
        fils1, fils2 : fils gauche et droit (None si feuille)
        est_feuille : True tant que le noeud n'a pas bifurqué
        desc1, desc2 : nombre de descendants vivants de chaque côté
        feuilles1, feuilles2 : coordonnées (x, y) des feuilles vivantes de chaque côté
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
        self.feuilles1 = []  # coordonnées des feuilles vivantes côté fils1
        self.feuilles2 = []  # coordonnées des feuilles vivantes côté fils2

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

def construire_paires_voisins(l):
    """
    Liste toutes les paires de cases adjacentes sur la grille l x l.

    Paramètres :
        l : côté de la grille

    Retourne :
        paires : liste de tuples ((x1, y1), (x2, y2))
    """
    paires = []
    for x in range(l):
        for y in range(l):
            if x + 1 < l:
                paires.append(((x, y), (x + 1, y)))
            if y + 1 < l:
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


def trouver_case_plus_proche(l, px, py):
    """
    Retourne les coordonnées de la case de la grille la plus proche de (px, py).

    Utile pour projeter un point continu (tiré selon une loi quelconque)
    sur la grille entière l x l.

    Paramètres :
        l : côté de la grille
        px, py : coordonnées du point continu (float)

    Retourne :
        (x, y) : case la plus proche (int, int)
    """
    x = int(np.clip(round(px), 0, l - 1))  # clip pour ne pas sortir de la grille
    y = int(np.clip(round(py), 0, l - 1))
    return x, y


def tirer_paire(l, schema, sigma=1.0, rayon=None):
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
                  uniformément dans [0, l-1], puis y = x + epsilon avec
                  epsilon ~ N(0, sigma²). La case la plus proche est retenue.

    Paramètres :
        l : côté de la grille
        schema : nom du schéma (str)
        sigma : écart-type du bruit pour le schéma diagonale (défaut : 1.0)
        rayon : rayon du cercle pour le schéma cercle (défaut : l/4)

    Retourne :
        (x1, y1, x2, y2) : coordonnées des deux cases
    """
    if rayon is None:
        rayon = l / 4.0

    def tirer_un_point():
        if schema == "uniforme":
            return np.random.randint(0, l), np.random.randint(0, l)

        if schema == "cercle":
            cx, cy = (l - 1) / 2.0, (l - 1) / 2.0
            theta = np.random.uniform(0, 2 * np.pi)
            # sqrt pour distribution uniforme sur le disque
            u = rayon * np.sqrt(np.random.uniform(0, 1))
            px = cx + np.cos(theta) * u
            py = cy + np.sin(theta) * u
            return trouver_case_plus_proche(l, px, py)

        if schema == "diagonale":
            px = np.random.uniform(0, l - 1)
            py = px + np.random.normal(0, sigma)
            return trouver_case_plus_proche(l, px, py)

        raise ValueError(f"Schéma inconnu : {schema}")

    # On tire deux points distincts
    for _ in range(10000):
        x1, y1 = tirer_un_point()
        x2, y2 = tirer_un_point()
        if (x1, y1) != (x2, y2):
            return x1, y1, x2, y2

    # Fallback si la zone est trop petite (ex. rayon très petit)
    return tirer_paire(l, "uniforme")


def generer_evenements(l, T, paires):
    """
    Génère une séquence de T événements Moran.

    Paramètres :
        l : côté de la grille
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

def estimer_T_mrca(l, n_essais):
    """
    Estime le nombre de pas pour que toute la grille ait un seul ancêtre commun.

    Paramètres :
        l : côté de la grille
        n_essais : nombre de répétitions

    Retourne :
        t_moyen : float
        t_max : int
    """
    paires = construire_paires_voisins(l)
    resultats = []

    for _ in range(n_essais):
        # Chaque case a un identifiant de lignée distinct au départ
        lignee = np.arange(l * l).reshape(l, l)
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
# Simulation FORWARD
# =============================================================================

def forward(l, T, evenements):
    """
    Calcule p(t) en forward en construisant un arbre de descendants.

    À chaque événement (A <- B), l'occupant de B bifurque : il devient un noeud
    interne avec deux fils (un en A, un en B). Tous les noeuds sont stockés dans
    l'ordre de création.

    À la fin, on parcourt la liste à l'envers (post-ordre itératif) pour calculer
    le nombre de descendants vivants de chaque côté. Un noeud est valide si
    desc1 >= 1 et desc2 >= 1. On calcule aussi toutes les distances entre les
    feuilles gauche et droite pour chaque noeud valide.

    Convention de temps : t=1 = 1 pas avant le présent.
    Les temps absolus des noeuds sont convertis en fin de fonction.

    Note : une version récursive avait été essayée mais provoquait des
    RecursionError pour T > ~500. Le parcours itératif à l'envers est équivalent.

    Paramètres :
        l : côté de la grille
        T : nombre de pas par simulation
        evenements : liste des T événements Moran générés pour cette répétition

    Retourne :
        temps_liste : liste des temps de divergence pour tous les noeuds valides
        distances_liste : liste des distances entre paires de feuilles correspondantes
        n_paires : nombre total de paires valides trouvées
    """
    # --- Initialisation de la grille ---
    # grille[x][y] pointe vers le Noeud actuellement en (x, y).
    # tous_les_noeuds garde tous les noeuds dans l'ordre de création :
    # utile pour le parcours itératif à l'envers.
    grille = [[None] * l for _ in range(l)]
    tous_les_noeuds = []

    for x in range(l):
        for y in range(l):
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
            # Une feuille vivante se référence elle-même dans feuilles1
            noeud.feuilles1 = [(noeud.x, noeud.y)]

    # --- Post-ordre itératif ---
    # Les pères sont toujours créés avant leurs fils dans tous_les_noeuds.
    # Parcourir à l'envers = traiter les fils avant leurs pères = post-ordre.
    for i in range(len(tous_les_noeuds) - 1, -1, -1):
        noeud = tous_les_noeuds[i]
        if noeud.fils1 is not None:  # noeud interne (a vraiment bifurqué)
            noeud.desc1 = noeud.fils1.desc1 + noeud.fils1.desc2
            noeud.desc2 = noeud.fils2.desc1 + noeud.fils2.desc2
            # feuilles accessibles via fils1 = toutes celles de fils1 des deux côtés
            noeud.feuilles1 = noeud.fils1.feuilles1 + noeud.fils1.feuilles2
            noeud.feuilles2 = noeud.fils2.feuilles1 + noeud.fils2.feuilles2

    # --- Collecte des noeuds valides ---
    # Pour chaque noeud valide, on calcule toutes les distances entre
    # les feuilles du côté gauche et les feuilles du côté droit.
    temps_liste = []
    distances_liste = []

    for noeud in tous_les_noeuds:
        if noeud.fils1 is not None:  # noeud interne
            if noeud.desc1 >= 1 and noeud.desc2 >= 1:
                # Toutes les combinaisons feuille_gauche x feuille_droite
                for (xa, ya) in noeud.feuilles1:
                    for (xb, yb) in noeud.feuilles2:
                        d = distance(xa, ya, xb, yb)
                        temps_liste.append(noeud.temps)
                        distances_liste.append(d)

    n_paires = len(temps_liste)

    # --- Conversion en temps depuis le présent ---
    # noeud.temps est absolu (1 = premier événement = T pas avant le présent).
    # Conversion : t_depuis_present = T - noeud.temps + 1
    # Après conversion, t=1 = 1 pas avant le présent, comme en backward.
    temps_array = np.array(temps_liste)
    distances_array = np.array(distances_liste)
    if len(temps_array) > 0:
        temps_array = T - temps_array + 1

    return temps_array, distances_array, n_paires


# =============================================================================
# Simulation BACKWARD
# =============================================================================

def backward(l, T, n_paires, evenements, schema="uniforme",
             d0_min=0.0, d0_max=float("inf"), sigma=1.0, rayon=None):
    """
    Simule la coalescence de n_paires paires en remontant un tableau d'événements.

    On tire n_paires paires au présent selon le schéma d'échantillonnage choisi,
    avec filtre optionnel sur leur distance initiale d0.
    On remonte les événements un par un : si une case A a été remplacée
    (A <- B), sa lignée vient de B. Quand les deux cases coïncident,
    elles ont coalescé.

    Convention de temps : t=1 = 1 pas avant le présent (dernier événement),
    t=T = T pas avant le présent (premier événement).

    Paramètres :
        l : côté de la grille
        T : nombre de pas par simulation
        n_paires : nombre de paires à tirer (fourni par le forward)
        evenements : liste des T événements Moran (les mêmes que pour le forward)
        schema : schéma d'échantillonnage (voir tirer_paire)
        d0_min : distance initiale minimale acceptée (défaut : 0)
        d0_max : distance initiale maximale acceptée (défaut : inf)
        sigma : écart-type pour le schéma diagonale (défaut : 1.0)
        rayon : rayon pour le schéma cercle (défaut : l/4)

    Retourne :
        temps_liste : array des temps de coalescence
        distances_liste : array des distances initiales correspondantes
        n_non_coal : nombre de paires non coalescées dans [0, T]
    """
    temps_liste = []
    distances_liste = []
    n_non_coal = 0

    for _ in range(n_paires):

        # Deux cases au présent selon le schéma, avec filtre sur d0
        for _ in range(10000):
            x1, y1, x2, y2 = tirer_paire(l, schema, sigma=sigma, rayon=rayon)
            d0 = distance(x1, y1, x2, y2)
            if d0_min <= d0 <= d0_max:
                break
        else:
            # Impossible de satisfaire le filtre : on saute cette paire
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
# Affichage
# =============================================================================

def calculer_n_eff(l, schema, sigma=1.0, rayon=None):
    """
    Calcule n_eff = nombre de cases dans la zone d'échantillonnage.

    Pour la courbe analytique avec schéma non uniforme, on remplace n (= l*l,
    population totale) par n_eff = nombre de cases dans la zone. Ainsi n_eff
    joue le même rôle que n dans la formule analytique (1 - 1/n_eff^2)^(t-1)
    * 1/n_eff^2, n_eff cases dans la zone jouent le même rôle que n dans la formule uniforme.

    Pour cercle : cases dont la distance au centre est <= rayon.
    Pour diagonale : cases avec |y - x| <= sigma (bande autour de y = x,
    cohérent avec le bruit utilisé dans tirer_paire).

    Paramètres :
        l : côté de la grille
        schema : "uniforme", "cercle" ou "diagonale"
        sigma : écart-type pour le schéma diagonale
        rayon : rayon pour le schéma cercle (défaut : l/4)

    Retourne :
        n_eff : int (nombre de cases dans la zone)
    """
    if rayon is None:
        rayon = l / 4.0

    if schema == "uniforme":
        # population totale
        return l * l

    cx, cy = (l - 1) / 2.0, (l - 1) / 2.0

    if schema == "cercle":
        n_eff = 0
        for x in range(l):
            for y in range(l):
                if distance(x, y, cx, cy) <= rayon:
                    n_eff = n_eff + 1
        return n_eff

    if schema == "diagonale":
        # Cases dans la bande à ±sigma autour de la diagonale y = x
        n_eff = 0
        for x in range(l):
            for y in range(l):
                if abs(y - x) <= sigma:
                    n_eff = n_eff + 1
        return n_eff

    # Schéma inconnu : on replie sur la population totale
    return l * l


def afficher_resultats(resultats_fwd, resultats_bwd, l, T,
                       afficher, sauvegarder, sigma=1.0, rayon=None):
    """
    Produit plusieurs figures :
    - Figure 1 : forward vs backward uniforme (la comparaison principale),
      avec la courbe analytique uniforme.
    - Figure 2 : distribution jointe (temps, distance) pour le backward uniforme
      et le forward.
    - Figure 3+ : une figure par schéma non-uniforme (cercle, diagonale),
      avec backward du schéma vs backward uniforme et courbe analytique du schéma.

    resultats_fwd et resultats_bwd sont des dictionnaires :
        resultats_fwd["temps"]     : array des temps forward
        resultats_fwd["distances"] : array des distances forward
        resultats_bwd["uniforme"]["temps"]     : array des temps backward uniforme
        resultats_bwd["uniforme"]["distances"] : array des distances backward uniforme
        (idem pour "cercle" et "diagonale")

    Paramètres :
        resultats_fwd : dict avec clés "temps" et "distances"
        resultats_bwd : dict de dicts, une entrée par schéma
        l : côté de la grille
        T : nombre de pas
        afficher : affiche les figures à l'écran
        sauvegarder : sauvegarde en .png
        sigma, rayon : paramètres des schémas (pour calculer n_eff)
    """
    n = l * l  # population totale

    t_fwd = resultats_fwd["temps"]
    d_fwd = resultats_fwd["distances"]
    t_bwd_unif = resultats_bwd["uniforme"]["temps"]
    d_bwd_unif = resultats_bwd["uniforme"]["distances"]

    # Bins communs pour toutes les figures marginales
    # (même axe x pour pouvoir comparer visuellement)
    t_max = T
    for schema in resultats_bwd:
        t_bwd = resultats_bwd[schema]["temps"]
        if len(t_bwd) > 0 and t_bwd.max() > t_max:
            t_max = t_bwd.max()
    if len(t_fwd) > 0 and t_fwd.max() > t_max:
        t_max = t_fwd.max()
    bins = np.linspace(0, t_max, 50)
    centres_bins = (bins[:-1] + bins[1:]) / 2
    largeur_bin = bins[1] - bins[0]

    # --- Figure 1 : forward vs backward uniforme ---
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    titre1 = (f"Moran spatialisé : forward vs backward uniforme — "
              f"grille {l}x{l} (n={n}), T={T}")
    fig1.suptitle(titre1, fontsize=11)

    if len(t_fwd) > 0:
        ax1.hist(t_fwd, bins=bins, density=True, alpha=0.5,
                 color="#D55E00", edgecolor="white",
                 label=f"Forward (n={len(t_fwd)})")
        ax1.axvline(t_fwd.mean(), color="#D55E00", lw=1.5, ls="--",
                    label=f"moy. forward = {t_fwd.mean():.0f}")

    if len(t_bwd_unif) > 0:
        ax1.hist(t_bwd_unif, bins=bins, density=True, alpha=0.5,
                 color="#0072B2", edgecolor="white",
                 label=f"Backward uniforme (n={len(t_bwd_unif)})")
        ax1.axvline(t_bwd_unif.mean(), color="#0072B2", lw=1.5, ls="--",
                    label=f"moy. bwd uniforme = {t_bwd_unif.mean():.0f}")

    # Courbe analytique uniforme : p(t) = (1 - 1/n^2)^(t-1) * 1/n^2
    n_eff_unif = calculer_n_eff(l, "uniforme")
    p_t_unif = (1 - 1 / n_eff_unif**2) ** (centres_bins - 1) * (1 / n_eff_unif**2)
    p_t_unif = p_t_unif / (p_t_unif.sum() * largeur_bin)
    ax1.plot(centres_bins, p_t_unif, color="darkgreen", lw=1.5, ls="-.",
             label=f"analytique uniforme : $n^2={n_eff_unif**2}$")

    ax1.legend(fontsize=9, framealpha=0.5)
    ax1.set_xlabel("Temps (pas de Moran, t=1 = présent)")
    ax1.set_ylabel("Densité")

    print(f"  Référence analytique : n^2 = {n**2} pas (n={n}, grille {l}x{l})")
    if len(t_fwd) > 0:
        print(f"  Forward   : moy = {t_fwd.mean():.1f}, écart-type = {t_fwd.std():.1f}")
    if len(t_bwd_unif) > 0:
        print(f"  Bwd unif. : moy = {t_bwd_unif.mean():.1f}, écart-type = {t_bwd_unif.std():.1f}")

    plt.tight_layout()
    if sauvegarder:
        nom = f"moran_v6_l{l}_T{T}_uniforme.png"
        plt.savefig(nom, dpi=150)
        print(f"  Graphique sauvegardé : {nom}")
    if afficher:
        plt.show()
    plt.close(fig1)

    # --- Figure 2 : distribution jointe temps / distance (uniforme + forward) ---
    n_sous_figures = 0
    if len(t_bwd_unif) > 0 and len(d_bwd_unif) > 0:
        n_sous_figures = n_sous_figures + 1
    if len(t_fwd) > 0 and len(d_fwd) > 0:
        n_sous_figures = n_sous_figures + 1

    if n_sous_figures > 0:
        fig2, axes = plt.subplots(1, n_sous_figures, figsize=(6 * n_sous_figures, 5))
        # Si un seul panneau, axes n'est pas une liste
        if n_sous_figures == 1:
            axes = [axes]

        titre2 = f"Distribution jointe temps / distance — grille {l}x{l} (n={n}), T={T}"
        fig2.suptitle(titre2, fontsize=11)

        idx = 0

        if len(t_bwd_unif) > 0 and len(d_bwd_unif) > 0:
            ax = axes[idx]
            idx = idx + 1

            bins_t = np.linspace(0, t_bwd_unif.max(), 40)
            bins_d = np.linspace(0, d_bwd_unif.max() + 0.5, 30)

            ax.hist2d(t_bwd_unif, d_bwd_unif, bins=[bins_t, bins_d],
                      cmap="Blues", density=True)
            ax.set_xlabel("Temps de coalescence t")
            ax.set_ylabel("Distance initiale d0")
            ax.set_title(f"Backward uniforme (n={len(t_bwd_unif)} points)")

            ax.axhline(d_bwd_unif.mean(), color="#0072B2", lw=1.2, ls="--",
                       label=f"moy. d0 = {d_bwd_unif.mean():.2f}")
            ax.legend(fontsize=8)

        if len(t_fwd) > 0 and len(d_fwd) > 0:
            ax = axes[idx]

            bins_t = np.linspace(0, t_fwd.max(), 40)
            bins_d = np.linspace(0, d_fwd.max() + 0.5, 30)

            ax.hist2d(t_fwd, d_fwd, bins=[bins_t, bins_d],
                      cmap="Oranges", density=True)
            ax.set_xlabel("Temps de divergence t")
            ax.set_ylabel("Distance entre feuilles (gauche × droite)")
            ax.set_title(f"Forward (n={len(t_fwd)} paires de feuilles)")

            ax.axhline(d_fwd.mean(), color="#D55E00", lw=1.2, ls="--",
                       label=f"moy. dist = {d_fwd.mean():.2f}")
            ax.legend(fontsize=8)

        plt.tight_layout()
        if sauvegarder:
            nom = f"moran_v6_l{l}_T{T}_joint.png"
            plt.savefig(nom, dpi=150)
            print(f"  Graphique sauvegardé : {nom}")
        if afficher:
            plt.show()
        plt.close(fig2)

    # --- Figures 3+ : une figure par schéma non-uniforme ---
    # Pour chaque schéma, on compare le backward du schéma au backward uniforme,
    # avec la courbe analytique du schéma.
    for schema in ["cercle", "diagonale"]:
        t_bwd = resultats_bwd[schema]["temps"]

        if len(t_bwd) == 0:
            continue

        fig, ax = plt.subplots(figsize=(9, 5))
        titre = (f"Moran spatialisé : backward {schema} vs uniforme — "
                 f"grille {l}x{l} (n={n}), T={T}")
        fig.suptitle(titre, fontsize=11)

        # Backward uniforme en référence (grisé)
        if len(t_bwd_unif) > 0:
            ax.hist(t_bwd_unif, bins=bins, density=True, alpha=0.3,
                    color="gray", edgecolor="white",
                    label=f"Backward uniforme (référence, n={len(t_bwd_unif)})")

        # Backward du schéma
        ax.hist(t_bwd, bins=bins, density=True, alpha=0.5,
                color="#0072B2", edgecolor="white",
                label=f"Backward {schema} (n={len(t_bwd)})")
        ax.axvline(t_bwd.mean(), color="#0072B2", lw=1.5, ls="--",
                   label=f"moy. bwd {schema} = {t_bwd.mean():.0f}")

        # Courbe analytique du schéma
        n_eff = calculer_n_eff(l, schema, sigma=sigma, rayon=rayon)
        p_t = (1 - 1 / n_eff**2) ** (centres_bins - 1) * (1 / n_eff**2)
        p_t = p_t / (p_t.sum() * largeur_bin)
        ax.plot(centres_bins, p_t, color="#D55E00", lw=1.5, ls="-.",
                label=f"analytique {schema} : $n_{{eff}}^2={n_eff**2}$ ($n_{{eff}}={n_eff}$)")

        print(f"  n_eff ({schema}) = {n_eff}  ->  n_eff^2 = {n_eff**2}")
        print(f"  Bwd {schema} : moy = {t_bwd.mean():.1f}, écart-type = {t_bwd.std():.1f}")

        ax.legend(fontsize=9, framealpha=0.5)
        ax.set_xlabel("Temps (pas de Moran, t=1 = présent)")
        ax.set_ylabel("Densité")

        plt.tight_layout()
        if sauvegarder:
            nom = f"moran_v6_l{l}_T{T}_{schema}.png"
            plt.savefig(nom, dpi=150)
            print(f"  Graphique sauvegardé : {nom}")
        if afficher:
            plt.show()
        plt.close(fig)


# =============================================================================
# Lecture des arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Moran spatialisé v6 : forward puis backward (3 schémas) par répétition"
)

parser.add_argument("--l", type=int, default=7,
                    help="Côté de la grille l x l (défaut : 7), population = l*l")

parser.add_argument("--T", type=int, default=50000,
                    help="Nombre de pas de Moran (défaut : 50000)")

parser.add_argument("--rep", type=int, default=200,
                    help="Nombre de répétitions (défaut : 200)")

parser.add_argument("--mode", type=str, default="compare",
                    choices=["estimer_T", "compare"],
                    help="estimer_T : estime le T_mrca | compare : simule forward + backward")

parser.add_argument("--sigma", type=float, default=1.0,
                    help="Écart-type du bruit pour le schéma diagonale (défaut : 1.0)")

parser.add_argument("--rayon", type=float, default=None,
                    help="Rayon du cercle pour le schéma cercle (défaut : l/4)")

parser.add_argument("--afficher", action="store_true",
                    help="Affiche les graphiques à l'écran")

parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde les graphiques en .png")

args = parser.parse_args()


# =============================================================================
# Lancement
# =============================================================================

paires = construire_paires_voisins(args.l)
n = args.l * args.l  # population totale
rayon_effectif = args.rayon if args.rayon is not None else args.l / 4.0

print(f"Grille {args.l}x{args.l} | population n={n} | "
      f"{len(paires)} paires voisines | T={args.T} | mode={args.mode} | "
      f"sigma={args.sigma} | rayon={rayon_effectif:.1f}")

if args.mode == "estimer_T":
    print(f"\nEstimation de T_mrca sur {args.rep} essais...")
    t_moy, t_max = estimer_T_mrca(args.l, n_essais=args.rep)
    print(f"  T_mrca moyen : {t_moy:.0f} pas")
    print(f"  T_mrca max   : {t_max} pas")
    print(f"  Suggestion : utiliser --T {int(t_max * 3)}")
    sys.exit(0)

if args.mode == "compare":

    print(f"\nSimulation (T={args.T}, rep={args.rep})...")

    # Accumulateurs pour tous les résultats
    temps_fwd_total = []
    distances_fwd_total = []

    temps_bwd_total = {"uniforme": [], "cercle": [], "diagonale": []}
    distances_bwd_total = {"uniforme": [], "cercle": [], "diagonale": []}
    n_nc_bwd = {"uniforme": 0, "cercle": 0, "diagonale": 0}
    n_zero_fwd = 0

    for rep in range(args.rep):

        print(f"  répétition {rep + 1}/{args.rep}...", end="\r", flush=True)

        # 1. On génère les événements Moran une seule fois pour cette répétition
        evenements = generer_evenements(args.l, args.T, paires)

        # 2. Forward : donne le nombre de paires valides pour cette répétition
        t_fwd, d_fwd, n_paires = forward(args.l, args.T, evenements)

        if n_paires == 0:
            n_zero_fwd = n_zero_fwd + 1
            # Pas de paires forward, on passe la répétition
            continue

        temps_fwd_total.extend(t_fwd)
        distances_fwd_total.extend(d_fwd)

        # 3. Backward pour chacun des 3 schémas, en tirant n_paires paires
        for schema in ["uniforme", "cercle", "diagonale"]:
            t_bwd, d_bwd, n_nc = backward(
                args.l, args.T, n_paires, evenements,
                schema=schema, sigma=args.sigma, rayon=args.rayon
            )
            temps_bwd_total[schema].extend(t_bwd)
            distances_bwd_total[schema].extend(d_bwd)
            n_nc_bwd[schema] = n_nc_bwd[schema] + n_nc

    # Conversion en arrays numpy
    t_fwd_arr = np.array(temps_fwd_total)
    d_fwd_arr = np.array(distances_fwd_total)

    resultats_fwd = {"temps": t_fwd_arr, "distances": d_fwd_arr}

    resultats_bwd = {}
    for schema in ["uniforme", "cercle", "diagonale"]:
        t_arr = np.array(temps_bwd_total[schema])
        d_arr = np.array(distances_bwd_total[schema])
        resultats_bwd[schema] = {"temps": t_arr, "distances": d_arr}

    # Affichage des stats
    print(f"\nRésultats :")
    print(f"  Forward : {len(t_fwd_arr)} paires valides au total "
          f"({len(t_fwd_arr) / args.rep:.1f} en moy. par répétition) "
          f"| répétitions sans paires : {n_zero_fwd}")

    for schema in ["uniforme", "cercle", "diagonale"]:
        t_bwd = resultats_bwd[schema]["temps"]
        n_total_bwd = len(t_bwd) + n_nc_bwd[schema]
        taux_nc = n_nc_bwd[schema] / n_total_bwd * 100 if n_total_bwd > 0 else 0
        msg_nc = " -- augmenter T !" if taux_nc > 5 else ""
        print(f"  Backward {schema} : {len(t_bwd)} coalescences "
              f"({n_nc_bwd[schema]} non coalescées, {taux_nc:.1f}%{msg_nc})")

    if len(t_fwd_arr) == 0:
        print("\nAucun résultat à afficher. Augmenter T.")
        sys.exit(0)

    afficher_resultats(
        resultats_fwd, resultats_bwd,
        args.l, args.T,
        args.afficher, args.sauvegarder,
        sigma=args.sigma, rayon=args.rayon
    )