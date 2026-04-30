# ----------------------------------------------------------------------
# Vérification de l'équivalence entre M1 et M2
# via la distribution jointe (distance, temps)
# ----------------------------------------------------------------------
# Auteur : Robin Pioch
# Contexte : Stage M1 Bioinformatique : Université de Montpellier
#            Sous la direction de Stéphane Guindon (LIRMM)
# Date : Avril 2026
#
# Description :
#   Ce script vérifie que les modèles M1 et M2 sont statistiquement
#   équivalents en comparant leur distribution jointe (distance, temps).
#
#   M1 : forward-in-time (adapté pour l'équivalence) :
#     On tire t uniformément dans [0, T], puis deux marcheurs partent
#     du même point et marchent pendant t pas. On enregistre (t, distance).
#
#   M2 : backward-in-time (adapté pour l'équivalence) :
#     On tire uniformément les positions de départ des deux marcheurs
#     sur la grille. On stoppe à la coalescence et on enregistre
#     (temps_coalescence, distance_initiale).
#
#   Si les deux modèles sont équivalents, les distributions de
#   (distance, temps) doivent être identiques.
#
# Usage :
#   python equivalence_M1_M2.py --m 0.5 --n 50 --T 500 --rep 2000
#
#   Options supplémentaires :
#     --afficher      affiche les graphiques à l'écran
#     --sauvegarder   sauvegarde les graphiques en .png
# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import argparse


def faire_un_pas(x, y, m, n):
    """
    Déplace un marcheur d'un pas sur la grille.

    À chaque appel le marcheur reste sur place avec probabilité (1 - m),
    ou se déplace vers l'un des voisins valides avec probabilité m.
    Les bords sont réfléchissants : un marcheur en bord de grille n'a que
    2 ou 3 voisins disponibles selon sa position (bord ou coin).

    Paramètres :
        x, y : position actuelle du marcheur (entiers)
        m : probabilité de se déplacer à chaque pas (float entre 0 et 1)
        n : taille de la grille (grille carrée n x n)

    Retourne :
        x, y : nouvelle position du marcheur
    """
    if np.random.rand() > m:
        return x, y  # le marcheur reste sur place

    # On construit la liste des voisins valides selon la position (bords réfléchissants)
    voisins = []
    if x > 0:
        voisins.append((x - 1, y)) # gauche
    if x < n - 1:
        voisins.append((x + 1, y)) # droite
    if y > 0:
        voisins.append((x, y - 1)) # bas
    if y < n - 1:
        voisins.append((x, y + 1)) # haut

    # On choisit un voisin au hasard parmi ceux disponibles
    idx = np.random.randint(len(voisins))
    x, y = voisins[idx]

    return x, y


def distance_euclidienne(x1, y1, x2, y2):
    """
    Calcule la distance euclidienne entre deux points de la grille.

    La formule de distance en 2D :
    sqrt((x2 - x1)^2 + (y2 - y1)^2).

    Paramètres :
        x1, y1 : coordonnées du premier marcheur
        x2, y2 : coordonnées du second marcheur

    Retourne :
        distance : float
    """
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def simuler_M1_joint(m, n, T, n_repetitions):
    """
    Simule M1 pour obtenir la distribution jointe (temps, distance).

    À chaque répétition, on tire t uniformément dans [0, T], puis on fait
    marcher deux marcheurs depuis un point tiré au hasard sur la grille pendant t pas.

    Paramètres :
        m : probabilité de se déplacer à chaque pas
        n : taille de la grille (n x n)
        T : temps maximal (t est tiré uniformément dans [0, T])
        n_repetitions : nombre de répétitions de la simulation

    Retourne :
        temps : liste des temps tirés (une valeur par répétition)
        distances : liste des distances finales correspondantes
    """
    temps = []
    distances = []

    for rep in range(n_repetitions):

        # On tire un temps aléatoire dans [0, T]
        t = np.random.randint(0, T + 1)

        # Les deux marcheurs partent du même point tiré au hasard sur la grille
        x0, y0 = np.random.randint(0, n), np.random.randint(0, n)
        x1, y1 = x0, y0
        x2, y2 = x0, y0

        # On fait avancer les deux marcheurs pendant t pas
        for i in range(t):
            x1, y1 = faire_un_pas(x1, y1, m, n)
            x2, y2 = faire_un_pas(x2, y2, m, n)

        d = distance_euclidienne(x1, y1, x2, y2)
        temps.append(t)
        distances.append(d)

    return np.array(temps), np.array(distances)


def simuler_M2_joint(m, n, T, n_repetitions):
    """
    Simule M2 (coalescence) pour obtenir la distribution jointe
    (temps de coalescence, distance initiale).

    À chaque répétition, on tire uniformément les positions de départ
    des deux marcheurs sur la grille. On les fait marcher jusqu'à
    coalescence. On enregistre le couple (temps de coalescence,
    distance initiale entre les deux marcheurs).

    Les répétitions qui n'ont pas coalescé avant T pas sont ignorées.

    Paramètres :
        m : probabilité de se déplacer à chaque pas
        n : taille de la grille (n x n)
        T : nombre de pas maximum avant d'abandonner la répétition
        n_repetitions : nombre de répétitions à tenter

    Retourne :
        temps_coalescence : liste des temps de coalescence observés
        distances : liste des distances initiales correspondantes
    """
    temps_coalescence = []
    distances = []

    for rep in range(n_repetitions):

        # On tire uniformément les positions de départ sur la grille
        x1, y1 = np.random.randint(0, n), np.random.randint(0, n)
        x2, y2 = np.random.randint(0, n), np.random.randint(0, n)

        d0 = distance_euclidienne(x1, y1, x2, y2)

        temps = 0

        # On avance jusqu'à coalescence ou jusqu'à T pas
        while (x1 != x2 or y1 != y2) and temps < T:
            x1, y1 = faire_un_pas(x1, y1, m, n)
            x2, y2 = faire_un_pas(x2, y2, m, n)
            temps += 1

        # On n'enregistre que les répétitions ayant coalescé dans [0, T]
        if x1 == x2 and y1 == y2:
            temps_coalescence.append(temps)
            distances.append(d0)

    return np.array(temps_coalescence), np.array(distances)


def afficher_comparaison_joint(temps_M1, distances_M1, temps_M2, distances_M2,
                                m, n, T, afficher, sauvegarder):
    """
    Affiche les densités jointes (temps, distance) pour M1 et M2
    sous forme d'histogrammes 2D.

    Si les deux modèles sont équivalents, les deux histogrammes doivent
    être visuellement identiques.

    Paramètres :
        temps_M1, distances_M1 : résultats de simuler_M1_joint
        temps_M2, distances_M2 : résultats de simuler_M2_joint
        m, n, T : paramètres de la simulation
        afficher : booléen, affiche les graphiques à l'écran si True
        sauvegarder : booléen, sauvegarde en .png si True
    """
    d_max = max(distances_M1.max(), distances_M2.max())
    bins_t = np.linspace(0, T, 40)
    bins_d = np.linspace(0, d_max, 40)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Équivalence M1 / M2 : densité jointe (temps, distance)\n"
        f"(m={m}, grille {n}×{n}, T={T})",
        fontsize=13
    )

    h1 = axes[0].hist2d(temps_M1, distances_M1, bins=[bins_t, bins_d],
                        density=True, cmap="Blues")
    fig.colorbar(h1[3], ax=axes[0], label="Densité")
    axes[0].set_xlabel("Temps t (pas)")
    axes[0].set_ylabel("Distance euclidienne")
    axes[0].set_title(f"M1 : forward-in-time (n={len(temps_M1)})")

    h2 = axes[1].hist2d(temps_M2, distances_M2, bins=[bins_t, bins_d],
                    density=True, cmap="Greens")
    fig.colorbar(h2[3], ax=axes[1], label="Densité")
    axes[1].set_xlabel("Temps de coalescence (pas)")
    axes[1].set_ylabel("Distance initiale euclidienne")
    axes[1].set_title(f"M2 : backward-in-time (n={len(temps_M2)})")

    plt.tight_layout()

    if sauvegarder:
        plt.savefig("equivalence_M1_M2_joint.png", dpi=150)
        print("  Graphique sauvegardé : equivalence_M1_M2_joint.png")
    if afficher:
        plt.show()


def afficher_comparaison_marginal(temps_M1, distances_M1, temps_M2, distances_M2,
                                   m, n, T, afficher, sauvegarder):
    """
    Affiche les distributions du temps et de la distance pour M1 et M2,
    superposées sur deux histogrammes.

    Si les deux modèles sont équivalents, les courbes doivent
    se superposer.

    Paramètres :
        temps_M1, distances_M1 : résultats de simuler_M1_joint
        temps_M2, distances_M2 : résultats de simuler_M2_joint
        m, n, T : paramètres de la simulation
        afficher : booléen, affiche les graphiques à l'écran si True
        sauvegarder : booléen, sauvegarde en .png si True
    """
    print("- Résultats équivalence M1 / M2 -")
    print(f"  Paramètres : m={m}, grille={n}x{n}, T={T}")
    print(f"  M1 : {len(temps_M1)} répétitions")
    print(f"       temps moyen = {temps_M1.mean():.1f}  |  distance moyenne = {distances_M1.mean():.2f}")
    print(f"  M2 : {len(temps_M2)} répétitions (coalescences dans [0, T])")
    print(f"       temps moyen = {temps_M2.mean():.1f}  |  distance moyenne = {distances_M2.mean():.2f}")

    # Axes communs pour les deux histogrammes
    d_max = max(distances_M1.max(), distances_M2.max())
    bins_t = np.linspace(0, T, 40)
    bins_d = np.linspace(0, d_max, 40)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Équivalence M1 / M2 : distributions du temps et de la distance\n"
        f"(m={m}, grille {n}×{n}, T={T})",
        fontsize=13
    )

    axes[0].hist(temps_M1, bins=bins_t, density=True, alpha=0.6,
                 color="steelblue", edgecolor="white", label=f"M1 (n={len(temps_M1)})")
    axes[0].hist(temps_M2, bins=bins_t, density=True, alpha=0.6,
                 color="#D55E00", edgecolor="white", label=f"M2 (n={len(temps_M2)})")
    axes[0].axvline(temps_M1.mean(), color="steelblue", linewidth=1, linestyle="--", label=f"moy. M1 = {temps_M1.mean():.0f}")
    axes[0].axvline(temps_M1.mean() + temps_M1.std(), color="steelblue", linewidth=0.8, linestyle=":", label=f"± σ M1 = {temps_M1.std():.0f}")
    axes[0].axvline(temps_M1.mean() - temps_M1.std(), color="steelblue", linewidth=0.8, linestyle=":")
    axes[0].axvline(temps_M2.mean(), color="#D55E00", linewidth=1, linestyle="--", label=f"moy. M2 = {temps_M2.mean():.0f}")
    axes[0].axvline(temps_M2.mean() + temps_M2.std(), color="#D55E00", linewidth=0.8, linestyle=":", label=f"± σ M2 = {temps_M2.std():.0f}")
    axes[0].axvline(temps_M2.mean() - temps_M2.std(), color="#D55E00", linewidth=0.8, linestyle=":")
    axes[0].set_xlabel("Temps (pas)")
    axes[0].set_ylabel("Densité")
    axes[0].set_title("Distribution du temps")
    axes[0].legend(fontsize=8, framealpha=0.5)

    axes[1].hist(distances_M1, bins=bins_d, density=True, alpha=0.6,
                 color="steelblue", edgecolor="white", label=f"M1 (n={len(distances_M1)})")
    axes[1].hist(distances_M2, bins=bins_d, density=True, alpha=0.6,
                 color="#D55E00", edgecolor="white", label=f"M2 (n={len(distances_M2)})")
    axes[1].axvline(distances_M1.mean(), color="steelblue", linewidth=1, linestyle="--", label=f"moy. M1 = {distances_M1.mean():.1f}")
    axes[1].axvline(distances_M1.mean() + distances_M1.std(), color="steelblue", linewidth=0.8, linestyle=":", label=f"± σ M1 = {distances_M1.std():.1f}")
    axes[1].axvline(distances_M1.mean() - distances_M1.std(), color="steelblue", linewidth=0.8, linestyle=":")
    axes[1].axvline(distances_M2.mean(), color="#D55E00", linewidth=1, linestyle="--", label=f"moy. M2 = {distances_M2.mean():.1f}")
    axes[1].axvline(distances_M2.mean() + distances_M2.std(), color="#D55E00", linewidth=0.8, linestyle=":", label=f"± σ M2 = {distances_M2.std():.1f}")
    axes[1].axvline(distances_M2.mean() - distances_M2.std(), color="#D55E00", linewidth=0.8, linestyle=":")
    axes[1].set_xlabel("Distance euclidienne")
    axes[1].set_ylabel("Densité")
    axes[1].set_title("Distribution de la distance")
    axes[1].legend(fontsize=8, framealpha=0.5)

    plt.tight_layout()

    if sauvegarder:
        plt.savefig("equivalence_M1_M2_marginal.png", dpi=150)
        print("  Graphique sauvegardé : equivalence_M1_M2_marginal.png")
    if afficher:
        plt.show()

def calculer_poids_t(temps_M1, distances_M1, temps_M2, distances_M2, T, n_bins=40):
    """
    Calcule les poids p(t) à utiliser dans M1 pour que sa distribution jointe
    coïncide avec celle de M2.

    L'idée est :
      - On fixe q(d) comme dans M2 (positions tirées uniformément sur la grille).
      - On veut que p(t, d) = q(t, d), donc :
            p(t) = q(t, d) / p(d | t)
      - p(d | t) s'estime en normalisant chaque colonne t de l'histogramme 2D de M1.
      - q(t, d) s'estime directement depuis l'histogramme 2D de M2.

    En pratique, pour chaque bin (intervalle de classe) de t :
      1. On normalise la colonne correspondante dans l'histogramme M1
        -> on obtient p(d | t) pour chaque bin de d
      2. On lit la valeur dans l'histogramme M2 à la même case
        -> on obtient q(t, d)
      3. On fait le rapport q(t, d) / p(d | t) et on somme sur tous les bins de d
        -> on obtient un poids pour ce t

    Paramètres :
        temps_M1, distances_M1 : résultats de simuler_M1_joint
        temps_M2, distances_M2 : résultats de simuler_M2_joint
        T : temps maximal utilisé dans les simulations
        n_bins : nombre de bins pour les histogrammes 2D (défaut : 40)

    Retourne :
        bins_t_centres : centres des bins de t (pour tirer dedans)
        poids : vecteur de poids normalisé sur t (somme = 1)
    """

    # On définit les axes communs des deux histogrammes 2D
    distance_max = max(distances_M1.max(), distances_M2.max())
    intervalles_t = np.linspace(0, T, n_bins + 1)  #numpy.linspace est une fonction qui génère des points régulièrement espacés entre une valeur de départ
    #  et une valeur de fin. Ici, on génère n_bins + 1 points pour définir n_bins bins (intervalle de classe) pour t.
    intervalles_d = np.linspace(0, distance_max, n_bins + 1) # numpy.histogram2d attend les bords des bins, d'où n_bins + 1 points pour définir n_bins bins

    # On construit les deux histogrammes 2D bruts (counts, pas densité)
    # histo[i, j] = nombre de fois où on a observé la distance i et le temps j
    histo_M1, _, _ = np.histogram2d(distances_M1, temps_M1, bins=[intervalles_d, intervalles_t])
    histo_M2, _, _ = np.histogram2d(distances_M2, temps_M2, bins=[intervalles_d, intervalles_t])

    poids = np.zeros(n_bins) # np.zeros crée un tableau de zéros de taille n_bins, qui va stocker les poids calculés pour chaque bin de t

    for j in range(n_bins):  # pour chaque bin de t

        distances_pour_ce_t_M1 = histo_M1[:, j] 
        total = distances_pour_ce_t_M1.sum()

        if total == 0:
            continue  # pas de données pour ce t, on laisse le poids à 0

        # 1 : normalisation de la colonne -> p(d | t)
        p_d_sachant_t = distances_pour_ce_t_M1 / total

        # 2 : fréquences jointes sous M2 pour ce t -> q(t, d)
        distances_pour_ce_t_M2 = histo_M2[:, j]

        # 3 : rapport q(t, d) / p(d | t), sommé sur d
        # On évite les divisions par zéro avec un masque
        cases_non_nulles = p_d_sachant_t > 0
        poids[j] = (distances_pour_ce_t_M2[cases_non_nulles] / p_d_sachant_t[cases_non_nulles]).sum()

    # On normalise les poids pour en faire une vraie distribution (somme = 1)
    if poids.sum() > 0:
        poids = poids / poids.sum()

    # Les centres des bins de t (pour savoir quelle valeur de t tirer)
    bins_t_centres = 0.5 * (intervalles_t[:-1] + intervalles_t[1:])

    return bins_t_centres, poids


def simuler_M1_joint_pondere(m, n, bins_t_centres, poids, n_repetitions):
    """
    Simule M1 en tirant t selon les poids calculés par calculer_poids_t,
    au lieu de tirer t uniformément.

    Tout le reste est identique à simuler_M1_joint : deux marcheurs partent
    du même point tiré au hasard sur la grille, marchent pendant t pas,
    et on enregistre le couple (t, distance).

    Paramètres :
        m : probabilité de se déplacer à chaque pas
        n : taille de la grille (n x n)
        bins_t_centres : valeurs de t possibles (centres des bins)
        poids : probabilité associée à chaque valeur de t (somme = 1)
        n_repetitions : nombre de répétitions

    Retourne :
        temps : liste des temps tirés
        distances : liste des distances finales correspondantes
    """
    temps = []
    distances = []

    for rep in range(n_repetitions):

        # On tire t selon les poids calculés (plus uniformément)
        t = int(np.random.choice(bins_t_centres, p=poids))

        x0, y0 = np.random.randint(0, n), np.random.randint(0, n)
        x1, y1 = x0, y0
        x2, y2 = x0, y0

        for i in range(t):
            x1, y1 = faire_un_pas(x1, y1, m, n)
            x2, y2 = faire_un_pas(x2, y2, m, n)

        d = distance_euclidienne(x1, y1, x2, y2)
        temps.append(t)
        distances.append(d)

    return np.array(temps), np.array(distances)


# ----------------------------------------------------------------------
# Lecture des arguments en ligne de commande
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Vérification de l'équivalence M1 / M2 via la distribution jointe (distance, temps)"
)

parser.add_argument("--m", type=float, default=0.5,
                    help="Probabilité de se déplacer à chaque pas (défaut : 0.5)")
parser.add_argument("--n", type=int, default=15,
                    help="Taille de la grille n x n (défaut : 15)")
parser.add_argument("--T", type=int, default=2000,
                    help="Temps maximal : t est tiré dans [0, T] pour M1 (défaut : 2000)")
parser.add_argument("--rep", type=int, default=8000,
                    help="Nombre de répétitions (défaut : 8000)")

parser.add_argument("--mode", type=str, default="joint", choices=["joint", "marginal", "pondere"],
                    help="Type de graphique : joint (défaut), marginal, ou pondere (M1 avec poids calculés depuis M2)")

parser.add_argument("--afficher", action="store_true",
                    help="Affiche les graphiques à l'écran")
parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde les graphiques en .png")

args = parser.parse_args()

# ----------------------------------------------------------------------
# Lancement
# ----------------------------------------------------------------------

temps_M1, distances_M1 = simuler_M1_joint(args.m, args.n, args.T, args.rep)
temps_M2, distances_M2 = simuler_M2_joint(args.m, args.n, args.T, args.rep)

print("- Résultats équivalence M1 / M2 -")
print(f"  Paramètres : m={args.m}, grille={args.n}x{args.n}, T={args.T}")
print(f"  M1 : {len(temps_M1)} répétitions")
print(f"       temps moyen = {temps_M1.mean():.1f}  |  distance moyenne = {distances_M1.mean():.2f}")
print(f"  M2 : {len(temps_M2)} répétitions (coalescences dans [0, T])")
print(f"       temps moyen = {temps_M2.mean():.1f}  |  distance moyenne = {distances_M2.mean():.2f}")

if args.mode == "joint":
    afficher_comparaison_joint(temps_M1, distances_M1, temps_M2, distances_M2,
                               args.m, args.n, args.T, args.afficher, args.sauvegarder)
elif args.mode == "marginal":
    afficher_comparaison_marginal(temps_M1, distances_M1, temps_M2, distances_M2,
                                  args.m, args.n, args.T, args.afficher, args.sauvegarder)
elif args.mode == "pondere":
    # On calcule les poids p(t) à partir des deux simulations initiales
    print("\n Calcul des poids p(t)...")
    bins_t_centres, poids = calculer_poids_t(temps_M1, distances_M1, temps_M2, distances_M2, args.T)

    # On refait tourner M1 en tirant t selon ces poids
    print("\n Nouveau round de M1 avec t pondéré...")
    temps_M1_pond, distances_M1_pond = simuler_M1_joint_pondere(
        args.m, args.n, bins_t_centres, poids, args.rep
    )

    print(f"  M1 pondéré : {len(temps_M1_pond)} répétitions")
    print(f"       temps moyen = {temps_M1_pond.mean():.1f}  |  distance moyenne = {distances_M1_pond.mean():.2f}")

    # Histogramme 2D : M1 pondéré vs M2
    afficher_comparaison_joint(temps_M1_pond, distances_M1_pond, temps_M2, distances_M2,
                               args.m, args.n, args.T, args.afficher, args.sauvegarder)

    # Histogrammes 1D : M1 pondéré vs M2
    afficher_comparaison_marginal(temps_M1_pond, distances_M1_pond, temps_M2, distances_M2,
                                  args.m, args.n, args.T, args.afficher, args.sauvegarder)