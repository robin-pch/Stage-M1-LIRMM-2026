# ----------------------------------------------------------------------
# Marche aléatoire sur grille 2D Modèles M1 et M2
# ----------------------------------------------------------------------
# Auteur : Robin Pioch
# Contexte : Stage M1 Bioinformatique : Université de Montpellier
#            Sous la direction de Stéphane Guindon (LIRMM)
# Date : Avril 2026
#
# Description :
#   Ce script implémente deux modèles de marche aléatoire sur une grille
#   carrée de côté n (bords réfléchissants).
#
#   M1 : forward-in-time :
#     Deux marcheurs partent du même point et se déplacent indépendamment
#     pendant t pas. On mesure la distance entre eux à la fin.
#
#   M2 : backward-in-time (coalescence) :
#     Deux marcheurs partent de points séparés par une distance d0.
#     On stoppe la simulation quand ils se retrouvent au même noeud.
#     On enregistre le temps écoulé jusqu'à cette coalescence.
#
# Usage :
#   python marche_aleatoire.py --modele M1 --m 0.5 --n 100 --t 500 --rep 200
#   python marche_aleatoire.py --modele M2 --m 0.5 --n 100 --d0 10 --rep 200
#
#   Options supplémentaires :
#     --afficher      affiche le graphique à l'écran
#     --sauvegarder   sauvegarde le graphique en .png
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


def simuler_M1(m, n, t, n_repetitions):
    """
    Simule le modèle M1 sur plusieurs répétitions.

    Deux marcheurs partent du même point.
    On les fait avancer pendant t pas de temps, indépendamment l'un de l'autre.
    On enregistre la distance euclidienne entre eux à la fin de chaque répétition.

    Paramètres :
        m : probabilité de se déplacer à chaque pas
        n : taille de la grille (n x n)
        t : nombre de pas de temps
        n_repetitions : nombre de répétitions de la simulation

    Retourne :
        distances : liste des distances finales (une par répétition)
    """
    distances = []

    for rep in range(n_repetitions):

        # Les deux marcheurs partent du même point tiré au hasard sur la grille
        x0, y0 = np.random.randint(0, n), np.random.randint(0, n)
        x1, y1 = x0, y0
        x2, y2 = x0, y0

        # On fait avancer les deux marcheurs pendant t pas
        for i in range(t):
            x1, y1 = faire_un_pas(x1, y1, m, n)
            x2, y2 = faire_un_pas(x2, y2, m, n)

        # On enregistre la distance finale entre les deux marcheurs
        d = distance_euclidienne(x1, y1, x2, y2)
        distances.append(d)

    return distances


def simuler_M2(m, n, d0, n_repetitions):
    """
    Simule le modèle M2 (coalescence) sur plusieurs répétitions.

    Deux marcheurs partent de deux points séparés par une distance d0.
    Le premier marcheur est placé au centre, le second est décalé de d0
    cases vers la droite. À chaque pas, les deux marcheurs se déplacent
    indépendamment. On stoppe quand ils sont sur la même case, et on
    enregistre le temps écoulé (temps de coalescence).

    Paramètres :
        m : probabilité de se déplacer à chaque pas
        n : taille de la grille (n x n)
        d0 : distance de départ entre les deux marcheurs
        n_repetitions : nombre de répétitions de la simulation

    Retourne :
        temps_coalescence : liste des temps de coalescence (un par répétition)
    """
    temps_coalescence = []

    for rep in range(n_repetitions):

        # La distance d0 est appliquée horizontalement à partir du centre de la grille
        x1, y1 = max(n // 2 - d0 // 2, 0), n // 2
        x2, y2 = min(n // 2 - d0 // 2 + d0, n - 1), n // 2

        temps = 0

        # On avance jusqu'a ce que les deux marcheurs soient au meme noeud
        while x1 != x2 or y1 != y2:
            x1, y1 = faire_un_pas(x1, y1, m, n)
            x2, y2 = faire_un_pas(x2, y2, m, n)
            temps += 1

        temps_coalescence.append(temps)

    return temps_coalescence


def afficher_resultats_M1(distances, m, n, t, afficher, sauvegarder):
    """
    Affiche les statistiques et le graphique des résultats du modèle M1.

    Paramètres :
        distances : liste des distances finales
        m, n, t : paramètres de la simulation
        afficher : booléen, affiche le graphique à l'écran si True
        sauvegarder : booléen, sauvegarde le graphique en .png si True
    """
    distances = np.array(distances)

    print("- Résultats M1 -")
    print(f"  Paramètres : m={m}, grille={n}x{n}, t={t} pas")
    print(f"  Nombre de répétitions : {len(distances)}")
    print(f"  Distance moyenne      : {distances.mean():.2f}")
    print(f"  Écart-type            : {distances.std():.2f}")
    print(f"  Distance min / max    : {distances.min():.2f} / {distances.max():.2f}")

    plt.figure(figsize=(8, 5))
    plt.hist(distances, bins=30, color="steelblue", edgecolor="white")
    plt.xlabel("Distance euclidienne entre les deux marcheurs")
    plt.ylabel("Nombre de répétitions")
    plt.title(f"M1 : Distribution des distances après t={t} pas\n(m={m}, grille {n}x{n})")
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.axvline(distances.mean(), color="navy", linewidth=1, linestyle="--", label=f"moyenne = {distances.mean():.1f}")
    plt.axvline(distances.mean() + distances.std(), color="navy", linewidth=0.8, linestyle=":", label=f"± écart-type")
    plt.axvline(distances.mean() - distances.std(), color="navy", linewidth=0.8, linestyle=":")
    plt.legend(fontsize=8, framealpha=0.5)
    plt.tight_layout()

    if sauvegarder:
        plt.savefig("resultats_M1.png", dpi=150)
        print("  Graphique sauvegardé : resultats_M1.png")
    if afficher:
        plt.show()


def afficher_resultats_M2(temps, m, n, d0, afficher, sauvegarder):
    """
    Affiche les statistiques et le graphique des résultats du modèle M2.

    Paramètres :
        temps : liste des temps de coalescence
        m, n, d0 : paramètres de la simulation
        afficher : booléen, affiche le graphique à l'écran si True
        sauvegarder : booléen, sauvegarde le graphique en .png si True
    """
    temps = np.array(temps)

    print("- Résultats M2 -")
    print(f"  Paramètres : m={m}, grille={n}x{n}, d0={d0}")
    print(f"  Nombre de répétitions : {len(temps)}")
    print(f"  Temps moyen de coalescence : {temps.mean():.2f} pas")
    print(f"  Écart-type                 : {temps.std():.2f}")
    print(f"  Temps min / max            : {temps.min()} / {temps.max()}")

    plt.figure(figsize=(8, 5))
    plt.hist(temps, bins=30, color="coral", edgecolor="white")
    plt.xlabel("Temps de coalescence (nombre de pas)")
    plt.ylabel("Nombre de répétitions")
    plt.title(f"M2 : Distribution des temps de coalescence\n(m={m}, grille {n}x{n}, d0={d0})")
    plt.axvline(temps.mean(), color="darkred", linewidth=1, linestyle="--", label=f"moyenne = {temps.mean():.0f}")
    plt.axvline(temps.mean() + temps.std(), color="darkred", linewidth=0.8, linestyle=":", label=f"± écart-type")
    plt.axvline(temps.mean() - temps.std(), color="darkred", linewidth=0.8, linestyle=":")
    plt.legend(fontsize=8, framealpha=0.5)
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.tight_layout()

    if sauvegarder:
        plt.savefig("resultats_M2.png", dpi=150)
        print("  Graphique sauvegardé : resultats_M2.png")
    if afficher:
        plt.show()


# ----------------------------------------------------------------------
# Lecture des arguments en ligne de commande
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Simulation de marche aléatoire sur grille 2D (M1 ou M2)"
)

parser.add_argument("--modele", type=str, required=True, choices=["M1", "M2"],
                    help="Modèle à simuler : M1 (forward) ou M2 (coalescence)")
parser.add_argument("--m", type=float, default=0.5,
                    help="Probabilité de se déplacer à chaque pas (défaut : 0.5)")
parser.add_argument("--n", type=int, default=100,
                    help="Taille de la grille n x n (défaut : 100)")
parser.add_argument("--rep", type=int, default=200,
                    help="Nombre de répétitions (défaut : 200)")

# Paramètres spécifiques à chaque modèle
parser.add_argument("--t", type=int, default=500,
                    help="[M1] Nombre de pas de temps (défaut : 500)")
parser.add_argument("--d0", type=int, default=10,
                    help="[M2] Distance initiale entre les deux marcheurs (défaut : 10)")

# Options d'affichage
parser.add_argument("--afficher", action="store_true",
                    help="Affiche le graphique à l'écran")
parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde le graphique en .png")

args = parser.parse_args()

# ----------------------------------------------------------------------
# Lancement
# ----------------------------------------------------------------------

if args.modele == "M1":
    distances = simuler_M1(args.m, args.n, args.t, args.rep)
    afficher_resultats_M1(distances, args.m, args.n, args.t, args.afficher, args.sauvegarder)

elif args.modele == "M2":
    temps = simuler_M2(args.m, args.n, args.d0, args.rep)
    afficher_resultats_M2(temps, args.m, args.n, args.d0, args.afficher, args.sauvegarder)