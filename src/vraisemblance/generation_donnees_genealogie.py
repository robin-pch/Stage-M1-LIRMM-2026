# -*- coding: utf-8 -*-
# Simule directement la genealogie des n_echantillons lignees, sans passer
# par un fichier d'evenements sur toute la population.
# t est deja en temps calendaire, pas de conversion a faire apres coup.
# Robin Pioch, stage M1 LIRMM, juillet 2026

# Usage : python generation_donnees_genealogie.py --l 100 --m 0.3 --n_echantillons 50 --id_jeu 0 --sauvegarder
#         python generation_donnees_genealogie.py --l 100 --m 0.3 --n_echantillons 50 --schema cercle --rayon 20 --sauvegarder

import numpy as np
import argparse
import csv
import os


def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def case_dans_zone(l, x, y, schema, sigma=1.0, rayon=None):
    """
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
    """Liste des cases valides pour le schema choisi, calculee une seule fois."""
    cases = []
    for x in range(l):
        for y in range(l):
            if case_dans_zone(l, x, y, schema, sigma=sigma, rayon=rayon):
                cases.append((x, y))
    return cases


def voisins(x, y, l):
    v = []
    if x > 0:
        v.append((x - 1, y))
    if x < l - 1:
        v.append((x + 1, y))
    if y > 0:
        v.append((x, y - 1))
    if y < l - 1:
        v.append((x, y + 1))
    return v


def simuler_genealogie(l, m, lam, n_echantillons, schema="uniforme", sigma=1.0, rayon=None, seed=None):
    """Renvoie les positions de depart et un dict {(i,j): t} des temps de coalescence."""
    if seed is not None:
        np.random.seed(seed)

    cases_valides = cases_dans_zone(l, schema, sigma=sigma, rayon=rayon)
    indices = np.random.choice(len(cases_valides), size=n_echantillons, replace=False)
    positions_depart = [cases_valides[i] for i in indices]

    lignees = []
    for i in range(n_echantillons):
        lignees.append({"pos": positions_depart[i], "membres": [i]})

    k = n_echantillons
    t = 0.0
    temps_coalescence = {}

    while k > 1:

        t = t + np.random.exponential(1.0 / (k * lam))

        idx_bouge = np.random.randint(k)
        x, y = lignees[idx_bouge]["pos"]
        voisins_valides = voisins(x, y, l)
        k_voisins = len(voisins_valides)

        u = np.random.rand()
        if not (u > k_voisins * m / 4):
            nouvelle_pos = voisins_valides[np.random.randint(k_voisins)]
        else:
            nouvelle_pos = (x, y)

        lignees[idx_bouge]["pos"] = nouvelle_pos

        idx_rencontre = None
        for j in range(k):
            if j != idx_bouge and lignees[j]["pos"] == nouvelle_pos:
                idx_rencontre = j
                break

        if idx_rencontre is not None:
            membres_a = lignees[idx_bouge]["membres"]
            membres_b = lignees[idx_rencontre]["membres"]
            for a in membres_a:
                for b in membres_b:
                    temps_coalescence[(min(a, b), max(a, b))] = t

            lignees[idx_bouge]["membres"] = membres_a + membres_b
            del lignees[idx_rencontre]
            k = k - 1

    return positions_depart, temps_coalescence


parser = argparse.ArgumentParser(
    description="Simule directement la genealogie des echantillons (sans fichier d'evenements)"
)
parser.add_argument("--l", type=int, required=True,
                    help="Cote de la grille")
parser.add_argument("--m", type=float, required=True,
                    help="Taux de migration")
parser.add_argument("--lam", type=float, default=1.0,
                    help="lambda (defaut : 1.0)")
parser.add_argument("--n_echantillons", type=int, default=50,
                    help="Nombre de cases tirees au present (defaut : 50)")
parser.add_argument("--schema", type=str, default="uniforme", choices=["uniforme", "cercle", "diagonale"],
                    help="Schema d'echantillonnage des positions de depart (defaut : uniforme)")
parser.add_argument("--sigma", type=float, default=1.0,
                    help="Largeur de bande pour le schema diagonale (defaut : 1.0)")
parser.add_argument("--rayon", type=float, default=None,
                    help="Rayon pour le schema cercle (defaut : l/4)")
parser.add_argument("--id_jeu", type=int, default=0,
                    help="Identifiant du jeu (defaut : 0)")
parser.add_argument("--dossier", type=str, default="donnees",
                    help="Dossier de sortie (defaut : donnees)")
parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde le csv de sortie")
parser.add_argument("--seed", type=int, default=None,
                    help="Graine pour reproductibilite")
args = parser.parse_args()

print(f"Grille {args.l}x{args.l} | m={args.m} | lam={args.lam} | schema={args.schema}")
print(f"{args.n_echantillons} echantillons -> {args.n_echantillons * (args.n_echantillons - 1) // 2} paires")

positions_depart, temps_coalescence = simuler_genealogie(
    args.l, args.m, args.lam, args.n_echantillons,
    schema=args.schema, sigma=args.sigma, rayon=args.rayon, seed=args.seed
)

print(f"{len(temps_coalescence)} paires coalescees.")

if not args.sauvegarder:
    print("Rien a sauvegarder (utiliser --sauvegarder).")
else:
    os.makedirs(args.dossier, exist_ok=True)
    nom_csv = os.path.join(args.dossier, f"donnees_jeu{args.id_jeu}_n{args.n_echantillons}.csv")
    with open(nom_csv, "w", newline="") as f:
        ecrivain = csv.writer(f)
        ecrivain.writerow(["id_jeu", "x1", "y1", "x2", "y2", "x1r", "y1r", "x2r", "y2r", "t", "coalesce", "m", "l", "lam"])
        for i in range(args.n_echantillons):
            for j in range(i + 1, args.n_echantillons):
                x1, y1 = positions_depart[i]
                x2, y2 = positions_depart[j]
                # coordonnees relatives (entre 0 et 1), utilisees par vraisemblance_ml.py
                # pour ne pas fuiter d'info sur l quand plusieurs l sont testes
                x1r, y1r = x1 / args.l, y1 / args.l
                x2r, y2r = x2 / args.l, y2 / args.l
                t_paire = temps_coalescence[(i, j)]
                ecrivain.writerow([args.id_jeu, x1, y1, x2, y2, x1r, y1r, x2r, y2r,
                                   t_paire, 1, args.m, args.l, args.lam])
    print(f"CSV exporte : {nom_csv}")
