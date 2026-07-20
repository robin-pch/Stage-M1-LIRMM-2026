# -*- coding: utf-8 -*-
# Genere T evenements Moran et les ecrit dans un fichier csv.
# Rien n'est garde en memoire, chaque evenement est ecrit au fur et a mesure.
# Sert d'entree a generation_donnees_evenements.py.
# Robin Pioch, stage M1 LIRMM, juin 2026

# Usage : python ecrire_evenements.py --l 100 --m 0.5 --sauvegarder
#         python ecrire_evenements.py --l 100 --m 0.5 --sauvegarder --dossier evenements

import numpy as np
import argparse
import csv
import os


def calculer_T(n, m, seuil=0.9999):
    """T tel que `seuil` des coalescences ont eu lieu (loi geo, p=m/n^2), avec une marge de 1.3."""
    p = m / (n * n)
    T = int(np.ceil(np.log(1.0 - seuil) / np.log(1.0 - p)) * 1.3)
    return T


def ecrire_evenements(l, T, m, chemin_csv, seed=None):
    """
    Genere T evenements Moran et les ecrit a l'envers dans chemin_csv.
    Les voisins sont precalcules une fois pour toutes les cases,
    et l'ecriture se fait d'un coup avec numpy (bien plus rapide
    que csv ligne par ligne).
    """
    if seed is not None:
        np.random.seed(seed)

    n = l * l

    # voisins de chaque case, calcules une seule fois
    k_par_case = np.zeros(n, dtype=np.int8)
    voisins_arr = np.full((n, 4), -1, dtype=np.int32)
    for idx in range(n):
        x, y = divmod(idx, l)
        j = 0
        if x > 0:
            voisins_arr[idx, j] = (x - 1) * l + y; j += 1
        if x < l - 1:
            voisins_arr[idx, j] = (x + 1) * l + y; j += 1
        if y > 0:
            voisins_arr[idx, j] = x * l + (y - 1); j += 1
        if y < l - 1:
            voisins_arr[idx, j] = x * l + (y + 1); j += 1
        k_par_case[idx] = j

    tous_evenements = np.empty((T, 4), dtype=np.int16)
    taille_bloc = 1_000_000
    t = 0

    while t < T:
        taille = min(taille_bloc, T - t)
        indices_B = np.random.randint(0, n, size=taille)
        tirages_u = np.random.rand(taille)
        tirages_v = np.random.rand(taille)

        for i in range(taille):
            idxB = indices_B[i]
            k = k_par_case[idxB]
            if not (tirages_u[i] > k * m / 4):
                idxA = voisins_arr[idxB, int(tirages_v[i] * k)]
                xA, yA = divmod(int(idxA), l)
            else:
                xA, yA = divmod(int(idxB), l)
            xB, yB = divmod(int(idxB), l)
            tous_evenements[t + i] = [xA, yA, xB, yB]

        t += taille

    # ecriture a l'envers d'un seul coup
    np.savetxt(
        chemin_csv,
        tous_evenements[::-1],
        fmt="%d",
        delimiter=",",
        header="xA,yA,xB,yB",
        comments=""
    )


parser = argparse.ArgumentParser(
    description="Genere les evenements Moran et les ecrit dans un fichier"
)
parser.add_argument("--l", type=int, default=100, help="Cote de la grille (defaut : 100)")
parser.add_argument("--m", type=float, default=1.0, help="Taux de migration (defaut : 1.0)")
parser.add_argument("--lam", type=float, default=1.0, help="lambda (defaut : 1.0)")
parser.add_argument("--seuil", type=float, default=0.9999,
                    help="Seuil pour calculer T (defaut : 0.9999)")
parser.add_argument("--T", type=int, default=None,
                    help="Nombre de pas. Si absent, calcule automatiquement.")
parser.add_argument("--dossier", type=str, default="evenements",
                    help="Dossier de sortie (defaut : evenements)")
parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde le fichier d'evenements")
parser.add_argument("--seed", type=int, default=None, help="Graine pour reproductibilite")
parser.add_argument("--nom", type=str, default=None,
                    help="Nom du fichier de sortie. Si absent, utilise evenements_l{l}_m{m:.3f}.csv")
args = parser.parse_args()

n = args.l * args.l
T = args.T if args.T is not None else calculer_T(n, args.m, seuil=args.seuil)

print(f"Grille {args.l}x{args.l} | n={n} | m={args.m} | lam={args.lam}")
print(f"T = {T}")

if not args.sauvegarder:
    print("Rien a sauvegarder (utiliser --sauvegarder).")
else:
    os.makedirs(args.dossier, exist_ok=True)
    nom_fichier = args.nom if args.nom is not None else f"evenements_l{args.l}_m{args.m:.3f}.csv"
    nom_csv = os.path.join(args.dossier, nom_fichier)
    print(f"Ecriture de {T} evenements dans {nom_csv}...")
    ecrire_evenements(args.l, T, args.m, nom_csv, seed=args.seed)
    print("Termine.")
