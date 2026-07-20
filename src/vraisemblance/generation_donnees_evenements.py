# -*- coding: utf-8 -*-
# Backward Moran : tire n_echantillons cases au present, puis simule la
# coalescence pour toutes les paires possibles entre ces cases.
# Prend en entree le fichier d'evenements genere par ecrire_evenements.py.
# Robin Pioch, stage M1 LIRMM, juillet 2026

# Usage :
#   python generation_donnees_evenements.py --entree evenements/evenements_tmp.csv
#       --l 20 --m 0.5 --n_echantillons 50 --id_jeu 0 --sauvegarder

import numpy as np
import argparse
import csv
import os


def charger_evenements(chemin_csv):
    """
    Charge les evenements dans un tableau numpy (colonnes xA,yA,xB,yB).
    En tableau numpy plutot qu'en liste de tuples Python, ca prend
    beaucoup moins de memoire, utile vu que les fichiers d'evenements
    peuvent faire plusieurs dizaines de millions de lignes.
    """
    evenements = np.loadtxt(chemin_csv, delimiter=",", skiprows=1, dtype=np.int32)
    return evenements


def backward_paires(evenements, l, lam, echantillon, seed=None):
    """
    Simule la coalescence backward pour toutes les paires de l'echantillon.
    echantillon : liste de cases tirees au present.
    Pour chaque paire, on remonte les evenements a l'envers jusqu'a coalescence.
    t_cal = t_moran / (n * lam).
    """
    if seed is not None:
        np.random.seed(seed)

    n = l * l
    T = len(evenements)
    paires = []

    for i in range(len(echantillon)):
        for j in range(i + 1, len(echantillon)):

            x1, y1 = echantillon[i]
            x2, y2 = echantillon[j]
            z1_depart = (x1, y1)
            z2_depart = (x2, y2)

            coalesce = False

            for k in range(T - 1, -1, -1):
                xA, yA, xB, yB = evenements[k]

                if (xA, yA) == (x1, y1):
                    x1, y1 = xB, yB
                if (xA, yA) == (x2, y2):
                    x2, y2 = xB, yB

                if (x1, y1) == (x2, y2):
                    t_moran = T - k
                    t_cal = t_moran / (n * lam)
                    paires.append((z1_depart, z2_depart, t_cal, 1))
                    coalesce = True
                    break

            if not coalesce:
                paires.append((z1_depart, z2_depart, None, 0))

    return paires


parser = argparse.ArgumentParser(
    description="Backward Moran depuis un fichier d'evenements"
)
parser.add_argument("--entree", type=str, required=True,
                    help="Fichier d'evenements (sortie de ecrire_evenements.py)")
parser.add_argument("--n_echantillons", type=int, default=50,
                    help="Nombre de cases tirees au present (defaut : 50). "
                         "Produit n*(n-1)/2 paires.")
parser.add_argument("--l", type=int, required=True,
                    help="Cote de la grille")
parser.add_argument("--m", type=float, required=True,
                    help="Taux de migration")
parser.add_argument("--lam", type=float, default=1.0,
                    help="lambda (defaut : 1.0)")
parser.add_argument("--id_jeu", type=int, default=0,
                    help="Identifiant du jeu (defaut : 0)")
parser.add_argument("--dossier", type=str, default="donnees",
                    help="Dossier de sortie (defaut : donnees)")
parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde le csv de sortie")
parser.add_argument("--seed", type=int, default=None,
                    help="Graine pour reproductibilite")
args = parser.parse_args()

n = args.l * args.l
n_paires = args.n_echantillons * (args.n_echantillons - 1) // 2
print(f"Grille {args.l}x{args.l} | m={args.m} | lam={args.lam}")
print(f"{args.n_echantillons} echantillons -> {n_paires} paires")

print(f"Chargement des evenements depuis {args.entree}...")
evenements = charger_evenements(args.entree)
print(f"{len(evenements)} evenements charges.")

# tirage des cases au present (sans remise)
if args.seed is not None:
    np.random.seed(args.seed)
indices = np.random.choice(n, size=args.n_echantillons, replace=False)
echantillon = [divmod(int(idx), args.l) for idx in indices]

paires = backward_paires(evenements, args.l, args.lam, echantillon, seed=None)

n_coalesce = sum(1 for (*_, c) in paires if c == 1)
print(f"{len(paires)} paires ({n_coalesce} coalescees, {len(paires)-n_coalesce} non coalescees).")

if not args.sauvegarder:
    print("Rien a sauvegarder (utiliser --sauvegarder).")
else:
    os.makedirs(args.dossier, exist_ok=True)
    nom_csv = os.path.join(args.dossier, f"donnees_jeu{args.id_jeu}.csv")
    with open(nom_csv, "w", newline="") as f:
        ecrivain = csv.writer(f)
        ecrivain.writerow(["id_jeu", "x1", "y1", "x2", "y2", "x1r", "y1r", "x2r", "y2r",
                           "t", "coalesce", "m", "l", "lam"])
        for (z1, z2, t_cal, coalesce) in paires:
            x1, y1 = z1
            x2, y2 = z2
            # coordonnees relatives (entre 0 et 1), utilisees par vraisemblance_ml.py
            # pour ne pas fuiter d'info sur l quand plusieurs l sont testes
            x1r, y1r = x1 / args.l, y1 / args.l
            x2r, y2r = x2 / args.l, y2 / args.l
            t_ecrit = t_cal if coalesce == 1 else ""
            ecrivain.writerow([args.id_jeu, x1, y1, x2, y2, x1r, y1r, x2r, y2r,
                               t_ecrit, coalesce, args.m, args.l, args.lam])
    print(f"CSV exporte : {nom_csv}")
