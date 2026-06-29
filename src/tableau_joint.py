# -*- coding: utf-8 -*-
# Proba jointe (z1, z2) des deux lignees issues d'un meme ancetre, a comparer
# aux simulations forward.
#
#   P(z1, z2 | t1, t2) = (1/n) p(z1, z2, t1+t2)
#
# p = proba de transition a temps continu, n = l*l. Tout en calendaire.
# Tableau : pour chaque couple (z1,z2), proba jointe a plusieurs temps
# (on coupe T en n_t parts egales).
#
# Usage : python tableau_joint.py --l 7 --T 50 --n_t 5 --m 1.0 --sauvegarder

import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv


def distance(x1, y1, x2, y2):
    """Distance euclidienne entre deux cases."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# --- proba de transition a temps continu (reprise de verif_approximation.py) ---

def proba_transition_exp(xy0, xy1, m, l, t, lam):
    """Probabilite de transition a temps continu."""
    x0r = xy0[0] + 1
    y0r = xy0[1] + 1
    x1r = xy1[0] + 1
    y1r = xy1[1] + 1
    n = l

    X = np.arange(1, n)   # 1 a n-1
    Y = np.arange(1, n)

    terme_const = 1.0 / n**2

    poids_X = np.exp(-(m/2) * lam * t * (1 - np.cos(np.pi * X / n)))
    cos_x0 = np.cos(np.pi * X * (0.5 - x0r) / n)
    cos_x1 = np.cos(np.pi * X * (0.5 - x1r) / n)
    terme_X = 2.0 / n**2 * np.sum(poids_X * cos_x0 * cos_x1)

    poids_Y = np.exp(-(m/2) * lam * t * (1 - np.cos(np.pi * Y / n)))
    cos_y0 = np.cos(np.pi * Y * (0.5 - y0r) / n)
    cos_y1 = np.cos(np.pi * Y * (0.5 - y1r) / n)
    terme_Y = 2.0 / n**2 * np.sum(poids_Y * cos_y0 * cos_y1)

    KX, KY = np.meshgrid(X, Y, indexing="ij")
    poids_XY = np.exp(-(m/2) * lam * t * (2 - np.cos(np.pi * KX / n) - np.cos(np.pi * KY / n)))
    cos_kx0 = np.cos(np.pi * KX * (0.5 - x0r) / n)
    cos_kx1 = np.cos(np.pi * KX * (0.5 - x1r) / n)
    cos_ky0 = np.cos(np.pi * KY * (0.5 - y0r) / n)
    cos_ky1 = np.cos(np.pi * KY * (0.5 - y1r) / n)
    terme_XY = 4.0 / n**2 * np.sum(
        poids_XY * cos_kx0 * cos_kx1 * cos_ky0 * cos_ky1
    )

    return terme_const + terme_X + terme_Y + terme_XY


def proba_jointe(z1, z2, m, l, t, lam):
    """Proba jointe au temps total t = t1+t2 : (1/n) p(z1, z2, t)."""
    n = l * l
    return proba_transition_exp(z1, z2, m, l, t, lam) / n


def construire_tableau(m, l, temps, lam):
    """Proba jointe pour chaque couple (z1,z2) et chaque temps. Renvoie (x1,y1,x2,y2,d,probas)."""
    lignes = []
    for x1 in range(l):
        for y1 in range(l):
            for x2 in range(l):
                for y2 in range(l):
                    d = distance(x1, y1, x2, y2)
                    probas = []
                    for t in temps:
                        p = proba_jointe((x1, y1), (x2, y2), m, l, t, lam)
                        probas.append(p)
                    lignes.append((x1, y1, x2, y2, d, probas))
    return lignes


def distribution_distance(lignes, indice_temps):
    """Moyenne des probas jointes par distance (moyenne, pas somme : sinon biais car certaines distances ont plus de couples)."""
    p_par_dist = {}
    n_par_dist = {}
    for (x1, y1, x2, y2, d, probas) in lignes:
        dr = round(d, 8)
        if dr not in p_par_dist:
            p_par_dist[dr] = 0.0
            n_par_dist[dr] = 0
        p_par_dist[dr] += probas[indice_temps]
        n_par_dist[dr] += 1
    d_vals = np.array(sorted(p_par_dist.keys()))
    p_vals = np.array([p_par_dist[dr] / n_par_dist[dr] for dr in d_vals])
    return d_vals, p_vals


def main():
    parser = argparse.ArgumentParser(
        description="Tableau joint des deux lignees (identite, proba exp)"
    )
    parser.add_argument("--l", type=int, default=7, help="Cote de la grille")
    parser.add_argument("--T", type=float, default=50.0, help="Temps total (calendaire)")
    parser.add_argument("--n_t", type=int, default=5, help="Nombre de temps (coupes de T)")
    parser.add_argument("--m", type=float, default=1.0, help="Taux de migration")
    parser.add_argument("--lam", type=float, default=1.0,
                        help="lam = 1/temps de generation (defaut 1)")
    parser.add_argument("--afficher", action="store_true", help="Affiche la figure")
    parser.add_argument("--sauvegarder", action="store_true", help="Sauvegarde figure et csv")
    args = parser.parse_args()

    l = args.l
    m = args.m
    T = args.T
    lam = args.lam
    n = l * l

    # on coupe T en n_t parties egales : t = T/n_t, 2T/n_t, ..., T
    temps = [T * k / args.n_t for k in range(1, args.n_t + 1)]

    print(f"Grille {l}x{l} | n={n} | m={m} | lam={lam}")
    print(f"Temps (calendaire) : {[round(t, 3) for t in temps]}")

    lignes = construire_tableau(m, l, temps, lam)

    # verif : a chaque temps, la somme des probas jointes doit faire 1
    for j, t in enumerate(temps):
        s = sum(probas[j] for (*_, probas) in lignes)
        print(f"  t={t:.3f} : somme des probas jointes = {s:.6f}")

    # export CSV (format large : une colonne par temps)
    nom_csv = f"tableau_joint_l{l}_T{T:.0f}_m{m:.2f}.csv"
    entete = ["x1", "y1", "x2", "y2", "d"] + [f"t={t:.3f}" for t in temps]
    with open(nom_csv, "w", newline="") as f:
        ecrivain = csv.writer(f)
        ecrivain.writerow(entete)
        for (x1, y1, x2, y2, d, probas) in lignes:
            ligne = [x1, y1, x2, y2, f"{d:.6f}"] + [f"{p:.8e}" for p in probas]
            ecrivain.writerow(ligne)
    print(f"CSV exporte : {nom_csv} ({len(lignes)} lignes)")

    if not args.afficher and not args.sauvegarder:
        return

    # --- figure : distribution de la distance entre les deux lignees, a chaque temps ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(f"P(distance entre les 2 lignees) selon le temps - "
                 f"grille {l}x{l}, m={m}", fontsize=10)
    for j, t in enumerate(temps):
        d_vals, p_vals = distribution_distance(lignes, j)
        ax.plot(d_vals, p_vals, marker="o", markersize=4, label=f"t={t:.2f}")
    ax.set_xlabel("Distance entre les deux lignees")
    ax.set_ylabel("Probabilite moyenne par couple")
    ax.legend(fontsize=8)
    plt.tight_layout()

    if args.sauvegarder:
        nom = f"tableau_joint_l{l}_T{T:.0f}_m{m:.2f}_distance.png"
        plt.savefig(nom, dpi=150)
        print(f"Sauvegarde : {nom}")
    if args.afficher:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
