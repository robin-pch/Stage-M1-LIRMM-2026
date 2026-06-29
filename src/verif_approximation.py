# -*- coding: utf-8 -*-
# Verification de l'approximation tau = T*m/n pour la lignee unique.
#
# Question : peut-on remplacer le melange (nombre ALEATOIRE de deplacements)
# par un seul nombre de deplacements, la moyenne tau = T*m/n ?
# La lignee est touchee T/n fois en moyenne, et migre avec proba m a chaque
# fois, d'ou T/n * m = T*m/n.
#
#   melange      : on moyenne P(arrivee|depart,tau) sur tous les tau possibles
#                  (loi binomiale B(T, m/n))
#   simple       : on prend juste tau = T*m/n (pas d'arrondi, sauf m=1 ou le
#                  mode damier donne des NaN pour un t non entier)
#   exp          : formule a temps continu (loi de Poisson sur le nb de pas)
#
# heatmap_analytique et heatmap_analytique_melange viennent de
# moran_lignee_unique.py. Seul changement : le poids binomial passe par
# scipy (binom.pmf) plutot que comb(...)*p**k, qui sous-deborde a zero
# quand T est grand.
#
# exp et melange devraient etre identiques avec lam=1 et t=tau : c'est ce
# qu'on verifie ici.
#
# Usage :
#   python verif_approximation.py --l 7 --T 5000 --m 1.0 --afficher

import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import binom


def distance(x1, y1, x2, y2):
    """Distance euclidienne entre deux cases."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


# --- fonctions reprises de moran_lignee_unique.py ---

def proba_transition_analytique(xy0, xy1, m, l, t):
    """P(xy1|xy0,m,l,t), marche aleatoire reflechie 2D en t pas (t entier)."""
    x0r = xy0[0] + 1
    y0r = xy0[1] + 1
    x1r = xy1[0] + 1
    y1r = xy1[1] + 1
    n = l

    X = np.arange(1, n + 1)
    Y = np.arange(1, n + 1)

    terme_const = 1.0 / n**2

    poids_X = (1 - m/2 + m/2 * np.cos(np.pi * X / n))**t
    cos_x0 = np.cos(np.pi * X * (0.5 - x0r) / n)
    cos_x1 = np.cos(np.pi * X * (0.5 - x1r) / n)
    terme_X = 2.0 / n**2 * np.sum(poids_X * cos_x0 * cos_x1)

    poids_Y = (1 - m/2 + m/2 * np.cos(np.pi * Y / n))**t
    cos_y0 = np.cos(np.pi * Y * (0.5 - y0r) / n)
    cos_y1 = np.cos(np.pi * Y * (0.5 - y1r) / n)
    terme_Y = 2.0 / n**2 * np.sum(poids_Y * cos_y0 * cos_y1)

    KX, KY = np.meshgrid(X, Y, indexing="ij")
    poids_XY = (1 - m + m/2 * (np.cos(np.pi * KX / n) + np.cos(np.pi * KY / n)))**t
    cos_kx0 = np.cos(np.pi * KX * (0.5 - x0r) / n)
    cos_kx1 = np.cos(np.pi * KX * (0.5 - x1r) / n)
    cos_ky0 = np.cos(np.pi * KY * (0.5 - y0r) / n)
    cos_ky1 = np.cos(np.pi * KY * (0.5 - y1r) / n)
    terme_XY = 4.0 / n**2 * np.sum(
        poids_XY * cos_kx0 * cos_kx1 * cos_ky0 * cos_ky1
    )

    return terme_const + terme_X + terme_Y + terme_XY


def heatmap_analytique(xy0, m, l, t):
    """P(xy1|xy0,m,l,t) pour toutes les cases. Renvoie un tableau l x l."""
    h = np.zeros((l, l))
    for x1 in range(l):
        for y1 in range(l):
            h[x1, y1] = proba_transition_analytique(xy0, (x1, y1), m, l, t)
    return h


# --- version temps continu (exponentielle) ---
# Au lieu d'elever la valeur propre a la puissance r, on ecrit direct la
# formule a temps continu : exp(-(m/2) lam t (1-cos(pi x/n))) pour les termes
# simples, exp(-(m/2) lam t (2-cos(pi x/n)-cos(pi y/n))) pour le terme double.
# r pas correspondent a t = r/lam ; en passant t=r/lam, lam se simplifie et
# on retombe sur le melange binomial.

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


def heatmap_exp(xy0, m, l, t, lam):
    """Version temps continu pour toutes les cases. Renvoie un tableau l x l."""
    h = np.zeros((l, l))
    for x1 in range(l):
        for y1 in range(l):
            h[x1, y1] = proba_transition_exp(xy0, (x1, y1), m, l, t, lam)
    return h


def heatmap_analytique_melange(xy0, m, l, T):
    """Melange analytique : la lignee est affectee un nombre aleatoire de fois."""
    n = l * l
    # a chaque pas de Moran, la lignee est touchee (proba 1/n) ET migre vraiment
    # (proba m) : elle se deplace donc avec proba m/n
    p = m / n

    h = np.zeros((l, l))

    # nombre de deplacements parmi T pas : loi binomiale B(T, m/n)
    poids = binom.pmf(np.arange(T + 1), T, p)

    for k in range(T + 1):
        if poids[k] < 1e-12:
            continue
        h += poids[k] * heatmap_analytique(xy0, m, l, k)

    return h


def proba_melange_transition(xy0, xy1, m, l, T):
    """Melange binomial pour une seule transition (depart -> arrivee). Sert au graphe proba/temps."""
    n = l * l
    p = m / n
    poids = binom.pmf(np.arange(T + 1), T, p)
    proba = 0.0
    for k in range(T + 1):
        if poids[k] < 1e-12:
            continue
        proba += poids[k] * proba_transition_analytique(xy0, xy1, m, l, k)
    return proba


def regrouper_par_distance(h, xy0, l):
    """Somme les probas de la heatmap par distance au depart. Renvoie (d_vals, p_vals)."""
    p_par_dist = {}
    for x1 in range(l):
        for y1 in range(l):
            d = round(distance(xy0[0], xy0[1], x1, y1), 8)
            if d not in p_par_dist:
                p_par_dist[d] = 0.0
            p_par_dist[d] += h[x1, y1]
    d_vals = np.array(sorted(p_par_dist.keys()))
    p_vals = np.array([p_par_dist[d] for d in d_vals])
    return d_vals, p_vals


def main():
    parser = argparse.ArgumentParser(
        description="Verification de l'approximation tau = T*m/n (lignee unique)"
    )
    parser.add_argument("--l", type=int, default=7, help="Cote de la grille")
    parser.add_argument("--T", type=int, default=5000, help="Temps de Moran")
    parser.add_argument("--m", type=float, default=1.0, help="Taux de migration")
    parser.add_argument("--lam", type=float, default=1.0,
                        help="Temps d'une generation en annee (version exp, defaut 1)")
    parser.add_argument("--r_max", type=int, default=40,
                        help="Nombre de pas max pour le graphe proba/temps")
    parser.add_argument("--afficher", action="store_true", help="Affiche la figure")
    parser.add_argument("--sauvegarder", action="store_true", help="Sauvegarde en png")
    args = parser.parse_args()

    l = args.l
    m = args.m
    T = args.T
    lam = args.lam
    n = l * l

    # depart au centre de la grille
    xy0 = (l // 2, l // 2)

    # nombre de deplacements de l'approximation simple : la moyenne T*m/n
    tau_simple = T * m / n

    print(f"Grille {l}x{l} | n={n} | T={T} | m={m} | lam={lam} | depart {xy0}")
    print(f"Approximation simple : tau = T*m/n = {tau_simple:.2f}")
    print()

    # les trois distributions a comparer
    h_melange = heatmap_analytique_melange(xy0, m, l, T)
    h_simple = heatmap_analytique(xy0, m, l, tau_simple)
    # r pas correspondent a t=r/lam, donc on passe t=tau/lam : lam se simplifie
    # et on retombe sur le melange
    h_exp = heatmap_exp(xy0, m, l, tau_simple / lam, lam)

    diff = h_melange - h_simple
    diff_exp = h_melange - h_exp
    print(f"Somme des probas (melange) : {h_melange.sum():.6f}")
    print(f"Somme des probas (simple)  : {h_simple.sum():.6f}")
    print(f"Somme des probas (exp)     : {h_exp.sum():.6f}")
    print(f"Diff. max (simple vs melange) : {np.abs(diff).max():.3e}")
    print(f"Diff. max (exp vs melange)    : {np.abs(diff_exp).max():.3e}")

    d_vals, p_mel = regrouper_par_distance(h_melange, xy0, l)
    _, p_sim = regrouper_par_distance(h_simple, xy0, l)
    _, p_exp = regrouper_par_distance(h_exp, xy0, l)
    print(f"Diff. max P(distance) (simple) : {np.abs(p_mel - p_sim).max():.3e}")
    print(f"Diff. max P(distance) (exp)    : {np.abs(p_mel - p_exp).max():.3e}")

    if not args.afficher and not args.sauvegarder:
        return

    base_nom = f"verif_approx_l{l}_T{T}_m{m:.2f}"

    # --- Figure 1 : les heatmaps des 3 approches + la difference exp - melange ---
    fig1, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig1.suptitle(f"Heatmaps des 3 approches - grille {l}x{l}, T={T}, m={m}",
                  fontsize=10)

    vmax = max(h_melange.max(), h_simple.max(), h_exp.max())

    im = axes[0].imshow(h_melange.T, origin="lower", cmap="Blues", vmin=0, vmax=vmax)
    axes[0].set_title("Melange binomial")
    plt.colorbar(im, ax=axes[0])

    im = axes[1].imshow(h_simple.T, origin="lower", cmap="Blues", vmin=0, vmax=vmax)
    axes[1].set_title(f"Simple (tau={tau_simple:.2f})")
    plt.colorbar(im, ax=axes[1])

    im = axes[2].imshow(h_exp.T, origin="lower", cmap="Blues", vmin=0, vmax=vmax)
    axes[2].set_title("Exponentielle (Poisson)")
    plt.colorbar(im, ax=axes[2])

    vd = np.abs(diff_exp).max()
    im = axes[3].imshow(diff_exp.T, origin="lower", cmap="RdBu", vmin=-vd, vmax=vd)
    axes[3].set_title("Exp - melange")
    plt.colorbar(im, ax=axes[3])

    plt.tight_layout()

    # --- Figure 2 : distribution de la distance au depart, 3 approches ---
    fig2, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(f"P(distance au depart) - grille {l}x{l}, T={T}, m={m}", fontsize=10)
    ax.plot(d_vals, p_mel, marker="o", markersize=5, label="melange")
    ax.plot(d_vals, p_sim, marker="s", markersize=5, alpha=0.7, label="simple")
    ax.plot(d_vals, p_exp, marker="^", markersize=5, alpha=0.7, label="exp")
    ax.set_xlabel("Distance")
    ax.set_ylabel("Probabilite")
    ax.legend()
    plt.tight_layout()

    # --- Figure 3 : proba d'une transition en fonction du temps, 3 approches ---
    # proba de revenir au depart (depart -> depart) en fonction du nombre de pas r
    cible = xy0
    r_vals = np.arange(0, args.r_max + 1)
    p_sim_t = []
    p_mel_t = []
    p_exp_t = []
    for r in r_vals:
        p_sim_t.append(proba_transition_analytique(xy0, cible, m, l, r))
        p_exp_t.append(proba_transition_exp(xy0, cible, m, l, r / lam, lam))
        T_r = int(round(r * n / m))   # temps de Moran correspondant a r pas
        p_mel_t.append(proba_melange_transition(xy0, cible, m, l, T_r))

    fig3, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(f"P(depart -> depart) selon le temps - grille {l}x{l}, m={m}",
                 fontsize=10)
    ax.plot(r_vals, p_sim_t, marker="s", markersize=4, label="simple (discret)")
    ax.plot(r_vals, p_mel_t, marker="o", markersize=4, alpha=0.7, label="melange")
    ax.plot(r_vals, p_exp_t, marker="^", markersize=4, alpha=0.7, label="exp (Poisson)")
    ax.set_xlabel("Nombre de pas r (= temps calendaire si lam=1)")
    ax.set_ylabel("Probabilite")
    ax.legend()
    plt.tight_layout()

    if args.sauvegarder:
        for fig, suff in [(fig1, "heatmaps"), (fig2, "distance"), (fig3, "proba_temps")]:
            nom = f"{base_nom}_{suff}.png"
            fig.savefig(nom, dpi=150)
            print(f"Sauvegarde : {nom}")
    if args.afficher:
        plt.show()
    plt.close(fig1)
    plt.close(fig2)
    plt.close(fig3)


if __name__ == "__main__":
    main()
