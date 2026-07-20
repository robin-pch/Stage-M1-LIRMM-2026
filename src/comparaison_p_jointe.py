# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Comparaison analytique vs simulation - p(z1,z2,t) = p(z1,z2|t) * p(t)
# ----------------------------------------------------------------------
# Auteur : Robin Pioch
# Stage M1 Bioinformatique, Université de Montpellier
# Encadrant : Stéphane Guindon (LIRMM)
# Juillet 2026
#
# On reprend la decomposition de la densite jointe p(z1,z2,t) en trois
# morceaux, et on trace chacun separement, dans l'ordre de la formule :
#   1) p(z1,z2|t) : distance sachant le temps, pour plusieurs tranches de
#      temps (fonctions reprises de comparaison_analytique.py)
#   2) p(t) : distribution du temps de coalescence seule (fonction et
#      figure reprises de comparaison_pt_selon_m.py)
#   3) p(z1,z2,t) : densite jointe (temps, distance), en seaborn
#      (fonction reprise de comparaison_analytique.py)
#
# Le seul changement par rapport a comparaison_analytique.py est que la
# figure 1 n'est plus mise a l'echelle de la jointe : elle compare une
# quantite conditionnelle (simulation, normalisee a 1 sur la tranche) a
# une quantite conditionnelle analytique (elle aussi normalisee a 1),
# donc plus de probleme d'ecrasement aux grands temps.
#
# Usage :
#   python comparaison_p_jointe.py --l 7 --T 50000 --rep 200 --afficher
#   python comparaison_p_jointe.py --l 7 --m 0.5 --rep 200 --sauvegarder
# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import argparse
import sys


# =============================================================================
# Classe Noeud (identique a comparaison_analytique.py)
# =============================================================================

class Noeud:
    """Noeud de l'arbre de descendance forward (un evenement Moran = deux fils)."""

    def __init__(self, x, y, temps):
        self.x = x
        self.y = y
        self.temps = temps
        self.fils1 = None
        self.fils2 = None
        self.est_feuille = True
        self.desc1 = 0
        self.desc2 = 0
        self.feuilles1 = []
        self.feuilles2 = []


# =============================================================================
# Fonctions utilitaires (identiques a comparaison_analytique.py)
# =============================================================================

def distance(x1, y1, x2, y2):
    """Distance euclidienne entre deux cases."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def generer_evenements(l, T, m):
    """
    Génère exactement T événements Moran.

    Schéma :
      - on choisit uniformément B parmi les n = l*l cases
      - si B a k voisins :
            u ~ U(0,1)
            si !(u > k*m/4) : migration vers un voisin choisi uniformément
            sinon           : A = B (pas de migration)

    Retourne une liste de T tuples ((xA, yA), (xB, yB)).
    """
    n = l * l
    evenements = []

    indices_B = np.random.randint(0, n, size=2 * T)
    tirages_u = np.random.rand(2 * T)

    i = 0

    while len(evenements) < T:

        if i >= len(indices_B):
            indices_B = np.random.randint(0, n, size=T)
            tirages_u = np.random.rand(T)
            i = 0

        xB, yB = divmod(indices_B[i], l)

        voisins = []

        if xB > 0:
            voisins.append((xB - 1, yB))
        if xB < l - 1:
            voisins.append((xB + 1, yB))
        if yB > 0:
            voisins.append((xB, yB - 1))
        if yB < l - 1:
            voisins.append((xB, yB + 1))

        k = len(voisins)
        u = tirages_u[i]
        i += 1

        if not (u > k * m / 4):
            xA, yA = voisins[np.random.randint(k)]
        else:
            xA, yA = xB, yB

        evenements.append(((xA, yA), (xB, yB)))

    return evenements


def calculer_T(n, m):
    """
    Calcule T à partir de la courbe analytique (loi géométrique, p = m/n^2).

    On cherche T tel que 99.99% des coalescences ont eu lieu avant T,
    puis on multiplie par 1.3 pour avoir une petite marge de sécurité.
    """
    p = m / (n * n)
    seuil = 0.9999
    T = int(np.ceil(np.log(1.0 - seuil) / np.log(1.0 - p)) * 1.3)
    return T


def forward_uniforme(l, T, evenements):
    """
    Forward sur toute la grille, sans zone restreinte.
    Retourne (temps_array, distances_array), en pas avant le present.
    """
    grille = [[None] * l for _ in range(l)]
    tous_les_noeuds = []

    for x in range(l):
        for y in range(l):
            noeud = Noeud(x=x, y=y, temps=0)
            grille[x][y] = noeud
            tous_les_noeuds.append(noeud)

    for t in range(T):
        A, B = evenements[t]
        xA, yA = A
        xB, yB = B

        if (xA, yA) == (xB, yB):
            continue

        pere = grille[xB][yB]

        fils_A = Noeud(x=xA, y=yA, temps=t + 1)
        fils_B = Noeud(x=xB, y=yB, temps=t + 1)

        pere.fils1 = fils_A
        pere.fils2 = fils_B
        pere.est_feuille = False
        pere.temps = t + 1

        grille[xA][yA].est_feuille = False
        grille[xA][yA] = fils_A
        grille[xB][yB] = fils_B

        tous_les_noeuds.append(fils_A)
        tous_les_noeuds.append(fils_B)

    for noeud in tous_les_noeuds:
        if noeud.est_feuille:
            noeud.desc1 = 1
            noeud.feuilles1 = [(noeud.x, noeud.y)]

    for i in range(len(tous_les_noeuds) - 1, -1, -1):
        noeud = tous_les_noeuds[i]
        if noeud.fils1 is not None:
            noeud.desc1 = noeud.fils1.desc1 + noeud.fils1.desc2
            noeud.desc2 = noeud.fils2.desc1 + noeud.fils2.desc2
            noeud.feuilles1 = noeud.fils1.feuilles1 + noeud.fils1.feuilles2
            noeud.feuilles2 = noeud.fils2.feuilles1 + noeud.fils2.feuilles2

    temps_liste = []
    distances_liste = []

    for noeud in tous_les_noeuds:
        if noeud.fils1 is not None:
            if noeud.desc1 >= 1 and noeud.desc2 >= 1:
                for (xa, ya) in noeud.feuilles1:
                    for (xb, yb) in noeud.feuilles2:
                        d = distance(xa, ya, xb, yb)
                        temps_liste.append(noeud.temps)
                        distances_liste.append(d)

    temps_array = np.array(temps_liste)
    distances_array = np.array(distances_liste)
    if len(temps_array) > 0:
        temps_array = T - temps_array + 1

    return temps_array, distances_array


# =============================================================================
# Formules analytiques (identiques a comparaison_analytique.py)
# =============================================================================

def proba_transition_exp(xy0, xy1, m, l, t, lam):
    """Probabilite de transition a temps continu."""
    x0r = xy0[0] + 1
    y0r = xy0[1] + 1
    x1r = xy1[0] + 1
    y1r = xy1[1] + 1
    n = l

    X = np.arange(1, n)
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


def proba_transition_exp_cond(z0, z1, m, l, t, lam):
    """Proba conditionnelle : met p(z0,z0,t) a zero et renormalise."""
    if z1 == z0:
        return 0.0
    p_z1 = proba_transition_exp(z0, z1, m, l, t, lam)
    p_z0 = proba_transition_exp(z0, z0, m, l, t, lam)
    denom = 1.0 - p_z0
    if denom <= 0:
        return 0.0
    return p_z1 / denom


def densite_temps_coalescence(t, n, m, lam):
    """
    Densite exponentielle du temps de coalescence en temps calendaire.
    """
    parametre = m * lam / n
    return parametre * np.exp(-parametre * t)


def construire_table_complete(l, m, temps_cal, lam):
    """
    Une ligne par couple (z1, z2) : colonnes x1, y1, x2, y2, d, puis une
    colonne p_cond par temps (p(z1,z2|t)). p_temps contient p(t), une
    valeur par temps (pas par couple).

    temps_cal est le temps de coalescence d'une seule lignee, sans facteur 2.
    proba_transition_exp attend le temps de diffusion cumule des deux
    lignees, donc on lui passe 2*t. densite_temps_coalescence recoit t
    directement, sans ce facteur.
    """
    lignes = []
    for x1 in range(l):
        for y1 in range(l):
            for x2 in range(l):
                for y2 in range(l):
                    d = distance(x1, y1, x2, y2)
                    p_cond_par_temps = [
                        proba_transition_exp((x1, y1), (x2, y2), m, l, 2 * t, lam) / (l * l)
                        for t in temps_cal
                    ]
                    lignes.append([x1, y1, x2, y2, d] + p_cond_par_temps)

    table = np.array(lignes)
    p_temps = np.array([
        densite_temps_coalescence(t, l * l, m, lam) for t in temps_cal
    ])
    return table, p_temps


def construire_table_complete_cond(l, m, temps_cal, lam):
    """
    Pareil que construire_table_complete mais avec proba_transition_exp_cond.
    """
    lignes = []
    for x1 in range(l):
        for y1 in range(l):
            for x2 in range(l):
                for y2 in range(l):
                    d = distance(x1, y1, x2, y2)
                    p_cond_par_temps = [
                        proba_transition_exp_cond((x1, y1), (x2, y2), m, l, 2 * t, lam) / (l * l)
                        for t in temps_cal
                    ]
                    lignes.append([x1, y1, x2, y2, d] + p_cond_par_temps)

    table = np.array(lignes)
    p_temps = np.array([
        densite_temps_coalescence(t, l * l, m, lam) for t in temps_cal
    ])
    return table, p_temps


def table_vers_Z(table, poids_par_temps, n_temps):
    """
    Reconstruit la matrice compacte Z (pour les figures) a partir de la
    table complete : Z[i,j] = somme sur les couples a distance d_vals[j]
    de p_cond_colonne_i * poids_par_temps[i].

    Si poids_par_temps = p_temps, Z est la densite jointe p(z1,z2,t).
    Si poids_par_temps = un tableau de 1, Z est juste p(z1,z2|t), la
    partie conditionnelle (chaque ligne de Z somme alors a peu pres a 1).
    """
    d_col = table[:, 4]
    distances_possibles = np.array(sorted(set(np.round(d_col, 8))))
    index_d = {d: j for j, d in enumerate(distances_possibles)}

    Z = np.zeros((n_temps, len(distances_possibles)))
    for ligne in table:
        d = round(ligne[4], 8)
        j = index_d[d]
        for i in range(n_temps):
            p_cond = ligne[5 + i]
            Z[i, j] += p_cond * poids_par_temps[i]

    return distances_possibles, Z


def tirer_distances_analytique(d_vals, p_vals, n_tirages):
    """
    Tire n_tirages distances selon la distribution p_vals sur d_vals.
    Sert a faire un KDE comparable a la simulation.
    """
    p_vals = np.clip(p_vals, 0, None)
    total = p_vals.sum()
    if total <= 0:
        return np.array([])
    probas = p_vals / total
    indices = np.random.choice(len(d_vals), size=n_tirages, p=probas)
    return d_vals[indices]


def preparer_tirages(t_fwd, d_fwd, d_vals, Z_brute, Z_cond, p_temps,
                     bins_t, centres_t, n_tirages, l, m, lam):
    """
    Tire des distances analytiques (brute et cond) pour chaque tranche de
    temps, pour la figure de densite jointe. Le nombre total de tirages
    (n_tirages) est reparti entre les tranches selon p_temps.
    """
    n = l * l
    t_fwd_cal = t_fwd / (n * lam)

    t_tire_brute = []
    d_tire_brute = []
    t_tire_cond = []
    d_tire_cond = []

    poids_temps = p_temps / p_temps.sum()
    n_tirer_par_tranche = np.round(poids_temps * n_tirages).astype(int)

    for i in range(len(centres_t)):
        n_tirer = n_tirer_par_tranche[i]
        if n_tirer == 0:
            continue

        d_b = tirer_distances_analytique(d_vals, Z_brute[i].copy(), n_tirer)
        d_c = tirer_distances_analytique(d_vals, Z_cond[i].copy(), n_tirer)

        t_b = np.random.uniform(bins_t[i], bins_t[i + 1], size=len(d_b))
        t_c = np.random.uniform(bins_t[i], bins_t[i + 1], size=len(d_c))

        t_tire_brute.extend(t_b)
        d_tire_brute.extend(d_b)
        t_tire_cond.extend(t_c)
        d_tire_cond.extend(d_c)

    return (
        t_fwd_cal,
        np.array(t_tire_brute), np.array(d_tire_brute),
        np.array(t_tire_cond),  np.array(d_tire_cond),
    )


# =============================================================================
# Figure 1 : p(z1,z2|t), distance sachant le temps
# =============================================================================

def figure_distance_sachant_temps(t_fwd, d_fwd, d_vals, Zcond_brute, Zcond_cond,
                                  bins_t, centres_t, bins_retenus,
                                  l, T, m, lam, afficher, sauvegarder):
    """
    P(distance | t) pour plusieurs tranches de temps, courbes discretes.

    Zcond_brute et Zcond_cond sont deja purement conditionnelles (chaque
    ligne somme a peu pres a 1), donc pas besoin de les remettre a
    l'echelle de la simulation : on normalise juste la simulation a 1
    sur sa tranche, et on compare directement.
    """
    fig, axes = plt.subplots(1, len(bins_retenus),
                             figsize=(4 * len(bins_retenus), 4),
                             sharey=False)
    if len(bins_retenus) == 1:
        axes = [axes]
    fig.suptitle(
        f"p(z1,z2|t) : distance sachant le temps - "
        f"grille {l}x{l}, m={m}, lam={lam}",
        fontsize=10
    )

    n = l * l
    t_fwd_cal = t_fwd / (n * lam)

    for k, i in enumerate(bins_retenus):
        ax = axes[k]
        t_label = f"t={centres_t[i]:.1f}"

        p_brute = Zcond_brute[i].copy()
        p_cond = Zcond_cond[i].copy()

        # distribution de la simulation sur cette tranche, normalisee a 1
        masque_t = (t_fwd_cal >= bins_t[i]) & (t_fwd_cal < bins_t[i + 1])
        d_sim = d_fwd[masque_t]
        p_sim = np.zeros(len(d_vals))
        for val in d_sim:
            val_r = round(val, 8)
            idx_d = np.searchsorted(d_vals, val_r)
            if idx_d < len(d_vals) and d_vals[idx_d] == val_r:
                p_sim[idx_d] += 1
        if p_sim.sum() > 0:
            p_sim = p_sim / p_sim.sum()

        ax.plot(d_vals, p_brute, color="steelblue", lw=2, marker="o",
                markersize=3, label="ana. sans correction")
        ax.plot(d_vals, p_cond, color="forestgreen", lw=2, marker="o",
                markersize=3, label="ana. avec correction")
        ax.plot(d_vals, p_sim, color="darkorange", lw=1.5, ls="--",
                marker="o", markersize=3, label=f"sim (n={masque_t.sum()})")
        ax.set_title(t_label, fontsize=9)
        ax.set_xlabel("Distance")
        if k == 0:
            ax.set_ylabel("p(distance | t)")
        ax.legend(fontsize=7)

    plt.tight_layout()
    if sauvegarder:
        nom = f"p_jointe_l{l}_T{T}_distance_sachant_temps.png"
        plt.savefig(nom, dpi=150)
        print(f"  Graphique sauvegarde : {nom}")
    if afficher:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 2 : p(t), distribution du temps seule (reprise de comparaison_pt_selon_m.py)
# =============================================================================

def figure_p_t(t_fwd_cal, l, m, lam, afficher, sauvegarder):
    """
    Histogramme du temps de coalescence simule, avec la courbe
    densite_temps_coalescence par dessus.
    """
    n = l * l
    t_max_affiche = 2 * n / (m * lam)

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f"p(t) : grille {l}x{l}, m={m}, lam={lam}", fontsize=10)

    ax.hist(t_fwd_cal[t_fwd_cal <= t_max_affiche], bins=60, density=True,
           alpha=0.5, color="darkorange", label=f"simulation (n={len(t_fwd_cal)})")

    t_axe = np.linspace(0.01, t_max_affiche, 300)
    p_theo = densite_temps_coalescence(t_axe, n, m, lam)
    ax.plot(t_axe, p_theo, color="navy", lw=2, label="densite_temps_coalescence")

    ax.set_xlim(0, t_max_affiche)
    ax.set_xlabel("Temps calendaire")
    ax.set_ylabel("Densite")
    ax.legend(fontsize=9)

    plt.tight_layout()
    if sauvegarder:
        nom = f"p_jointe_l{l}_temps.png"
        plt.savefig(nom, dpi=150)
        print(f"  Graphique sauvegarde : {nom}")
    if afficher:
        plt.show()
    plt.close(fig)


# =============================================================================
# Figure 3 : p(z1,z2,t), densite jointe (reprise de comparaison_analytique.py)
# =============================================================================

def figure_densite_jointe(t_fwd, d_fwd, d_vals, Z_brute, Z_cond, p_temps,
                          bins_t, centres_t, n_tirages,
                          l, T, m, lam, afficher, sauvegarder):
    """
    Contours KDE 2D (temps, distance) avec seaborn. Trois panneaux :
    simulation, analytique sans correction, analytique avec correction.
    """
    MAX_SCATTER = 10000

    t_fwd_cal, t_tire_brute, d_tire_brute, t_tire_cond, d_tire_cond = \
        preparer_tirages(t_fwd, d_fwd, d_vals, Z_brute, Z_cond, p_temps,
                         bins_t, centres_t, n_tirages, l, m, lam)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"p(z1,z2,t) : densite jointe (temps, distance) - "
        f"grille {l}x{l}, m={m}, lam={lam}",
        fontsize=10
    )

    donnees = [
        (t_fwd_cal,    d_fwd,        "Oranges", f"Forward simulation (n={len(t_fwd)} paires)"),
        (t_tire_brute, d_tire_brute, "Blues",   "Analytique sans correction"),
        (t_tire_cond,  d_tire_cond,  "Greens",  "Analytique avec correction"),
    ]

    for ax, (dt, dd, cmap, titre) in zip(axes, donnees):
        if len(dt) > 1:
            if len(dt) > MAX_SCATTER:
                idx = np.random.choice(len(dt), size=MAX_SCATTER, replace=False)
                dt_plot = dt[idx]
                dd_plot = dd[idx]
            else:
                dt_plot = dt
                dd_plot = dd
            sns.kdeplot(x=dt_plot, y=dd_plot, ax=ax, cmap=cmap, fill=False, thresh=0.05)
        ax.set_title(titre, fontsize=9)
        ax.set_xlabel("Temps calendaire (t1+t2)")
        ax.set_ylabel("Distance")

    plt.tight_layout()
    if sauvegarder:
        nom = f"p_jointe_l{l}_T{T}_densite.png"
        plt.savefig(nom, dpi=150)
        print(f"  Graphique sauvegarde : {nom}")
    if afficher:
        plt.show()
    plt.close(fig)


# =============================================================================
# Lecture des arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="p(z1,z2,t) = p(z1,z2|t) * p(t) : trois figures dans l'ordre de la formule"
)

parser.add_argument("--l", type=int, default=7,
                    help="Cote de la grille l x l (defaut : 7)")

parser.add_argument("--T", type=int, default=None,
                    help="Nombre de pas. Si absent, calcule automatiquement.")

parser.add_argument("--m", type=float, default=1.0,
                    help="Taux de migration (defaut : 1.0)")

parser.add_argument("--lam", type=float, default=1.0,
                    help="lambda = 1/temps de generation (defaut : 1.0)")

parser.add_argument("--rep", type=int, default=200,
                    help="Nombre de repetitions (defaut : 200)")

parser.add_argument("--n_tirages", type=int, default=2000,
                    help="Nb de valeurs tirees dans l'analytique pour le KDE (defaut : 2000)")

parser.add_argument("--n_temps", type=int, default=50,
                    help="Nb de tranches de temps pour le tableau analytique (defaut : 50)")

parser.add_argument("--afficher", action="store_true",
                    help="Affiche les graphiques")

parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde les graphiques en .png")

parser.add_argument("--seed", type=int, default=None,
                    help="Graine pour reproductibilite")

args = parser.parse_args()


# =============================================================================
# Lancement
# =============================================================================

n = args.l * args.l

if args.T is not None:
    T = args.T
else:
    T = calculer_T(n, args.m)
    print(f"T calcule automatiquement : T = {T}")

print(f"Grille {args.l}x{args.l} | n={n} | T={T} | m={args.m} | lam={args.lam}")
print(f"Simulation forward (T={T}, rep={args.rep})...")

if args.seed is not None:
    np.random.seed(args.seed)
    print(f"Graine fixee : {args.seed}")

t_fwd_total = []
d_fwd_total = []

for rep in range(args.rep):
    print(f"  repetition {rep + 1}/{args.rep}...", end="\r", flush=True)
    evenements = generer_evenements(args.l, T, args.m)
    t_rep, d_rep = forward_uniforme(args.l, T, evenements)
    t_fwd_total.extend(t_rep)
    d_fwd_total.extend(d_rep)

t_fwd = np.array(t_fwd_total)
d_fwd = np.array(d_fwd_total)

print(f"\nSimulation terminee : {len(t_fwd)} paires au total")

if len(t_fwd) == 0:
    print("Aucune paire. Augmenter T ou rep.")
    sys.exit(0)

if not args.afficher and not args.sauvegarder:
    print("Rien a afficher (utiliser --afficher ou --sauvegarder).")
    sys.exit(0)

t_fwd_cal = t_fwd / (n * args.lam)

# tranches de temps pour le tableau analytique
temps_moran = np.array([T * k / args.n_temps for k in range(1, args.n_temps + 1)])
temps_cal = temps_moran / (n * args.lam)
bins_t = np.linspace(0, temps_cal.max(), args.n_temps + 1)
centres_t = (bins_t[:-1] + bins_t[1:]) / 2
n_temps_reel = len(centres_t)

print("Calcul de la table complete (sans correction)...")
table_brute, p_temps = construire_table_complete(args.l, args.m, centres_t, args.lam)
print("Calcul de la table complete (avec correction)...")
table_cond, _ = construire_table_complete_cond(args.l, args.m, centres_t, args.lam)

# version jointe p(z1,z2,t), pour la figure 3
d_vals, Z_brute = table_vers_Z(table_brute, p_temps, n_temps_reel)
_, Z_cond = table_vers_Z(table_cond, p_temps, n_temps_reel)

# version purement conditionnelle p(z1,z2|t), pour la figure 1
# (poids de 1 au lieu de p_temps : chaque ligne somme alors a 1)
poids_uns = np.ones(n_temps_reel)
_, Zcond_brute = table_vers_Z(table_brute, poids_uns, n_temps_reel)
_, Zcond_cond = table_vers_Z(table_cond, poids_uns, n_temps_reel)

# tranches avec assez de paires en simulation, au plus 4
bins_retenus = []
for i in range(len(centres_t)):
    masque = (t_fwd_cal >= bins_t[i]) & (t_fwd_cal < bins_t[i + 1])
    if masque.sum() >= 100:
        bins_retenus.append(i)
if len(bins_retenus) > 4:
    pas = len(bins_retenus) // 4
    bins_retenus = bins_retenus[::pas][:4]

# les trois figures, dans l'ordre de la formule p(z1,z2,t) = p(z1,z2|t) * p(t)

if len(bins_retenus) > 0:
    figure_distance_sachant_temps(t_fwd, d_fwd, d_vals, Zcond_brute, Zcond_cond,
                                  bins_t, centres_t, bins_retenus,
                                  args.l, T, args.m, args.lam,
                                  args.afficher, args.sauvegarder)
else:
    print("Pas assez de paires par tranche pour la figure 1, augmenter rep.")

figure_p_t(t_fwd_cal, args.l, args.m, args.lam, args.afficher, args.sauvegarder)

figure_densite_jointe(t_fwd, d_fwd, d_vals, Z_brute, Z_cond, p_temps,
                      bins_t, centres_t, args.n_tirages,
                      args.l, T, args.m, args.lam,
                      args.afficher, args.sauvegarder)
