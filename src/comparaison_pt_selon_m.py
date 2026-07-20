# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------
# Comparaison p(t) : densite_temps_coalescence vs simulation, en faisant
# varier m
# ----------------------------------------------------------------------
# Auteur : Robin Pioch
# Stage M1 Bioinformatique, Universite de Montpellier
# Encadrant : Stephane Guindon (LIRMM)
# Juin 2026
#
# Un seul objectif : pour plusieurs valeurs de m, comparer l'histogramme
# du temps de coalescence simule (toutes distances confondues) a la
# densite theorique densite_temps_coalescence. Un sous-graphique par m,
# tous avec le meme T (donc directement comparables entre eux).
#
# Usage :
#   python comparaison_pt_selon_m.py --l 7 --T 57491 --rep 100
#   python comparaison_pt_selon_m.py --l 7 --T 57491 --rep 100 --m_liste 1.0 0.5 0.1 0.05
# ----------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import argparse


# =============================================================================
# Classe Noeud (identique a moran_spatialise.py)
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
# Simulation (identique a comparaison_analytique.py)
# =============================================================================

def generer_evenements(l, T, m):
    """
    Genere T evenements Moran. A chaque evenement, B est choisi uniformement,
    puis A est soit B lui-meme (proba 1 - k*m/n), soit un voisin de B tire
    au hasard (proba k*m/n, k = nombre de voisins de B).
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


def forward_uniforme(l, T, evenements):
    """
    Simulation forward sur toute la grille. Retourne pour chaque paire de
    lignees son temps de coalescence (en pas Moran avant le present).
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

    for i in range(len(tous_les_noeuds) - 1, -1, -1):
        noeud = tous_les_noeuds[i]
        if noeud.fils1 is not None:
            noeud.desc1 = noeud.fils1.desc1 + noeud.fils1.desc2
            noeud.desc2 = noeud.fils2.desc1 + noeud.fils2.desc2

    temps_liste = []
    for noeud in tous_les_noeuds:
        if noeud.fils1 is not None:
            if noeud.desc1 >= 1 and noeud.desc2 >= 1:
                # nb de paires qui coalescent exactement a ce noeud
                nb_paires = noeud.desc1 * noeud.desc2
                temps_liste.extend([noeud.temps] * nb_paires)

    temps_array = np.array(temps_liste)
    if len(temps_array) > 0:
        temps_array = T - temps_array + 1

    return temps_array


# =============================================================================
# Formule theorique (identique a comparaison_analytique.py)
# =============================================================================

def densite_temps_coalescence(t, n, m, lam):
    """
    Densite exponentielle du temps de coalescence (formule corrigee par
    Stephane, mail du 02/07). La conversion cycles -> temps calendaire ne
    depend pas de m, mais la densite si : la proba de coalescer a un cycle
    donne est m/n^2, donc m revient dans le parametre de l'exponentielle.
    """
    parametre = m * lam / n
    return parametre * np.exp(-parametre * t)


# =============================================================================
# Lecture des arguments
# =============================================================================

parser = argparse.ArgumentParser(
    description="Comparaison p(t) : densite_temps_coalescence vs simulation, selon m"
)

parser.add_argument("--l", type=int, default=7,
                    help="Cote de la grille l x l (defaut : 7)")

parser.add_argument("--T", type=int, default=60000,
                    help="Nombre de pas de Moran, le meme pour tous les m (defaut : 60000)")

parser.add_argument("--lam", type=float, default=1.0,
                    help="lambda = 1/temps de generation (defaut : 1.0)")

parser.add_argument("--rep", type=int, default=100,
                    help="Nombre de repetitions par m (defaut : 100)")

parser.add_argument("--m_liste", type=float, nargs="+", default=[1.0, 0.5, 0.1, 0.05],
                    help="Liste des m a tester (defaut : 1.0 0.5 0.1 0.05)")

parser.add_argument("--t_max_affiche", type=float, default=None,
                    help="Largeur de l'axe X (temps calendaire), identique pour tous les sous-graphiques. Si absent, calcule automatiquement (2 fois n/lam, cf. nouvelle formule de densite_temps_coalescence)")

parser.add_argument("--sauvegarder", action="store_true",
                    help="Sauvegarde le graphique en .png")

parser.add_argument("--seed", type=int, default=None,
                    help="Graine pour reproductibilite")

args = parser.parse_args()


# =============================================================================
# Lancement
# =============================================================================

n = args.l * args.l
T = args.T

if args.t_max_affiche is not None:
    t_max_affiche = args.t_max_affiche
else:
    # 2 fois la moyenne theorique (n / (m*lam)), pour s'adapter automatiquement
    # a la taille de grille et a m : sinon un axe fixe en dur coupe la courbe
    # trop court pour les grandes grilles ou les petits m (distribution plus
    # etalee, cf. densite_temps_coalescence corrigee du 02/07). On prend le
    # plus petit m de la liste car c'est lui qui donne la moyenne la plus
    # grande, donc l'axe le plus large.
    m_min = min(args.m_liste)
    t_max_affiche = 2 * n / (m_min * args.lam)

print(f"Grille {args.l}x{args.l} | n={n} | T={T} (fixe pour tous les m) | lam={args.lam}")
print(f"Axe X (temps) fixe a {t_max_affiche:.1f} pour tous les sous-graphiques")

if args.seed is not None:
    np.random.seed(args.seed)
    print(f"Graine fixee : {args.seed}")

fig, axes = plt.subplots(1, len(args.m_liste), figsize=(5 * len(args.m_liste), 4.5))
if len(args.m_liste) == 1:
    axes = [axes]

for ax, m in zip(axes, args.m_liste):
    print(f"\nSimulation pour m={m} (T={T}, rep={args.rep})...")

    t_total = []
    for rep in range(args.rep):
        print(f"  repetition {rep + 1}/{args.rep}...", end="\r", flush=True)
        evenements = generer_evenements(args.l, T, m)
        t_rep = forward_uniforme(args.l, T, evenements)
        t_total.extend(t_rep)

    t_moran = np.array(t_total)
    print(f"\n  {len(t_moran)} paires au total")

    # conversion en temps calendaire (sans facteur 2 ni m, cf. correction du 26/06 puis 02/07)
    t_cal = t_moran / (n * args.lam)

    # axe X fixe, la meme valeur pour tous les sous-graphiques (meme l,
    # meme T, peu importe m) : comme ca on voit vraiment si la courbe
    # change de forme avec m, sans effet de zoom different a chaque fois
    ax.hist(t_cal[t_cal <= t_max_affiche], bins=60, density=True, alpha=0.5,
            color="darkorange", label="simulation")

    t_axe = np.linspace(0.01, t_max_affiche, 300)
    p_theo = densite_temps_coalescence(t_axe, n, m, args.lam)
    ax.plot(t_axe, p_theo, color="navy", lw=2, label="densite_temps_coalescence")

    ax.set_xlim(0, t_max_affiche)
    ax.set_title(f"m={m}")
    ax.set_xlabel("Temps calendaire")
    if ax is axes[0]:
        ax.set_ylabel("Densite")
    ax.legend(fontsize=8)

fig.suptitle(f"p(t) theorique vs simulation, selon m - grille {args.l}x{args.l}, T={T}", fontsize=11)
plt.tight_layout()

if args.sauvegarder:
    nom = f"comparaison_pt_selon_m_l{args.l}_T{T}.png"
    plt.savefig(nom, dpi=150)
    print(f"\nGraphique sauvegarde : {nom}")

plt.show()