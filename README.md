# Stage M1 Bioinformatique - Robin Pioch

**LIRMM, Université de Montpellier**
Encadrant : Stéphane Guindon
2026

## Contexte

Ce dépôt regroupe les scripts développés durant le stage, portant sur l'étude in silico
de modèles de génétique spatiale.

## Structure

```
src/                    --> scripts Python
src/vraisemblance/      --> génération de données et inférence du taux de migration
docs/                   --> rapport bibliographique et journal de stage
results/                --> graphiques générés (non versionnés)
```

## Scripts

### `src/marche_aleatoire.py` : Modèles M1 et M2

**M1 (forward-in-time)** : deux marcheurs partent du même point et se déplacent
indépendamment pendant `t` pas. On mesure la distance euclidienne entre eux à la fin.

**M2 (backward-in-time / coalescence)** : deux marcheurs partent de points séparés
par une distance `d0`. On stoppe quand ils se retrouvent au même noeud et on enregistre
le temps de coalescence.

```bash
python src/marche_aleatoire.py --modele M1 --m 0.5 --n 100 --t 500 --rep 200
python src/marche_aleatoire.py --modele M2 --m 0.5 --n 100 --d0 10 --rep 200
```

| Option          | Description                                      | Défaut |
|-----------------|---------------------------------------------------|--------|
| `--modele`      | Modèle à simuler : `M1` ou `M2`                  | -      |
| `--m`           | Probabilité de se déplacer à chaque pas          | `0.5`  |
| `--n`           | Taille de la grille (n x n)                      | `100`  |
| `--rep`         | Nombre de répétitions                            | `200`  |
| `--t`           | [M1] Nombre de pas de temps                      | `500`  |
| `--d0`          | [M2] Distance initiale entre les deux marcheurs  | `10`   |
| `--afficher`    | Affiche le graphique à l'écran                   | -      |
| `--sauvegarder` | Sauvegarde le graphique en `.png`                | -      |

---

### `src/marche_aleatoire_eq.py` : Vérification d'équivalence M1/M2

Vérifie que M1 et M2 sont équivalents en comparant leur distribution
jointe (distance, temps). Les deux modèles sont adaptés pour que distance et temps
soient tous les deux aléatoires, puis leurs densités jointes sont comparées visuellement.

```bash
python src/marche_aleatoire_eq.py --m 0.5 --n 15 --T 2000 --rep 8000
```

| Option          | Description                                           | Défaut  |
|-----------------|---------------------------------------------------------|---------|
| `--m`           | Probabilité de se déplacer à chaque pas                | `0.5`   |
| `--n`           | Taille de la grille (n x n)                             | `15`    |
| `--T`           | Temps maximal (t tiré dans [0, T] pour M1)              | `2000`  |
| `--rep`         | Nombre de répétitions                                   | `8000`  |
| `--mode`        | Type de graphique : `joint` (2D) ou `marginal` (1D)     | `joint` |
| `--afficher`    | Affiche les graphiques à l'écran                        | -       |
| `--sauvegarder` | Sauvegarde les graphiques en `.png`                     | -       |

---

### `src/moran_spatialise.py` : Processus de Moran spatialisé

Compare les distributions forward et backward sur une grille l x l.
A chaque pas, on tire B au hasard sur la grille. B se propage sur A avec
probabilité m/4 par direction disponible, et 1 - k*m/4 de rester sur place
(k = nb de voisins de B). Les événements "rester sur place" sont inclus
dans la liste et comptent comme un pas de temps.

**Forward** : on construit un arbre de descendants (classe `Noeud`). Pour chaque
noeud valide (descendants vivants des deux côtés), on enregistre le temps et
la distance entre feuilles.

**Backward** : on tire deux individus au présent selon un schéma d'échantillonnage
et on remonte les événements jusqu'à coalescence.

Les trois schémas (uniforme, cercle, diagonale) tournent à chaque répétition.
T est calculé automatiquement depuis la courbe analytique si non fourni.

```bash
python src/moran_spatialise.py --l 7 --rep 200 --afficher
python src/moran_spatialise.py --l 7 --T 50000 --rep 200 --sauvegarder
```

| Option          | Description                                        | Défaut |
|-----------------|------------------------------------------------------|--------|
| `--l`           | Côté de la grille (population = l*l)               | `7`    |
| `--T`           | Nombre de pas. Si absent, calculé automatiquement  | -      |
| `--m`           | Taux de migration                                  | `1.0`  |
| `--rep`         | Nombre de répétitions                              | `200`  |
| `--sigma`       | Ecart-type pour le schéma diagonale                | `1.0`  |
| `--rayon`       | Rayon pour le schéma cercle                        | `l/4`  |
| `--quantile`    | Percentile pour le crop des histogrammes           | `99`   |
| `--afficher`    | Affiche les graphiques à l'écran                   | -      |
| `--sauvegarder` | Sauvegarde les graphiques en `.png`                | -      |

---

### `src/moran_lignee_unique.py` : Diffusion d'une lignée unique

Même processus de Moran que `moran_spatialise.py`, mais on ne suit qu'une seule
lignée depuis un point de départ `(x0, y0)` (centre de la grille par défaut).
On construit le même arbre forward, puis on tire une feuille vivante au hasard
parmi les descendants de `(x0, y0)` au temps T. Si la lignée s'est fait écraser,
la répétition est comptée comme éteinte.

On compare la distribution des positions obtenues avec la formule analytique
de la marche aléatoire 2D. La formule reçoit `t = T/n`.

```bash
python src/moran_lignee_unique.py --l 10 --T 500 --rep 3000 --m 0.5 --afficher
python src/moran_lignee_unique.py --l 10 --T 500 --rep 3000 --m 0.5 --sauvegarder
python src/moran_lignee_unique.py --l 10 --T 500 --m 0.5 --analytique_3d --afficher
```

| Option             | Description                                            | Défaut |
|--------------------|----------------------------------------------------------|--------|
| `--l`              | Côté de la grille (population = l*l)                    | `7`    |
| `--T`              | Nombre de pas. Si absent, calculé automatiquement       | -      |
| `--m`              | Taux de migration                                       | `1.0`  |
| `--rep`            | Nombre de répétitions                                   | `500`  |
| `--x0`             | Coordonnée x du point de départ                         | centre |
| `--y0`             | Coordonnée y du point de départ                         | centre |
| `--afficher`       | Affiche les graphiques à l'écran                        | -      |
| `--sauvegarder`    | Sauvegarde les graphiques en `.png`                     | -      |
| `--analytique_3d`  | Affiche uniquement la surface 3D analytique             | -      |

---

### `src/verif_approximation.py` : Vérification de l'approximation tau = T·m/n

Pour une lignée unique, vérifie qu'on peut remplacer le mélange (nombre
aléatoire de déplacements) par un seul nombre de déplacements moyen
`tau = T*m/n`. Compare trois approches analytiques : `simple` (un seul tau),
`melange` (moyenne pondérée sur tous les tau possibles, loi binomiale) et
`exp` (formule à temps continu). Affiche les heatmaps des trois approches,
la distribution de distance au départ, et la probabilité de transition en
fonction du temps.

```bash
python src/verif_approximation.py --l 7 --T 5000 --m 1.0 --afficher
```

| Option          | Description                                              | Défaut |
|-----------------|--------------------------------------------------------------|--------|
| `--l`           | Côté de la grille (population = l*l)                    | `7`    |
| `--T`           | Temps de Moran                                           | `5000` |
| `--m`           | Taux de migration                                        | `1.0`  |
| `--lam`         | lambda = 1/temps de génération                           | `1.0`  |
| `--r_max`       | Nombre de pas max pour le graphe proba/temps              | `40`   |
| `--afficher`    | Affiche les graphiques à l'écran                         | -      |
| `--sauvegarder` | Sauvegarde les graphiques en `.png`                       | -      |

---

### `src/tableau_joint.py` : Tableau joint des deux lignées

Calcule la probabilité jointe analytique `P(z1, z2 | t)` que deux lignées
issues d'un même ancêtre se retrouvent en `z1` et `z2`, pour un temps total
`t` donné (formule à temps continu). Le temps total `T` est coupé en
plusieurs tranches égales, et le tableau exporté contient une colonne de
probabilité par tranche de temps. Sert de référence analytique à comparer
aux simulations forward.

```bash
python src/tableau_joint.py --l 7 --T 50 --n_t 5 --m 1.0 --sauvegarder
```

| Option          | Description                                              | Défaut |
|-----------------|--------------------------------------------------------------|--------|
| `--l`           | Côté de la grille (population = l*l)                    | `7`    |
| `--T`           | Temps total (calendaire)                                 | `50.0` |
| `--n_t`         | Nombre de temps (coupes de T)                             | `5`    |
| `--m`           | Taux de migration                                        | `1.0`  |
| `--lam`         | lambda = 1/temps de génération                           | `1.0`  |
| `--afficher`    | Affiche la figure                                         | -      |
| `--sauvegarder` | Sauvegarde la figure et le csv                            | -      |

---

### `src/comparaison_analytique.py` : P(distance) simulation vs analytique

Compare la distribution `P(distance)` obtenue par simulation forward avec
deux versions analytiques : sans correction (`proba_transition_exp`,
distance nulle incluse) et avec correction (`proba_transition_exp_cond`,
distance nulle mise à zéro puis renormalisée). Un density plot (KDE) par
tranche de temps, simulation et analytique superposées.

```bash
python src/comparaison_analytique.py --l 7 --rep 200 --afficher
python src/comparaison_analytique.py --l 7 --rep 200 --sauvegarder --seed 42
```

| Option          | Description                                                    | Défaut |
|-----------------|------------------------------------------------------------------|--------|
| `--l`           | Côté de la grille (population = l*l)                            | `7`    |
| `--T`           | Nombre de pas. Si absent, calculé automatiquement                | -      |
| `--m`           | Taux de migration                                                | `1.0`  |
| `--lam`         | lambda = 1/temps de génération                                   | `1.0`  |
| `--rep`         | Nombre de répétitions                                             | `200`  |
| `--n_tirages`   | Nb de valeurs tirées dans l'analytique pour le KDE                | `2000` |
| `--n_temps`     | Nb de tranches de temps pour le tableau analytique                | `50`   |
| `--afficher`    | Affiche les graphiques                                            | -      |
| `--sauvegarder` | Sauvegarde les graphiques en `.png`                               | -      |
| `--seed`        | Graine pour reproductibilité                                      | -      |

---

### `src/comparaison_pt_selon_m.py` : Densité du temps de coalescence selon m

Pour plusieurs valeurs de `m`, compare l'histogramme du temps de
coalescence simulé (toutes distances confondues) à la densité théorique
`densite_temps_coalescence`. Un sous-graphique par `m`, tous avec le même
`T` pour rester comparables entre eux.

```bash
python src/comparaison_pt_selon_m.py --l 7 --T 57491 --rep 100
python src/comparaison_pt_selon_m.py --l 7 --T 57491 --rep 100 --m_liste 1.0 0.5 0.1 0.05
```

| Option             | Description                                                                | Défaut              |
|---------------------|-----------------------------------------------------------------------------|----------------------|
| `--l`              | Côté de la grille (population = l*l)                                       | `7`                  |
| `--T`              | Nombre de pas de Moran, le même pour tous les m                            | `60000`              |
| `--lam`            | lambda = 1/temps de génération                                             | `1.0`                |
| `--rep`            | Nombre de répétitions par m                                                | `100`                |
| `--m_liste`        | Liste des m à tester                                                       | `1.0 0.5 0.1 0.05`   |
| `--t_max_affiche`  | Largeur de l'axe X (temps calendaire). Si absent, calculé automatiquement  | -                    |
| `--sauvegarder`    | Sauvegarde le graphique en `.png`                                          | -                    |
| `--seed`           | Graine pour reproductibilité                                               | -                    |

---

### `src/comparaison_temps_calendaire.py` : Moran vs marche aléatoire en temps calendaire

À distance de départ fixée, compare le temps de coalescence en temps
calendaire entre le processus de Moran (`generer_evenements` +
`forward_uniforme`, comme dans `comparaison_analytique.py`) et une marche
aléatoire à deux lignées (une des deux bouge à chaque tour, temps cumulé
via une loi exponentielle de paramètre `lam`). Un sous-graphique par
distance testée.

```bash
python src/comparaison_temps_calendaire.py --l 7 --rep 500 --afficher
python src/comparaison_temps_calendaire.py --l 7 --distances 1 2 4 6 --rep 500 --sauvegarder
```

| Option           | Description                                                            | Défaut  |
|-------------------|--------------------------------------------------------------------------|---------|
| `--l`            | Côté de la grille (population Moran = l*l)                              | `7`     |
| `--m`            | Taux de migration                                                       | `1.0`   |
| `--lam`          | lambda = 1/temps de génération                                          | `1.0`   |
| `--distances`    | Liste des distances d0 à comparer. Si absent, calculé automatiquement   | -       |
| `--rep`          | Nombre de répétitions du forward Moran                                  | `500`   |
| `--rep_marche`   | Nombre de simulations marche aléatoire                                  | `20000` |
| `--T`            | Nombre de pas de Moran. Si absent, calculé automatiquement               | -       |
| `--afficher`     | Affiche les graphiques                                                  | -       |
| `--sauvegarder`  | Sauvegarde les graphiques en `.png`                                     | -       |
| `--seed`         | Graine pour reproductibilité                                            | -       |

---

### `src/vraisemblance/` : Inférence du taux de migration

Génère des jeux de données (position de paires de lignées + temps de coalescence)
selon deux méthodes, puis estime `m` et/ou `l` par maximum de vraisemblance.

**Deux méthodes de génération**, produisant le même format de csv en sortie :

- **Liste d'événements** (`ecrire_evenements.py` + `generation_donnees_evenements.py`) :
  la "vraie" méthode Moran. Génère tous les événements sur la grille complète, puis
  remonte le temps pour chaque paire d'échantillons jusqu'à coalescence. Coûteux en
  mémoire et en temps, gardé comme référence pour valider l'approximation ci-dessous.
- **Généalogie directe** (`generation_donnees_genealogie.py`) : simule directement
  la généalogie des lignées échantillonnées (approximation par marche aléatoire),
  sans passer par un fichier d'événements. Beaucoup plus rapide, méthode utilisée
  en pratique pour les runs de production.

```bash
python src/vraisemblance/generation_donnees_genealogie.py --l 100 --m 0.3 --n_echantillons 50 --id_jeu 0 --sauvegarder
python src/vraisemblance/ecrire_evenements.py --l 100 --m 0.5 --sauvegarder
python src/vraisemblance/generation_donnees_evenements.py --entree evenements/evenements_tmp.csv --l 20 --m 0.5 --n_echantillons 50 --sauvegarder
```

**Vraisemblance**, deux scripts selon si `l` est fixé ou estimé :

| Script                          | Paramètre(s) estimé(s) | Mode                        |
|----------------------------------|-------------------------|------------------------------|
| `vraisemblance.py`               | `m` seul                | `l` fixé à sa vraie valeur  |
| `vraisemblance_ml.py` (défaut)   | `m` et `l` ensemble     | grille ou `--optimiser`     |
| `vraisemblance_ml.py --m_fixe`   | `l` seul                | `m` fixé à sa vraie valeur  |

```bash
python src/vraisemblance/vraisemblance.py --entree donnees/donnees_jeu0.csv --n_m 50 --sauvegarder
python src/vraisemblance/vraisemblance_ml.py --entree donnees/donnees_jeu0.csv --n_m 20 --n_l 20 --sauvegarder
python src/vraisemblance/vraisemblance_ml.py --entree donnees/donnees_jeu0.csv --m_fixe --sauvegarder
python src/vraisemblance/vraisemblance_ml.py --entree donnees/donnees_jeu0.csv --optimiser --sauvegarder
```

| Option                | Description                                                     | Défaut       |
|-------------------------|--------------------------------------------------------------------|---------------|
| `--entree`             | Csv de données (un seul jeu)                                       | -             |
| `--n_m`, `--n_l`       | Nombre de valeurs testées en grille                                | `50` / `20`   |
| `--l_min`, `--l_max`   | Bornes de `l` testées (`vraisemblance_ml.py`)                      | `5` / `200`   |
| `--cond`               | Exclut les paires à distance nulle (renormalise)                   | -             |
| `--echantillonnage`    | Corrige pour un schéma d'échantillonnage non uniforme              | -             |
| `--m_fixe`             | Fixe `m` à sa vraie valeur, estime `l` seul (`vraisemblance_ml.py`) | -             |
| `--optimiser`          | Nelder-Mead / Brent au lieu d'une grille complète                  | -             |
| `--sauvegarder`        | Exporte les résultats en `.csv`                                    | -             |

**Graphiques**, un script par échelle (un seul jeu / agrégat de plusieurs jeux)
et par pipeline (`m` seul / `m` et `l`) :

| Script                            | Échelle          | Paramètres                                              |
|-------------------------------------|--------------------|------------------------------------------------------------|
| `graphique_un_jeu.py`              | un seul jeu       | `m`                                                        |
| `graphique_un_jeu_ml.py`           | un seul jeu       | `m` et `l` (carte de chaleur)                              |
| `graphique_vraisemblance.py`       | plusieurs jeux    | `m`                                                        |
| `graphique_vraisemblance_ml.py`    | plusieurs jeux    | `m` et `l` (un jeu de figures par taille d'échantillon)    |

```bash
python src/vraisemblance/graphique_un_jeu.py --donnees donnees/donnees_jeu0.csv --resultats resultats/resultats_jeu0.csv --afficher
python src/vraisemblance/graphique_vraisemblance.py --dossier_donnees donnees --dossier_resultats resultats --afficher
python src/vraisemblance/graphique_vraisemblance_ml.py --dossier_donnees donnees --dossier_resultats resultats --afficher
```

**Scripts d'orchestration** (local et cluster IFB, SLURM) : chaque méthode de
génération a sa version locale et sa version cluster, et l'analyse par
vraisemblance a son propre script de soumission cluster.

| Script                                    | Rôle                                                          |
|----------------------------------------------|--------------------------------------------------------------|
| `lancer_generation_genealogie_local.py`    | Génère `n_jeux` jeux en local (généalogie)                   |
| `lancer_generation_genealogie_cluster.py`  | Idem, un job SLURM par jeu                                    |
| `lancer_generation_evenements_local.py`    | Génère `n_jeux` jeux en local (liste d'événements)            |
| `lancer_generation_evenements_cluster.py`  | Idem, job array SLURM throttle (quota disque)                 |
| `lancer_analyse_cluster.py`                | Calcule `vraisemblance.py` sur le cluster, un job par fichier |
| `lancer_analyse_ml_cluster.py`             | Idem avec `vraisemblance_ml.py`                                |

Les chemins de cluster (`DOSSIER_SCRIPTS`, `DOSSIER_TRAVAIL`, compte SLURM) sont
en constantes en haut de chaque script `*_cluster.py`, à adapter si besoin.

## Dépendances

```bash
pip install -r requirements.txt
```