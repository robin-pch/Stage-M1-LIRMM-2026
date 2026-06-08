# Stage M1 Bioinformatique - Robin Pioch

**LIRMM, Université de Montpellier**  
Encadrant : Stéphane Guindon  
2026

## Contexte

Ce dépôt regroupe les scripts développés durant le stage, portant sur l'étude in silico
de modèles de génétique spatiale.

## Structure

```
src/        --> scripts Python
docs/       --> rapport bibliographique et journal de stage
results/    --> graphiques générés (non versionnés)
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
|-----------------|--------------------------------------------------|--------|
| `--modele`      | Modèle à simuler : `M1` ou `M2`                 | -      |
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

| Option          | Description                                              | Défaut   |
|-----------------|----------------------------------------------------------|----------|
| `--m`           | Probabilité de se déplacer à chaque pas                  | `0.5`    |
| `--n`           | Taille de la grille (n x n)                              | `15`     |
| `--T`           | Temps maximal (t tiré dans [0, T] pour M1)               | `2000`   |
| `--rep`         | Nombre de répétitions                                    | `8000`   |
| `--mode`        | Type de graphique : `joint` (2D) ou `marginal` (1D)      | `joint`  |
| `--afficher`    | Affiche les graphiques à l'écran                         | -        |
| `--sauvegarder` | Sauvegarde les graphiques en `.png`                      | -        |

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

| Option          | Description                                       | Défaut |
|-----------------|---------------------------------------------------|--------|
| `--l`           | Côté de la grille (population = l*l)              | `7`    |
| `--T`           | Nombre de pas. Si absent, calculé automatiquement | -      |
| `--m`           | Taux de migration                                 | `1.0`  |
| `--rep`         | Nombre de répétitions                             | `200`  |
| `--sigma`       | Ecart-type pour le schéma diagonale               | `1.0`  |
| `--rayon`       | Rayon pour le schéma cercle                       | `l/4`  |
| `--quantile`    | Percentile pour le crop des histogrammes          | `99`   |
| `--afficher`    | Affiche les graphiques à l'écran                  | -      |
| `--sauvegarder` | Sauvegarde les graphiques en `.png`               | -      |

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

| Option             | Description                                           | Défaut |
|--------------------|-------------------------------------------------------|--------|
| `--l`              | Côté de la grille (population = l*l)                  | `7`    |
| `--T`              | Nombre de pas. Si absent, calculé automatiquement     | -      |
| `--m`              | Taux de migration                                     | `1.0`  |
| `--rep`            | Nombre de répétitions                                 | `500`  |
| `--x0`             | Coordonnée x du point de départ                       | centre |
| `--y0`             | Coordonnée y du point de départ                       | centre |
| `--afficher`       | Affiche les graphiques à l'écran                      | -      |
| `--sauvegarder`    | Sauvegarde les graphiques en `.png`                   | -      |
| `--analytique_3d`  | Affiche uniquement la surface 3D analytique           | -      |

## Dépendances

```bash
pip install -r requirements.txt
```