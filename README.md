# Stage M1 Bioinformatique – Robin Pioch

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
par une distance `d0`. On stoppe quand ils se retrouvent au même nœud et on enregistre
le temps de coalescence.

```bash
python src/marche_aleatoire.py --modele M1 --m 0.5 --n 100 --t 500 --rep 200
python src/marche_aleatoire.py --modele M2 --m 0.5 --n 100 --d0 10 --rep 200
```

Options disponibles :

| Option          | Description                                      | Défaut |
|-----------------|--------------------------------------------------|--------|
| `--modele`      | Modèle à simuler : `M1` ou `M2`                 | -      |
| `--m`           | Probabilité de se déplacer à chaque pas          | `0.5`  |
| `--n`           | Taille de la grille (n × n)                      | `100`  |
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

Options disponibles :

| Option          | Description                                              | Défaut   |
|-----------------|----------------------------------------------------------|----------|
| `--m`           | Probabilité de se déplacer à chaque pas                  | `0.5`    |
| `--n`           | Taille de la grille (n × n)                              | `15`     |
| `--T`           | Temps maximal (t tiré dans [0, T] pour M1)               | `2000`   |
| `--rep`         | Nombre de répétitions                                    | `8000`   |
| `--mode`        | Type de graphique : `joint` (2D) ou `marginal` (1D)      | `joint`  |
| `--afficher`    | Affiche les graphiques à l'écran                         | -        |
| `--sauvegarder` | Sauvegarde les graphiques en `.png`                      | -        |

## Dépendances

```bash
pip install -r requirements.txt
```
