# 🔀 Génération Procédurale de Circuits

Le projet utilise maintenant un système avancé de génération procédurale de circuits.
Cela permet à l'IA d'apprendre la **conduite généraliste** plutôt que de mémoriser un seul circuit.

## Comment ça marche ?

### 1. Splines (Catmull-Rom)
Au lieu de rectangles fixes, le générateur crée une série de points de contrôle aléatoires en cercle, puis les relie avec une courbe mathématique fluide (Spline de Catmull-Rom).

### 2. Collision par Masque (Pixel-Perfect)
La route est dessinée sur une image virtuelle (masque).
Pour savoir si la voiture est sur la route, on regarde simplement la couleur du pixel sous la voiture.
Cela permet des formes de circuits complexes (virages serrés, épingles, lignes droites...).

### 3. Entraînement Dynamique
À chaque nouvel épisode d'entraînement (`reset()`), un **nouveau circuit unique** est généré.
- L'IA doit utiliser ses capteurs (LiDAR) pour voir la route.
- Elle ne peut plus "tricher" en apprenant la position x,y des virages.

## Configuration

Pour revenir au circuit statique (rectangulaire), vous pouvez modifier `src/rl/wrappers.py` :

```python
# Mode Procédural (Défaut)
self.ta = TimeAttackEnv(track=None, procedural=True)

# Mode Statique (Ancien)
# self.ta = TimeAttackEnv(build_track())
```

## Impact sur l'Apprentissage

- **Début** : L'apprentissage sera plus lent car la tâche est plus dure.
- **Long terme** : L'IA sera beaucoup plus robuste et capable de conduire sur n'importe quel circuit inconnu.
