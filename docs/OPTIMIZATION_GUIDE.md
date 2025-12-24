# 🚀 Guide d'Optimisation "State of the Art"

Nous utilisons un script d'optimisation avancé inspiré de **RL-Zoo3**, l'état de l'art pour Stable Baselines3.

## 🌟 Fonctionnalités Avancées

- **Algorithme** : TPE (Tree-structured Parzen Estimator) multivarié
- **Pruning** : MedianPruner (coupe les essais médiocres automatiquement)
- **Persistance** : Sauvegarde SQL (SQLite) automatique
- **Architecture** : Optimise non seulement les hyperparamètres mais aussi la **structure du réseau** (taille, activation, init)

## 🛠️ Lancer l'optimisation

```bash
# Lancer 50 essais (prend environ 1-2h)
python3 manage.py optimize --trials 50
```

### Options utiles
- `--timeout 3600` : Arrêter après 1 heure
- `--clear` : Effacer l'étude précédente et recommencer à zéro

## 📊 Visualisation (Dashboard)

Vous pouvez visualiser l'optimisation en temps réel !

1. Installer le dashboard :
   ```bash
   pip install optuna-dashboard
   ```

2. Lancer le dashboard :
   ```bash
   optuna-dashboard sqlite:///data/optimization/optuna_study.db
   ```
   Rendez-vous sur `http://127.0.0.1:8080`

## 🔎 Espace de Recherche

Nous optimisons TOUT :

### 🧠 Architecture du Réseau
- **net_arch** : `tiny`, `small`, `medium` (profondeur et largeur)
- **activation_fn** : `ReLU` vs `Tanh`
- **ortho_init** : Initialisation orthogonale des poids (True/False)

### ⚙️ Paramètres PPO
- **batch_size** : 64 à 512
- **n_steps** : 1024 à 8192
- **gamma** : Facteur d'oubli
- **learning_rate** : Vitesse d'apprentissage
- **ent_coef** : Exploration
- **clip_range** : Stabilité
- **n_epochs** : Nombre de passages par update
- **gae_lambda** : Lissage de l'avantage
- **max_grad_norm** : Clipping des gradients

## 📝 Utiliser les meilleurs paramètres

Une fois l'optimisation terminée, les meilleurs paramètres sont sauvegardés dans `data/optimization/best_hyperparams.txt`.

Pour entraîner avec ces paramètres "ultimes" :

```bash
python3 manage.py train --use-best-params
```

---
*Ce système est équivalent à ce que les chercheurs utilisent pour battre les benchmarks RL.*
