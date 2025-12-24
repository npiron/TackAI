# 🎛️ Éditeur d'Hyperparamètres - Instructions

## Accès Rapide

L'éditeur d'hyperparamètres sera bientôt disponible dans le dashboard.

En attendant, tu peux éditer directement le fichier :

```bash
nano logs/best_hyperparams.txt
```

## Presets Disponibles

### 🛡️ Stable (Recommandé) - Actuellement actif
```json
{
    "n_steps": 2048,
    "batch_size": 256,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "ent_coef": 0.005,
    "learning_rate": 0.0003,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5
}
```

### ⚡ Rapide (Apprentissage accéléré)
```json
{
    "n_steps": 1024,
    "batch_size": 512,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "learning_rate": 0.0005,
    "clip_range": 0.2,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5
}
```

### 🎯 Fine-tuning (Optimisation finale)
```json
{
    "n_steps": 2048,
    "batch_size": 128,
    "gamma": 0.995,
    "gae_lambda": 0.95,
    "ent_coef": 0.001,
    "learning_rate": 0.0001,
    "clip_range": 0.15,
    "max_grad_norm": 0.3,
    "vf_coef": 0.5
}
```

## Utilisation

1. Copie le preset que tu veux
2. Colle-le dans `logs/best_hyperparams.txt`
3. Lance l'entraînement avec `--use-best-params`

```bash
python3 rl_train.py --use-best-params --steps 2000000
```

## Note

L'interface graphique complète sera ajoutée dans une prochaine mise à jour.
Pour l'instant, l'édition manuelle du fichier fonctionne parfaitement !
