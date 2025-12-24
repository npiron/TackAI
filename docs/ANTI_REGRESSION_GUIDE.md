# 🛡️ Guide Anti-Régression

## Problème : Catastrophic Forgetting

L'IA apprend bien, puis **oublie** progressivement ce qu'elle a appris.

## Solutions Implémentées

### 1. ✅ Learning Rate Schedule (NOUVEAU)
Le learning rate diminue progressivement :
- **Début (0-30%)** : `5e-4` → Apprentissage rapide
- **Milieu (30-70%)** : `3e-4` → Stabilisation
- **Fin (70-100%)** : `1e-4` → Fine-tuning sans oublier

### 2. 💾 Sauvegarder le meilleur modèle

Quand tu vois que l'IA performe bien (100% checkpoints) :

```bash
# Copie le modèle actuel
cp logs/XXXXX_steps.zip models/best_model.zip
```

Si l'IA régresse après, tu peux recharger :

```bash
python3 rl_train.py --load models/best_model.zip --steps 1000000
```

### 3. 📊 Monitoring

Regarde ces métriques dans le dashboard :
- **Success Rate** : Devrait rester à 100%
- **Avg Checkpoints** : Devrait rester à 9
- **Reward** : Peut fluctuer, c'est normal

Si Success Rate < 80% pendant 200+ épisodes → **STOP et reload**

### 4. ⚙️ Hyperparamètres Anti-Régression

```json
{
    "ent_coef": 0.005,  // Moins d'exploration (était 0.01)
    "clip_range": 0.15,  // Moins de changements brusques
    "max_grad_norm": 0.3  // Gradients plus petits
}
```

## Quand Relancer ?

### ✅ Continue si :
- Success Rate > 80%
- Reward fluctue mais ne s'effondre pas
- L'IA termine toujours le circuit

### ❌ Redémarre si :
- Success Rate < 50% pendant 500 épisodes
- Reward chute de >50%
- L'IA ne passe plus le 1er checkpoint

## Commande Optimale

```bash
# Avec learning rate schedule + checkpoints fréquents
python3 rl_train.py --use-best-params --steps 3000000
```

Le modèle sera sauvegardé tous les 50k steps dans `logs/`.
