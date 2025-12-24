# 🧠 Configuration DQN - Guide Complet

## Vue d'Ensemble

Ce projet utilise **DQN (Deep Q-Network)** pour entraîner une IA à conduire une voiture sur circuit avec des **contrôles discrets on/off** (pas de pourcentages d'accélération).

## 🎮 Espace d'Actions - Discrete(9)

L'espace d'actions est **complètement discret** avec 9 actions possibles (boutons on/off uniquement):

| Action | Description | Steer | Accel | Brake |
|--------|-------------|-------|-------|-------|
| 0 | Idle (aucune entrée) | 0.0 | 0.0 | 0.0 |
| 1 | Accelerate | 0.0 | 1.0 | 0.0 |
| 2 | Brake | 0.0 | 0.0 | 1.0 |
| 3 | Left (tourner gauche) | -1.0 | 0.0 | 0.0 |
| 4 | Right (tourner droite) | 1.0 | 0.0 | 0.0 |
| 5 | Left + Accelerate | -1.0 | 1.0 | 0.0 |
| 6 | Right + Accelerate | 1.0 | 1.0 | 0.0 |
| 7 | Left + Brake | -1.0 | 0.0 | 1.0 |
| 8 | Right + Brake | 1.0 | 0.0 | 1.0 |

### ✅ Avantages de 9 Actions

1. **Contrôle Complet**: Toutes les combinaisons nécessaires pour conduire
2. **Freinage dans les Virages**: Essentiel pour bien négocier les courbes (actions 7 et 8)
3. **Contrôle Fin**: Tourner sans accélérer pour ajustements précis (actions 3 et 4)
4. **100% Discret**: Aucun pourcentage, seulement on/off (0.0 ou 1.0)

### ❌ Ancien Système (5 Actions)

L'ancien système n'avait que 5 actions et manquait:
- ❌ Left + Brake
- ❌ Right + Brake
- ❌ Left seul
- ❌ Right seul

Ces actions sont **critiques** pour un contrôle optimal.

## ⚙️ Hyperparamètres DQN

### Configuration Actuelle (Optimisée)

```python
{
    "buffer_size": 200_000,           # Mémoire de replay (200k transitions)
    "learning_starts": 5_000,         # Commence à apprendre après 5k steps
    "batch_size": 256,                # Taille des lots d'apprentissage
    "gamma": 0.99,                    # Facteur de discount
    "train_freq": 4,                  # Entraîne tous les 4 steps
    "gradient_steps": 2,              # ✨ 2 mises à jour par entraînement (amélioré)
    "target_update_interval": 1000,   # Mise à jour du réseau cible
    "exploration_fraction": 0.15,     # ✨ Explore 15% du temps (amélioré)
    "exploration_initial_eps": 1.0,   # 100% aléatoire au début
    "exploration_final_eps": 0.05,    # 5% aléatoire à la fin
    "learning_rate": "schedule"       # Décroît progressivement
}
```

### 🎯 Améliorations Apportées

1. **`gradient_steps: 2`** (avant: 1)
   - Plus de mises à jour par step d'entraînement
   - Apprentissage plus efficace
   
2. **`exploration_fraction: 0.15`** (avant: 0.1)
   - 15% du temps total consacré à l'exploration
   - Meilleure découverte de stratégies

3. **Action Space: 9** (avant: 5)
   - Contrôle complet de la voiture
   - Manœuvres plus complexes possibles

### 📊 Impact des Paramètres Clés

#### buffer_size (200,000)
- **Rôle**: Mémoire des expériences passées
- ⬆️ Plus grand = Plus stable, mais plus de RAM
- ⬇️ Plus petit = Moins stable, mais plus rapide
- **200k** = Bon équilibre

#### gradient_steps (2)
- **Rôle**: Nombre de mises à jour du réseau par étape
- ⬆️ Plus = Apprend plus vite de chaque expérience
- ⬇️ Moins = Plus conservateur
- **2** = Efficace sans surapprentissage

#### exploration_fraction (0.15)
- **Rôle**: Portion du temps consacrée à l'exploration aléatoire
- ⬆️ Plus = Plus d'exploration (bon pour environnements complexes)
- ⬇️ Moins = Plus d'exploitation (bon si déjà performant)
- **0.15** = 15% du temps total

#### gamma (0.99)
- **Rôle**: Importance des récompenses futures
- **0.99** = Pense à long terme (bon pour circuits)

## 🏗️ Architecture Réseau

```python
policy_kwargs = {
    "net_arch": [256, 256]  # 2 couches de 256 neurones
}
```

### Options d'Architecture

| Taille | Neurones | Usage |
|--------|----------|-------|
| Small | [64, 64] | Environnements simples, CPU |
| Medium | [256, 256] | **Recommandé** - Bon équilibre |
| Large | [512, 512] | Environnements complexes, GPU fort |

**Actuel**: Medium ([256, 256]) - Optimal pour ce projet

## 📈 Learning Rate Schedule

```python
def lr_schedule(progress_remaining):
    """
    Décroissance progressive du learning rate:
    - Début: 1e-3 (exploration rapide)
    - Fin: 1e-5 (fine-tuning)
    """
    return 1e-5 + (1e-3 - 1e-5) * progress_remaining
```

**Avantage**: L'IA apprend vite au début, puis se stabilise pour éviter l'oubli catastrophique.

## 🔄 Comparaison DQN vs PPO

| Aspect | DQN | PPO |
|--------|-----|-----|
| Type | Off-Policy | On-Policy |
| Mémoire | Replay Buffer (200k) | Petit buffer |
| Efficacité | Réutilise les données | Données jetées |
| Stabilité | Très stable | Stable |
| Vitesse | Plus lent | Plus rapide |
| **Meilleur pour** | Actions discrètes | Actions continues |

**Conclusion**: DQN est **parfait** pour ce projet car:
1. ✅ Actions 100% discrètes (on/off)
2. ✅ Réutilise les expériences (efficace)
3. ✅ Très stable pour l'apprentissage

## 🚀 Utilisation

### Entraînement Standard

```bash
python3 rl_train.py --steps 2000000
```

### Entraînement Visuel (Debug)

```bash
python3 rl_train.py --visual --steps 100000
```

### Continuer un Entraînement

```bash
python3 rl_train.py --load data/checkpoints/model.zip
```

### Avec Hyperparamètres Optimisés

```bash
python3 rl_train.py --use-best-params
```

## 📝 Notes Importantes

### ✅ Ce qui est Correct

1. **Actions Discrètes**: Discrete(9) - 100% on/off, pas de pourcentages
2. **DQN pour Discret**: DQN est optimal pour des actions discrètes
3. **Buffer de Replay**: 200k transitions = bonne mémoire
4. **Learning Rate Schedule**: Évite l'oubli catastrophique

### ⚠️ Points d'Attention

1. **RAM**: Buffer de 200k consomme ~2-3 GB de RAM
2. **Exploration**: Les 15% premiers du training sont aléatoires
3. **Temps**: DQN prend plus de temps que PPO mais est plus stable

## 🎓 Pour Aller Plus Loin

- [Documentation Stable-Baselines3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html)
- [Guide des Hyperparamètres](./HYPERPARAMETERS_GUIDE.md)
- [Guide d'Optimisation](./OPTIMIZATION_GUIDE.md)
