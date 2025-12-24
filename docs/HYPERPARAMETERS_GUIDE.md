# 🎛️ Guide des Hyperparamètres

## Paramètres Actuels (Conservateurs & Stables)

```json
{
    "n_steps": 2048,          // ✅ Stable
    "batch_size": 256,        // ✅ Équilibré
    "gamma": 0.995,           // ✅ Long-terme
    "gae_lambda": 0.95,       // ✅ Standard
    "ent_coef": 0.005,        // ✅ Peu d'exploration (bon pour stabilité)
    "learning_rate": 0.0003,  // ✅ Modéré (avec schedule)
    "clip_range": 0.2,        // ✅ Standard PPO
    "max_grad_norm": 0.5,     // ✅ Gradients contrôlés
    "vf_coef": 0.5            // ✅ Value function standard
}
```

## 📊 Impact de Chaque Paramètre

### 1. **n_steps** (Steps avant update)
- **Valeur** : `2048`
- **Impact** : Nombre de steps collectés avant de mettre à jour le réseau
- ⬆️ Plus haut = Plus stable, mais apprend plus lentement
- ⬇️ Plus bas = Apprend plus vite, mais peut être instable
- **Recommandation** : `2048` pour stabilité, `1024` si tu veux plus de vitesse

### 2. **batch_size** (Taille des lots)
- **Valeur** : `256`
- **Impact** : Combien d'exemples utilisés par update
- ⬆️ Plus haut = Gradients plus stables, mais moins de diversité
- ⬇️ Plus bas = Plus de diversité, mais gradients bruités
- **Recommandation** : `256` est optimal pour la plupart des cas

### 3. **learning_rate** (Vitesse d'apprentissage)
- **Valeur** : `0.0003` (3e-4) avec **schedule décroissant**
- **Impact** : Taille des pas d'apprentissage
- ⬆️ Plus haut = Apprend plus vite, mais risque d'oublier (catastrophic forgetting)
- ⬇️ Plus bas = Apprend lentement, mais stable
- **Recommandation** : 
  - Début : `5e-4` (rapide)
  - Milieu : `3e-4` (stable)
  - Fin : `1e-4` (fine-tuning)

### 4. **ent_coef** (Coefficient d'entropie / Exploration)
- **Valeur** : `0.005`
- **Impact** : Encourage l'IA à explorer de nouvelles stratégies
- ⬆️ Plus haut = Plus d'exploration (bon au début)
- ⬇️ Plus bas = Exploitation (bon quand l'IA maîtrise)
- **Recommandation** : 
  - Début : `0.01` (explore)
  - Après 1M steps : `0.005` (exploite)

### 5. **gamma** (Discount factor)
- **Valeur** : `0.995`
- **Impact** : Importance des récompenses futures
- ⬆️ Plus haut (proche de 1) = Pense très long-terme
- ⬇️ Plus bas = Préfère récompenses immédiates
- **Recommandation** : `0.995` pour circuits (long-terme)

### 6. **clip_range** (PPO Clip)
- **Valeur** : `0.2`
- **Impact** : Limite les changements brusques de politique
- ⬆️ Plus haut = Permet plus de changements
- ⬇️ Plus bas = Plus conservateur
- **Recommandation** : `0.2` est standard PPO

## 🎯 Quand Utiliser Quoi ?

### Apprentissage Rapide (mais risqué)
```json
{
    "n_steps": 1024,
    "batch_size": 512,
    "learning_rate": 0.0005,
    "ent_coef": 0.01
}
```
✅ Bon pour : Expérimentation rapide
❌ Risque : Instabilité, catastrophic forgetting

### Apprentissage Stable (recommandé)
```json
{
    "n_steps": 2048,
    "batch_size": 256,
    "learning_rate": 0.0003,
    "ent_coef": 0.005
}
```
✅ Bon pour : Entraînement long et fiable
❌ Inconvénient : Plus lent

### Fine-Tuning (après 1M+ steps)
```json
{
    "n_steps": 2048,
    "batch_size": 128,
    "learning_rate": 0.0001,
    "ent_coef": 0.001
}
```
✅ Bon pour : Optimiser un modèle déjà bon
❌ Inconvénient : Très lent

## 🔧 Comment Tester ?

1. **Lance avec paramètres stables** (actuels)
2. **Observe pendant 500k steps**
3. Si ça marche bien → Continue
4. Si c'est trop lent → Augmente `learning_rate` à `5e-4`
5. Si c'est instable → Réduis `ent_coef` à `0.003`

## 💡 Astuce

Le **Learning Rate Schedule** (déjà implémenté) est plus important que les valeurs fixes !
Il commence rapide et ralentit automatiquement pour éviter l'oubli.
