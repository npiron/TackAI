# 📹 Personal Best Replay System

## Comment ça marche ?

Quand l'IA bat son **Personal Best** pendant l'entraînement, sa trajectoire est **automatiquement sauvegardée** dans `replays/personal_best.json`.

## Utilisation

### 1. Entraîner l'IA
```bash
python3 rl_train.py --steps 2000000
```

Attends de voir dans les logs :
```
🏆 NEW PB: 42.35s (-2.15s) +107 pts | 📹 Replay saved!
```

### 2. Regarder le Replay
```bash
python3 watch_pb_replay.py
```

### Contrôles
- **SPACE** : Pause/Resume
- **R** : Restart
- **ESC** : Quit

## Ce qui est enregistré

- ✅ **Trajectoire complète** (toutes les positions)
- ✅ **Temps du lap**
- ✅ **Timestamp** (quand le PB a été fait)

## Fichier de Replay

`replays/personal_best.json` :
```json
{
    "time": 42.35,
    "trajectory": [
        [120.5, 580.2],
        [122.1, 579.8],
        ...
    ],
    "timestamp": 1703234567.89
}
```

## Intégration Dashboard (à venir)

Un bouton **"📹 Watch PB"** sera ajouté au dashboard pour lancer le replay directement depuis l'interface.

## Astuces

- Le replay se met à jour **automatiquement** à chaque nouveau PB
- Tu peux comparer visuellement les trajectoires en gardant l'ancien fichier
- Le ghost trail montre le chemin parcouru

## Prochaines Fonctionnalités

- [ ] Sauvegarder top 3 replays
- [ ] Comparer 2 replays côte à côte
- [ ] Exporter en vidéo
- [ ] Ralenti/Accéléré
