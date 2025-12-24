#!/usr/bin/env python3
"""
Script de nettoyage intelligent du projet Trackmania RL Clone.
Garde les 10 meilleurs modèles et nettoie les vieux logs.
"""

import os
import glob
import shutil
from datetime import datetime, timedelta

def cleanup_old_models(keep_count=10):
    """Garde seulement les N modèles les plus récents."""
    print(f"🧹 Nettoyage des modèles (garde les {keep_count} plus récents)...")
    
    # Trouver tous les modèles
    models = glob.glob("data/checkpoints/*.zip")
    
    if not models:
        print("   ℹ️  Aucun modèle trouvé.")
        return
    
    # Trier par date de modification (plus récent en premier)
    models.sort(key=os.path.getmtime, reverse=True)
    
    print(f"   📊 {len(models)} modèles trouvés")
    
    # Garder les N premiers, supprimer le reste
    to_keep = models[:keep_count]
    to_delete = models[keep_count:]
    
    if not to_delete:
        print(f"   ✅ Déjà optimisé ({len(models)} modèles)")
        return
    
    # Supprimer les vieux modèles
    deleted_size = 0
    for model in to_delete:
        size = os.path.getsize(model)
        deleted_size += size
        os.remove(model)
        
        # Supprimer aussi le fichier vecnormalize associé
        vec_file = model.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vec_file):
            deleted_size += os.path.getsize(vec_file)
            os.remove(vec_file)
    
    print(f"   🗑️  Supprimé {len(to_delete)} modèles ({deleted_size / 1024 / 1024:.1f} MB)")
    print(f"   ✅ Gardé {len(to_keep)} modèles les plus récents")

def cleanup_old_logs(days=7):
    """Supprime les logs de plus de N jours."""
    print(f"🧹 Nettoyage des logs (garde les {days} derniers jours)...")
    
    cutoff = datetime.now() - timedelta(days=days)
    deleted_count = 0
    deleted_size = 0
    
    for log_dir in ["data/logs/training", "data/logs/ai", "data/logs/game", "data/logs/optimization"]:
        if not os.path.exists(log_dir):
            continue
            
        for log_file in glob.glob(f"{log_dir}/*.log"):
            mtime = datetime.fromtimestamp(os.path.getmtime(log_file))
            if mtime < cutoff:
                size = os.path.getsize(log_file)
                deleted_size += size
                os.remove(log_file)
                deleted_count += 1
    
    if deleted_count > 0:
        print(f"   🗑️  Supprimé {deleted_count} logs ({deleted_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"   ✅ Aucun vieux log à supprimer")

def cleanup_old_monitoring(days=14):
    """Supprime les fichiers de monitoring de plus de N jours."""
    print(f"🧹 Nettoyage du monitoring (garde les {days} derniers jours)...")
    
    if not os.path.exists("data/monitoring"):
        print("   ℹ️  Aucun fichier de monitoring")
        return
    
    cutoff = datetime.now() - timedelta(days=days)
    deleted_count = 0
    deleted_size = 0
    
    for csv_file in glob.glob("data/monitoring/*.csv*"):
        mtime = datetime.fromtimestamp(os.path.getmtime(csv_file))
        if mtime < cutoff:
            size = os.path.getsize(csv_file)
            deleted_size += size
            os.remove(csv_file)
            deleted_count += 1
    
    if deleted_count > 0:
        print(f"   🗑️  Supprimé {deleted_count} fichiers ({deleted_size / 1024 / 1024:.1f} MB)")
    else:
        print(f"   ✅ Aucun vieux fichier de monitoring")

def show_disk_usage():
    """Affiche l'utilisation disque."""
    print("\n📊 Utilisation disque:")
    
    for directory in ["data/checkpoints", "data/logs", "data/monitoring"]:
        if os.path.exists(directory):
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(directory)
                for filename in filenames
            )
            print(f"   {directory}: {total_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    print("=" * 60)
    print("🏎️  NETTOYAGE DU PROJET TRACKMANIA RL CLONE")
    print("=" * 60)
    print()
    
    # Afficher l'état avant
    print("📊 AVANT:")
    show_disk_usage()
    print()
    
    # Nettoyage
    cleanup_old_models(keep_count=10)
    print()
    cleanup_old_logs(days=7)
    print()
    cleanup_old_monitoring(days=14)
    print()
    
    # Afficher l'état après
    print("📊 APRÈS:")
    show_disk_usage()
    print()
    print("✅ Nettoyage terminé !")
