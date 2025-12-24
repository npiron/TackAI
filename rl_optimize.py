import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import torch
import torch.nn as nn
import multiprocessing
import os
import sys
import time

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from trackmania_clone import TimeAttackEnv
from src.rl.wrappers import GymTimeAttack

# ==============================================================================
# 🧠 CONFIGURATION "STATE OF THE ART"
# ==============================================================================
# Inspiré de RL-Zoo3 pour DQN
# Stockage SQLite pour pouvoir reprendre l'optimisation et utiliser optuna-dashboard
STORAGE_URL = "sqlite:///data/optimization/optuna_study.db"
STUDY_NAME = "dqn_trackmania_v1"

class TrialEvalCallback(BaseCallback):
    """
    Callback avancé pour le pruning (élagage) des mauvais essais.
    Évalue l'agent périodiquement et signale le score à Optuna.
    Si le score est trop mauvais par rapport aux autres essais, on coupe !
    """
    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=10000, deterministic=True, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.trial = trial
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.last_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            # Synchronisation CRITIQUE des stats de normalisation
            if isinstance(self.model.get_vec_normalize_env(), VecNormalize):
                 self.eval_env.obs_rms = self.model.get_vec_normalize_env().obs_rms
                 self.eval_env.ret_rms = self.model.get_vec_normalize_env().ret_rms
            
            # Évaluation rapide mais assez précise
            mean_reward, std_reward = evaluate_policy(
                self.model, 
                self.eval_env, 
                n_eval_episodes=self.n_eval_episodes, 
                deterministic=self.deterministic
            )
            
            self.last_mean_reward = mean_reward
            
            # Reporter à Optuna
            self.trial.report(mean_reward, self.n_calls)
            
            # Pruning (Couper si mauvais)
            if self.trial.should_prune():
                print(f"✂️  Trial pruned at step {self.n_calls} (Reward: {mean_reward:.2f})")
                raise optuna.exceptions.TrialPruned()
                
            if self.verbose > 0:
                print(f"Step {self.n_calls}: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                
        return True

def make_env():
    # Headless pour la vitesse maximale
    # Procedural=True par défaut pour l'optimisation
    return GymTimeAttack(render_mode=None, procedural=True)

def sample_dqn_params(trial):
    """
    Espace de recherche complet inspiré de RL-Zoo3 pour DQN
    """
    # 1. Hyperparamètres d'entraînement
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    
    # 2. Hyperparamètres DQN spécifiques
    buffer_size = trial.suggest_categorical("buffer_size", [50000, 100000, 200000, 500000]) # Attention RAM
    # Si buffer_size trop gros par rapport à la RAM, réduire manuellement ici
    
    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32])
    gradient_steps = trial.suggest_categorical("gradient_steps", [1, 2, 4, 8])
    
    target_update_interval = trial.suggest_categorical("target_update_interval", [500, 1000, 5000, 10000])
    
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.1, 0.5)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.1)

    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.5, 1, 5, 10, 40])
    
    # 3. Architecture du Réseau
    # Pour DQN on spécifie juste [x, y], pas pi/vf
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    
    return {
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "target_update_interval": target_update_interval,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
        "max_grad_norm": max_grad_norm,
        "net_arch": net_arch_type
    }

def objective(trial):
    # Récupérer les hyperparamètres
    hyperparams = sample_dqn_params(trial)
    
    # Extraire les paramètres non-DQN
    net_arch_type = hyperparams.pop("net_arch")
    
    # Définir l'architecture
    net_archs = {
        "small": [64, 64],
        "medium": [256, 256],
        "large": [512, 512],
    }
    
    # Créer policy_kwargs (On garde activation par défaut ReLU pour DQN généralement)
    policy_kwargs = dict(
        net_arch=net_archs[net_arch_type],
    )
    
    # Configuration Environnement
    n_cpu = os.cpu_count() or 4
    
    # Créer environnement d'entraînement (rapide, vectorisé)
    venv = make_vec_env(make_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    # Normalisation est toujours utile
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Créer environnement d'évaluation (stable, non bruité)
    eval_env = make_vec_env(make_env, n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_env.training = False # Important: ne pas mettre à jour les stats pendant l'éval

    try:
        model = DQN(
            "MlpPolicy", 
            venv, 
            verbose=0, 
            policy_kwargs=policy_kwargs,
            **hyperparams
        )
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        venv.close()
        eval_env.close()
        return -float('inf')
    
    # Callback de Pruning
    # Vérifie toutes les 10k steps environ
    pruning_callback = TrialEvalCallback(
        eval_env, 
        trial, 
        n_eval_episodes=5, 
        eval_freq=max(5000, 10000 // n_cpu), # Adapter la fréquence
        deterministic=True
    )
    
    # Entraînement
    # Budget pour l'optimisation
    # DQN est plus lent à converger au début à cause du buffer filling
    total_timesteps = 100_000 # Un peu moins que PPO car DQN est plus lent en wall-clock si gros buffer
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=pruning_callback)
    except optuna.exceptions.TrialPruned:
        venv.close()
        eval_env.close()
        raise
    except Exception as e:
        print(f"❌ Training failed: {e}")
        venv.close()
        eval_env.close()
        return -float('inf')

    # Évaluation Finale (plus robuste, 10 épisodes)
    eval_env.obs_rms = venv.obs_rms # Synchro finale
    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
    
    venv.close()
    eval_env.close()

    return mean_reward

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--clear", action="store_true", help="Clear existing study definition")
    args = parser.parse_args()
    
    os.makedirs("data/optimization", exist_ok=True)

    print(f"🧠 Starting State-of-the-Art Optimization")
    print(f"   Trials: {args.trials}")
    print(f"   Storage: {STORAGE_URL}")
    print(f"   Dashboard: optuna-dashboard {STORAGE_URL}")
    
    # Pruner: Coupe la moitié des essais les moins bons (médiane)
    # Commence à couper après 20k steps (n_warmup_steps)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=20000, n_min_trials=10)
    
    if args.clear and os.path.exists("data/optimization/optuna_study.db"):
        print("🗑️  Clearing previous study...")
        optuna.delete_study(study_name=STUDY_NAME, storage=STORAGE_URL)

    # Création de l'étude persistante
    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="maximize", 
        sampler=TPESampler(n_startup_trials=10, multivariate=True), # Multivariate TPE is better
        pruner=pruner,
        storage=STORAGE_URL,
        load_if_exists=True
    )
    
    print(f"📊 Current best value: {study.best_value if len(study.trials) > 0 and study.best_value else 'None'}")
    
    try:
        study.optimize(objective, n_trials=args.trials, timeout=args.timeout, n_jobs=1)
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Progress is saved in DB.")
    
    print("\n✅ Optimization finished!")
    if len(study.trials) > 0:
        print(f"🏆 Best value: {study.best_value:.2f}")
        
        # Save best params to text file for easy reading by rl_train.py
        with open("data/optimization/best_hyperparams.txt", "w") as f:
            f.write(str(study.best_params))
            print("✅ Parameters saved to data/optimization/best_hyperparams.txt")
            
        print("\n💡 NOTE: You can now train with these params using:")
        print("   python3 manage.py train --use-best-params")
        print("\n📊 View detailed results with:")
        print("   pip install optuna-dashboard")
        print(f"   optuna-dashboard {STORAGE_URL}")
    else:
        print("❌ No trials completed.")

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
