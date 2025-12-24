import pytest
import subprocess
import os
import shutil
import glob
import time
import sys

class TestE2ETraining:
    
    @pytest.fixture
    def setup_logs(self):
        """Ensure clean logs directory for testing."""
        os.makedirs("logs", exist_ok=True)
        # Cleanup pre-existing test logs
        for f in glob.glob("logs/test_e2e_*"):
            os.remove(f)
        yield
        # Cleanup after test
        for f in glob.glob("logs/test_e2e_*"):
            os.remove(f)

    def test_training_runs_short_session(self, setup_logs):
        """Test that we can run a very short training session without crashing."""
        run_id = f"test_e2e_{int(time.time())}"
        
        # We need enough steps to trigger at least one loop, but not too many.
        # PPO default n_steps is 2048. If we set total steps small, it might finish fast.
        
        # Find venv python
        venv_python = os.path.abspath(".venv/bin/python")
        if not os.path.exists(venv_python):
            venv_python = sys.executable # Fallback
            
        cmd = [
            venv_python, "rl_train.py",
            "--steps", "100",  # Very short!
            "--run-id", run_id
        ]
        
        # Run process
        # Capture output to avoid spamming test runner
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Check exit code
        assert result.returncode == 0, f"Training failed with error:\n{result.stderr}"
        
        # Check artifacts
        # 1. Log file should exist
        logs = glob.glob(f"logs/{run_id}.log")
        # Note: Depending on how we implemented logging, it might be separate from the monitor log
        # In rl_train we do: print to stdout, dashboard redirects to file.
        # But we also have monitor logs: logs/monitor_headless_{run_id}_{pid}.csv
        
        monitor_logs = glob.glob(f"logs/monitor_*_{run_id}_*.csv")
        assert len(monitor_logs) > 0, "No monitor logs created"
        
        # 2. Model file should exist upon completion
        # logic: logs/{run_id}_final.zip
        final_model = f"logs/{run_id}_final.zip"
        assert os.path.exists(final_model), f"Final model {final_model} not found"
        
        # 3. VecNormalize
        vec_norm = f"logs/{run_id}_vecnormalize.pkl"
        assert os.path.exists(vec_norm), f"VecNormalize {vec_norm} not found"
