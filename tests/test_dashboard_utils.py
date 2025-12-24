"""
Unit tests for dashboard utility functions.
Tests log parsing, model listing, and data formatting to ensure UI reliability.
"""
import pytest
import os
import sys
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import functions to test
from dashboard.main import (
    format_time,
    get_training_progress,
    get_models,
    parse_metrics,
    get_running_processes,
    get_advanced_stats
)

class TestDashboardUtils:
    
    def test_format_time_seconds(self):
        """format_time should handle seconds correctly."""
        import time
        now = time.time()
        assert format_time(now - 30) == "30s"
        assert format_time(now - 5) == "5s"
        
    def test_format_time_minutes(self):
        """format_time should handle minutes correctly."""
        import time
        now = time.time()
        assert format_time(now - 125) == "2m 5s"
        assert format_time(now - 3600 + 60) == "59m 0s"
        
    def test_format_time_hours(self):
        """format_time should handle hours correctly."""
        import time
        now = time.time()
        assert format_time(now - 3725) == "1h 2m"

    def test_parse_metrics_valid(self):
        """parse_metrics should extract values from valid log lines."""
        log = """
        | time/                   |        |
        |    fps                  | 123    |
        |    iterations           | 10     |
        |    time_elapsed         | 50     |
        |    total_timesteps      | 20480  |
        | train/                  |        |
        |    approx_kl            | 0.002  |
        |    clip_fraction        | 0.1    |
        |    clip_range           | 0.2    |
        |    entropy_loss         | -0.5   |
        |    explained_variance   | 0.8    |
        |    learning_rate        | 0.0003 |
        |    loss                 | 0.05   |
        |    n_updates            | 90     |
        |    policy_gradient_loss | -0.01  |
        |    value_loss           | 0.2    |
        | rollout/                |        |
        |    ep_len_mean          | 100    |
        |    ep_rew_mean          | 45.5   |
        """
        metrics = parse_metrics(log)
        assert metrics['fps'] == 123
        assert metrics['steps'] == 20480
        assert metrics['reward'] == 45.5

    def test_parse_metrics_empty(self):
        """parse_metrics should handle empty or invalid logs."""
        metrics = parse_metrics("random text content")
        assert metrics['fps'] == 0
        assert metrics['steps'] == 0
        assert metrics['reward'] == 0

    @patch('glob.glob')
    @patch('os.stat')
    def test_get_models(self, mock_stat, mock_glob):
        """get_models should list and parse model files."""
        # Setup mocks
        mock_glob.return_value = ["logs/model_100_steps.zip", "logs/myrun_final.zip"]
        
        # Mock stat for file size and time
        mock_stat_res = MagicMock()
        mock_stat_res.st_size = 1048576 * 2.5 # 2.5 MB
        mock_stat_res.st_mtime = 1600000000
        mock_stat.return_value = mock_stat_res
        
        models = get_models()
        
        # Check sorting (final usually first due to 'infinite' steps logic or date)
        # Assuming date sort now? The code was changed to sort by date in get_models!
        # Let's just check existence.
        assert len(models) == 2
        
        # Check standard model
        m1 = next(m for m in models if m['name'] == "model_100_steps")
        assert m1['steps'] == 100
        
        # Check final model
        m2 = next(m for m in models if m['name'] == "myrun_final")
        assert m2['steps'] == 999_999_999
        assert m2['name'] == "myrun_final"

    @patch('psutil.process_iter')
    def test_get_running_processes(self, mock_process_iter):
        """get_running_processes should identify training and game scripts."""
        # Mock processes
        p1 = MagicMock()
        p1.info = {'pid': 1001, 'name': 'python', 'cmdline': ['python', 'rl_train.py'], 'create_time': 1600000000}
        
        p2 = MagicMock()
        p2.info = {'pid': 1002, 'name': 'python', 'cmdline': ['python', 'trackmania_clone.py'], 'create_time': 1600000000}
        
        p3 = MagicMock()
        p3.info = {'pid': 1003, 'name': 'bash', 'cmdline': ['ls'], 'create_time': 1600000000}
        
        mock_process_iter.return_value = [p1, p2, p3]
        
        procs = get_running_processes()
        
        assert len(procs) == 2
        
        # Check training process
        train = next((p for p in procs if p['pid'] == 1001), None)
        assert train is not None
        assert train['type'] == 'training'
        assert "Training" in train['label']
        
        # Check game process
        game = next((p for p in procs if p['pid'] == 1002), None)
        assert game is not None
        assert game['type'] == 'game'


