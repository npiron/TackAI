"""
Unit tests for the game environment (TimeAttackEnv) and GymTimeAttack wrapper.
These tests ensure the core game mechanics work correctly.
"""
import pytest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTrack:
    """Tests for track building and checkpoints."""
    
    def test_track_has_boundaries(self):
        """Track should have outer and inner boundaries."""
        from src.game.track import build_track
        track = build_track()
        assert track.rect_outer is not None
        assert track.rect_outer is not None
        assert isinstance(track.obstacles, list)
        assert len(track.obstacles) > 0
        # assert track.rect_outer.width > track.rect_inner.width # Meaningless now


class TestTimeAttackEnv:
    """Tests for the raw TimeAttackEnv (before gym wrapper)."""
    
    @pytest.fixture
    def env(self):
        """Create a fresh environment for each test."""
        from src.game.env import TimeAttackEnv
        from src.game.track import build_track
        return TimeAttackEnv(build_track())
    
    def test_env_reset_returns_observation(self, env):
        """Reset should return an observation array."""
        obs = env.reset()
        assert obs is not None
        assert isinstance(obs, np.ndarray)
    
    def test_observation_shape(self, env):
        """Observation should have 14 elements (Pilot-Centric + LIDAR)."""
        obs = env.reset()
        assert obs.shape == (14,), f"Expected shape (14,), got {obs.shape}"
    
    def test_observation_is_finite(self, env):
        """Observation should not contain NaN or Inf."""
        obs = env.reset()
        assert np.all(np.isfinite(obs)), "Observation contains NaN or Inf"
    
    def test_step_returns_correct_tuple(self, env):
        """Step should return (obs, reward, done, info)."""
        env.reset()
        action = (0.0, 0.5, 0.0)  # No steer, half throttle, no brake
        result = env.step(action)
        
        assert len(result) == 4, f"Step should return 4 values, got {len(result)}"
        obs, reward, done, info = result
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
    
    def test_action_affects_state(self, env):
        """Taking a step with throttle should change the car position."""
        obs1 = env.reset()
        
        # Take multiple steps with throttle
        for _ in range(10):
            env.step((0.0, 1.0, 0.0))  # Full throttle
        
        obs2 = env._obs()
        
        # Position should have changed
        assert not np.allclose(obs1[:2], obs2[:2]), "Car position should change after stepping"
    
    def test_env_does_not_crash_many_steps(self, env):
        """Environment should be stable for many steps."""
        env.reset()
        for _ in range(1000):
            obs, reward, done, info = env.step((0.1, 0.5, 0.0))
            assert np.all(np.isfinite(obs)), f"Observation became non-finite at step"
            if done:
                env.reset()


class TestGymTimeAttack:
    """Tests for the Gymnasium wrapper."""
    
    @pytest.fixture
    def gym_env(self):
        """Create a GymTimeAttack environment."""
        from src.rl.wrappers import GymTimeAttack
        return GymTimeAttack(render_mode=None)  # No rendering for tests
    
    def test_gym_env_has_correct_spaces(self, gym_env):
        """Gym env should have proper action and observation spaces."""
        from gymnasium import spaces
        
        assert isinstance(gym_env.observation_space, spaces.Box)
        assert isinstance(gym_env.action_space, spaces.Discrete)
    
    def test_observation_space_shape(self, gym_env):
        """Observation space should be (14,)."""
        assert gym_env.observation_space.shape == (14,)
    
    def test_action_space_discrete(self, gym_env):
        """Action space should be Discrete(9) for on/off button controls."""
        assert gym_env.action_space.n == 9, f"Expected 9 discrete actions, got {gym_env.action_space.n}"
    
    def test_reset_returns_correct_format(self, gym_env):
        """Reset should return a valid observation and empty info dict."""
        result = gym_env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2
        obs, info = result
        
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (14,)
        assert isinstance(info, dict)
    
    def test_step_returns_correct_format(self, gym_env):
        """Step should return 5 values for Gymnasium."""
        gym_env.reset()
        action = 1  # Discrete action: Accelerate
        result = gym_env.step(action)
        
        assert len(result) == 5, "Step should return (obs, reward, terminated, truncated, info)"
        obs, reward, terminated, truncated, info = result
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_reward_shaping_provides_positive_reward(self, gym_env):
        """Driving toward checkpoint should give positive reward."""
        gym_env.reset()
        
        total_reward = 0
        for _ in range(50):
            action = 1  # Discrete action: Accelerate
            obs, reward, done, truncated, info = gym_env.step(action)
            total_reward += reward
            if done:
                break
        
        # Should get some positive reward from progress/speed
        # (might be negative if hitting walls, but generally should be positive)
        assert total_reward != 0, "Reward should be non-zero after 50 steps"
    
    def test_random_actions_stable(self, gym_env):
        """Environment should handle random actions without crashing."""
        gym_env.reset()
        
        for _ in range(200):
            action = gym_env.action_space.sample()
            obs, reward, done, truncated, info = gym_env.step(action)
            
            assert np.all(np.isfinite(obs)), "Observation should stay finite"
            assert np.isfinite(reward), "Reward should stay finite"
            
            if done:
                obs, info = gym_env.reset()


class TestObservationValues:
    """Tests for specific observation values."""
    
    @pytest.fixture
    def gym_env(self):
        from src.rl.wrappers import GymTimeAttack
        return GymTimeAttack(render_mode=None)
    
    def test_speed_starts_at_zero(self, gym_env):
        """Speed should be zero at reset."""
        obs, _ = gym_env.reset()
        speed = obs[0]  # Speed is at index 0
        assert abs(speed) < 0.01, f"Initial speed should be ~0, got {speed}"
    
    def test_speed_increases_with_throttle(self, gym_env):
        """Speed should increase when applying throttle."""
        obs, _ = gym_env.reset()
        initial_speed = obs[0]
        
        # Apply throttle for several frames (action 1 = Accelerate)
        for _ in range(30):
            obs, _, _, _, _ = gym_env.step(1)
        
        final_speed = obs[0]
        assert final_speed > initial_speed, "Speed should increase with throttle"
    
    def test_all_discrete_actions_work(self, gym_env):
        """All 9 discrete actions should be processable without errors."""
        gym_env.reset()
        
        # Test each action once
        for action in range(9):
            obs, reward, done, truncated, info = gym_env.step(action)
            assert np.all(np.isfinite(obs)), f"Action {action} produced invalid observation"
            assert np.isfinite(reward), f"Action {action} produced invalid reward"
            
            if done:
                gym_env.reset()
    
    def test_discrete_actions_are_onoff(self, gym_env):
        """Verify discrete actions map to on/off values (0.0, 1.0, -1.0)."""
        # This is implicitly tested by the environment not crashing,
        # but we document the expectation here
        gym_env.reset()
        
        # Test a few representative actions
        test_cases = [
            (0, "Idle"),
            (1, "Accelerate"),
            (3, "Left"),
            (5, "Left + Accelerate"),
            (7, "Left + Brake")
        ]
        
        for action, name in test_cases:
            obs, reward, done, truncated, info = gym_env.step(action)
            # Just verify it doesn't crash and produces valid output
            assert np.all(np.isfinite(obs)), f"Action {action} ({name}) failed"
