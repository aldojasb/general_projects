import pytest
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl.cstr.optimization.base_agent import ActorCriticNet, collect_trajectories, compute_gae


# ===== PYTEST FIXTURES =====
# Fixtures provide reusable test data and setup for multiple test functions

@pytest.fixture
def cstr_state_dim() -> int:
    """
    Fixture providing the state dimension for CSTR (Continuous Stirred Tank Reactor).
    
    CSTR state typically includes:
    - Ca: Concentration of reactant A
    - Cb: Concentration of reactant B  
    - T: Temperature
    
    Returns:
        int: State dimension (3 for CSTR)
    """
    return 3


@pytest.fixture
def cstr_action_dim() -> int:
    """
    Fixture providing the action dimension for CSTR control.
    
    CSTR action typically includes:
    - Cooling jacket temperature adjustment
    
    Returns:
        int: Action dimension (1 for CSTR)
    """
    return 1


@pytest.fixture
def sample_cstr_states(cstr_state_dim: int) -> torch.Tensor:
    """
    Fixture providing sample CSTR state tensors for testing.
    
    Creates realistic CSTR state data:
    - Ca: Concentration A (0.0 to 1.0)
    - Cb: Concentration B (0.0 to 1.0)
    - T: Temperature (300K to 400K)
    
    Args:
        cstr_state_dim: State dimension from fixture
        
    Returns:
        torch.Tensor: Batch of CSTR states with shape [batch_size, state_dim]
    """
    # Create realistic CSTR states: [Ca, Cb, T]
    states = torch.tensor([
        [0.8, 0.2, 350.0],  # High A, low B, moderate temp
        [0.5, 0.5, 340.0],  # Balanced concentrations, cooler
        [0.2, 0.8, 360.0],  # Low A, high B, warmer
        [0.9, 0.1, 330.0],  # Very high A, very low B, cool
    ], dtype=torch.float32)
    
    return states


@pytest.fixture
def mock_cstr_environment():
    """
    Fixture providing a mock CSTR environment for testing collect_trajectories.
    
    Creates a simple mock environment that:
    - Returns realistic CSTR states
    - Accepts temperature adjustment actions
    - Returns appropriate rewards and done flags
    - Simulates reactor dynamics
    
    Returns:
        MockEnvironment: A mock CSTR environment for testing
    """
    class MockCSTREnvironment:
        def __init__(self):
            self.step_count = 0
            self.max_steps = 100
            self.current_state = np.array([0.8, 0.2, 350.0])  # Initial CSTR state
            
        def reset(self):
            """Reset environment to initial state."""
            self.step_count = 0
            self.current_state = np.array([0.8, 0.2, 350.0])
            return self.current_state.copy()
            
        def step(self, action):
            """Simulate one step in the CSTR environment."""
            # Convert action to numpy if it's a tensor
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            
            # Simple CSTR dynamics simulation
            temp_adjustment = action[0] if isinstance(action, (list, np.ndarray)) else action
            
            # Update temperature (bounded between 300K and 400K)
            new_temp = np.clip(self.current_state[2] + temp_adjustment, 300.0, 400.0)
            
            # Simple concentration dynamics
            # Reactant A decreases, B increases (simplified reaction)
            new_ca = np.clip(self.current_state[0] - 0.01, 0.0, 1.0)
            new_cb = np.clip(self.current_state[1] + 0.01, 0.0, 1.0)
            
            # Update state
            self.current_state = np.array([new_ca, new_cb, new_temp])
            
            # Calculate reward based on conversion efficiency and safety
            conversion_efficiency = new_cb  # Higher B = better conversion
            safety_penalty = 0.0 if 300 <= new_temp <= 400 else -10.0  # Safety bounds
            
            reward = conversion_efficiency + safety_penalty
            
            # Episode ends if unsafe conditions or max steps reached
            self.step_count += 1
            done = (new_temp < 300 or new_temp > 400 or self.step_count >= self.max_steps)
            
            return self.current_state.copy(), reward, done, {}
    
    return MockCSTREnvironment()


@pytest.fixture
def sample_trajectory_data():
    """
    Fixture providing sample trajectory data for testing compute_gae.
    
    Creates realistic trajectory data that would be returned by collect_trajectories:
    - rewards: Conversion efficiency rewards from CSTR control
    - dones: Episode termination flags
    - values: Critic's value estimates for each state
    
    Returns:
        tuple: (rewards, dones, values) for GAE computation
    """
    # Sample trajectory data (10 timesteps)
    rewards = [15.2, 12.8, 18.1, 14.5, 16.3, 13.7, 17.9, 15.8, 14.2, 16.7]
    dones = [False, False, False, False, False, False, False, False, False, True]
    values = [15.0, 13.0, 17.5, 14.0, 16.0, 13.5, 17.0, 15.5, 14.0, 16.5]
    
    return rewards, dones, values


@pytest.fixture
def short_trajectory_data():
    """
    Fixture providing short trajectory data for testing edge cases.
    
    Creates a short trajectory (3 timesteps) to test GAE with minimal data.
    
    Returns:
        tuple: (rewards, dones, values) for GAE computation
    """
    rewards = [10.0, 12.0, 8.0]
    dones = [False, False, True]
    values = [10.5, 11.5, 8.5]
    
    return rewards, dones, values


@pytest.fixture
def batch_size() -> int:
    """
    Fixture providing batch size for testing.
    
    Returns:
        int: Batch size for testing (4 samples)
    """
    return 4


class TestActorCriticNet:
    """Test suite for ActorCriticNet class from base_agent.py"""
    
    def test_model_initialization(self, cstr_state_dim: int, cstr_action_dim: int):
        """
        Test that ActorCriticNet initializes correctly with proper architecture.
        """
        # Given: Valid state and action dimensions for CSTR
        state_dim = cstr_state_dim  # 3 for CSTR: [Ca, Cb, T]
        action_dim = cstr_action_dim  # 1 for cooling temperature adjustment
        
        # When: Creating a new ActorCriticNet instance
        model = ActorCriticNet(state_dim=state_dim, action_dim=action_dim)
        
        # Then: Model should have correct architecture and trainable parameters
        # Check that model is an instance of ActorCriticNet
        assert isinstance(model, ActorCriticNet)
        # Check that model is a PyTorch Module
        assert isinstance(model, torch.nn.Module)
        # Check that mean_head has correct output dimension
        assert model.mean_head.out_features == action_dim
        # Check that log_std parameter has correct shape
        assert model.log_std.shape == (action_dim,)
        # Check that critic network has correct architecture
        assert model.critic[0].in_features == state_dim
        # Check that model has trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0, "Model should have trainable parameters"
        # Check that log_std is initialized to zeros
        assert torch.allclose(model.log_std, torch.zeros(action_dim)), \
            "log_std should be initialized to zeros"

    def test_forward_pass_output_shapes(self,
    cstr_state_dim: int,
    cstr_action_dim: int, 
    sample_cstr_states: torch.Tensor,
    batch_size: int):
        """
        Test that forward pass returns correct output shapes for CSTR control.
        
        """
        # Given: A properly initialized model and sample CSTR states
        state_dim = cstr_state_dim  # 3 for CSTR: [Ca, Cb, T]
        action_dim = cstr_action_dim  # 1 for cooling temperature adjustment
        states = sample_cstr_states  # Shape: [batch_size, 3] for CSTR states
        model = ActorCriticNet(state_dim=state_dim, action_dim=action_dim)
        
        # When: Performing a forward pass through the model
        action_mean, action_std, state_value = model(states)
        
        # Then: Outputs should have correct shapes for actor and critic components
        # Check action_mean shape: [batch_size, action_dim]
        expected_mean_shape = (batch_size, cstr_action_dim)
        assert action_mean.shape == expected_mean_shape, \
            f"Expected action_mean shape {expected_mean_shape}, got {action_mean.shape}"
        # Check action_std shape: [action_dim] (same for all batch samples)
        expected_std_shape = (cstr_action_dim,)
        assert action_std.shape == expected_std_shape, \
            f"Expected action_std shape {expected_std_shape}, got {action_std.shape}"
        # Check state_value shape: [batch_size, 1]
        expected_value_shape = (batch_size, 1)
        assert state_value.shape == expected_value_shape, \
            f"Expected state_value shape {expected_value_shape}, got {state_value.shape}"

    def test_action_std_clamping(self,
    cstr_state_dim: int,
    cstr_action_dim: int,  
    sample_cstr_states: torch.Tensor):
        """
        Test that action standard deviation is properly clamped for numerical stability.
        """
        # Given: A model with log_std parameter that can be modified
        state_dim = cstr_state_dim  # 3 for CSTR: [Ca, Cb, T]
        action_dim = cstr_action_dim  # 1 for cooling temperature adjustment
        states = sample_cstr_states  # Shape: [batch_size, 3] for CSTR states
        model = ActorCriticNet(state_dim=state_dim, action_dim=action_dim)
        
        # When: Setting extreme log_std values and performing forward pass
        # Test extreme negative log_std (should be clamped to -20)
        model.log_std.data.fill_(-50.0)  # Extreme negative value
        _, action_std_negative, _ = model(states)

        #Then: Action std should be clamped to reasonable range [exp(-20), exp(2)]
        expected_min_std = torch.exp(torch.tensor(-20.0))
        assert torch.all(action_std_negative >= expected_min_std), \
            f"Action std should be clamped to minimum {expected_min_std}"
        # Test extreme positive log_std (should be clamped to 2)
        model.log_std.data.fill_(10.0)  # Extreme positive value
        _, action_std_positive, _ = model(states)
        expected_max_std = torch.exp(torch.tensor(2.0))
        assert torch.all(action_std_positive <= expected_max_std), \
            f"Action std should be clamped to maximum {expected_max_std}"

    def test_forward_pass_deterministic_outputs(self,
    cstr_state_dim: int,
    cstr_action_dim: int,  
    sample_cstr_states: torch.Tensor):
        """
        Test that model outputs are consistent for same input but different between calls.
        Why This Test Is Important:
        1. Debugging:
        If outputs aren't identical, something is wrong with the model
        Could indicate initialization issues, numerical instability, or bugs
        2. Reproducibility:
        Ensures model behavior is predictable
        Critical for debugging and validation
        3. Training Stability:
        Deterministic forward pass is essential for stable training
        Stochastic behavior should only come from sampling, not computation
        """
        # Given: A model and fixed input states
        state_dim = cstr_state_dim  # 3 for CSTR: [Ca, Cb, T]
        action_dim = cstr_action_dim  # 1 for cooling temperature adjustment
        model = ActorCriticNet(state_dim=state_dim, action_dim=action_dim)
        states = sample_cstr_states
        
        # When: Performing multiple forward passes with same input
        mean1, std1, value1 = model(states)
        mean2, std2, value2 = model(states)
        
        # Then: Action means and state values should be identical, but std should be consistent
        # Check that action means are identical (deterministic)
        assert torch.allclose(mean1, mean2), \
            "Action means should be identical for same input"
        # Check that state values are identical (deterministic)
        assert torch.allclose(value1, value2), \
            "State values should be identical for same input"
        # Check that action stds are identical (deterministic)
        assert torch.allclose(std1, std2), \
            "Action stds should be identical for same input"

    def test_gradient_flow(self,
    cstr_state_dim: int,
    cstr_action_dim: int,  
    sample_cstr_states: torch.Tensor):
        """
        Test that gradients flow properly through the network for training.
        """
        # Given: A model and input states
        state_dim = cstr_state_dim  # 3 for CSTR: [Ca, Cb, T]
        action_dim = cstr_action_dim  # 1 for cooling temperature adjustment
        model = ActorCriticNet(state_dim=state_dim, action_dim=action_dim)
        states = sample_cstr_states
        
        # When: Computing a dummy loss and performing backward pass
        mean, std, value = model(states)
        # Create a dummy loss (sum of all outputs)
        dummy_loss = mean.sum() + std.sum() + value.sum()
        # Perform backward pass
        dummy_loss.backward()
        
        # Then: Gradients should be computed for all trainable parameters
        for name, param in model.named_parameters():
            assert param.grad is not None, \
                f"Gradient should be computed for parameter {name}"
            assert not torch.isnan(param.grad).any(), \
                f"Gradient should not contain NaN values for parameter {name}"

    def test_model_cpu_compatibility(self, cstr_state_dim: int, cstr_action_dim: int):
        """
        Test that model works correctly on CPU device.
        """
        # Given: Model initialization parameters
        state_dim = cstr_state_dim
        action_dim = cstr_action_dim
        
        # When: Creating model and moving to CPU device
        model = ActorCriticNet(state_dim=state_dim, action_dim=action_dim)
        model_cpu = model.to('cpu')
        states_cpu = torch.randn(2, state_dim, device='cpu')
        mean_cpu, std_cpu, value_cpu = model_cpu(states_cpu)
        
        # Then: Model should work correctly on CPU with all outputs on CPU
        # Check that all outputs are on CPU device
        assert mean_cpu.device.type == 'cpu', \
            "Action mean should be on CPU device"
        assert std_cpu.device.type == 'cpu', \
            "Action std should be on CPU device"
        assert value_cpu.device.type == 'cpu', \
            "State value should be on CPU device"
        
        # Check that outputs have correct shapes
        assert mean_cpu.shape == (2, action_dim), \
            "Action mean should have correct shape on CPU"
        assert std_cpu.shape == (action_dim,), \
            "Action std should have correct shape on CPU"
        assert value_cpu.shape == (2, 1), \
            "State value should have correct shape on CPU"

    def test_model_cuda_compatibility(self, cstr_state_dim: int, cstr_action_dim: int):
        """
        Test that model works correctly on CUDA device (if available).
        """
        # Given: Model initialization parameters and CUDA availability
        state_dim = cstr_state_dim
        action_dim = cstr_action_dim
        
        # Skip test if CUDA is not available
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available - skipping CUDA compatibility test")
        
        # When: Creating model and moving to CUDA device
        model = ActorCriticNet(state_dim=state_dim, action_dim=action_dim)
        model_cuda = model.to('cuda')
        states_cuda = torch.randn(2, state_dim, device='cuda')
        mean_cuda, std_cuda, value_cuda = model_cuda(states_cuda)
        
        # Then: Model should work correctly on CUDA with all outputs on CUDA
        # Check that all outputs are on CUDA device
        assert mean_cuda.device.type == 'cuda', \
            "Action mean should be on CUDA device"
        assert std_cuda.device.type == 'cuda', \
            "Action std should be on CUDA device"
        assert value_cuda.device.type == 'cuda', \
            "State value should be on CUDA device"
        # Check that outputs have correct shapes
        assert mean_cuda.shape == (2, action_dim), \
            "Action mean should have correct shape on CUDA"
        assert std_cuda.shape == (action_dim,), \
            "Action std should have correct shape on CUDA"
        assert value_cuda.shape == (2, 1), \
            "State value should have correct shape on CUDA"

    def test_model_output_ranges(self,
    cstr_state_dim: int,
    cstr_action_dim: int, 
    sample_cstr_states: torch.Tensor):
        """
        Test that model outputs are within reasonable ranges for CSTR control.
        """
        # Given: A model and realistic CSTR states
        state_dim = cstr_state_dim  # 3 for CSTR: [Ca, Cb, T]
        action_dim = cstr_action_dim  # 1 for cooling temperature adjustment
        model = ActorCriticNet(state_dim=state_dim, action_dim=action_dim)
        states = sample_cstr_states
        
        # When: Performing forward pass
        action_mean, action_std, state_value = model(states)
        
        # Then: Outputs should be within reasonable ranges for CSTR control
        # Action mean should be finite (not NaN or inf)
        assert torch.isfinite(action_mean).all(), \
            "Action mean should be finite"
        # Action std should be positive and finite
        assert torch.all(action_std > 0), \
            "Action std should be positive"
        assert torch.isfinite(action_std).all(), \
            "Action std should be finite"
        # State value should be finite
        assert torch.isfinite(state_value).all(), \
            "State value should be finite"
        # Action std should be reasonable for CSTR control (not too large)
        max_reasonable_std = 10.0  # Maximum reasonable std for temperature control
        assert torch.all(action_std <= max_reasonable_std), \
            f"Action std should not exceed {max_reasonable_std}"

    def test_model_with_different_batch_sizes(self,
    cstr_state_dim: int,
    cstr_action_dim: int):
        """
        Test that model handles different batch sizes correctly.
        """
        # Given: A model and different batch sizes
        state_dim = cstr_state_dim  # 3 for CSTR: [Ca, Cb, T]
        action_dim = cstr_action_dim  # 1 for cooling temperature adjustment
        model = ActorCriticNet(state_dim=state_dim, action_dim=action_dim)
        
        # Test different batch sizes
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            # When: Performing forward pass with different batch sizes
            states = torch.randn(batch_size, state_dim)
            mean, std, value = model(states)
            
            # Then: Model should handle all batch sizes correctly
            assert mean.shape == (batch_size, action_dim), \
                f"Action mean should have shape ({batch_size}, {action_dim})"
            assert std.shape == (action_dim,), \
                f"Action std should have shape ({action_dim},)"
            assert value.shape == (batch_size, 1), \
                f"State value should have shape ({batch_size}, 1)"


class TestCollectTrajectories:
    """Test suite for collect_trajectories function from base_agent.py"""
    
    def test_collect_trajectories_length_and_shapes(self, 
                                                   cstr_state_dim: int,
                                                   cstr_action_dim: int,
                                                   mock_cstr_environment):
        """
        Test that collect_trajectories function returns correct lengths and shapes.
        """
        # Given: A properly initialized model and mock environment
        state_dim = cstr_state_dim  # 3 for CSTR: [Ca, Cb, T]
        action_dim = cstr_action_dim  # 1 for cooling temperature adjustment
        model = ActorCriticNet(state_dim=state_dim, action_dim=action_dim)
        env = mock_cstr_environment
        
        # When: Collecting trajectories with specific step count
        steps = 10
        states, actions, rewards, dones, values, log_probs = collect_trajectories(model, env, steps=steps)
        
        # Then: All trajectory components should have correct lengths and shapes
        # Check that all returned lists have the expected length
        assert len(states) == steps, \
            f"Expected {steps} states, got {len(states)}"
        assert len(actions) == steps, \
            f"Expected {steps} actions, got {len(actions)}"
        assert len(rewards) == steps, \
            f"Expected {steps} rewards, got {len(rewards)}"
        assert len(dones) == steps, \
            f"Expected {steps} dones, got {len(dones)}"
        assert len(values) == steps, \
            f"Expected {steps} values, got {len(values)}"
        assert len(log_probs) == steps, \
            f"Expected {steps} log_probs, got {len(log_probs)}"
        
        # Check that states are numpy arrays with correct shape
        for i, state in enumerate(states):
            assert isinstance(state, np.ndarray), \
                f"State {i} should be numpy array, got {type(state)}"
            assert state.shape == (state_dim,), \
                f"State {i} should have shape ({state_dim},), got {state.shape}"
        
        # Check that actions are numpy arrays with correct shape
        for i, action in enumerate(actions):
            assert isinstance(action, np.ndarray), \
                f"Action {i} should be numpy array, got {type(action)}"
            assert action.shape == (action_dim,), \
                f"Action {i} should have shape ({action_dim},), got {action.shape}"

    def test_collect_trajectories_data_types(self, 
                                           cstr_state_dim: int,
                                           cstr_action_dim: int,
                                           mock_cstr_environment):
        """
        Test that collect_trajectories function returns correct data types.
        """
        # Given: A properly initialized model and mock environment
        state_dim = cstr_state_dim  # 3 for CSTR: [Ca, Cb, T]
        action_dim = cstr_action_dim  # 1 for cooling temperature adjustment
        model = ActorCriticNet(state_dim=state_dim, action_dim=action_dim)
        env = mock_cstr_environment
        
        # When: Collecting trajectories with default parameters
        states, actions, rewards, dones, values, log_probs = collect_trajectories(model, env, steps=10)
        
        # Then: All trajectory components should have correct data types
        # Check that rewards are numeric values
        for i, reward in enumerate(rewards):
            assert isinstance(reward, (int, float)), \
                f"Reward {i} should be numeric, got {type(reward)}"
        
        # Check that dones are boolean values
        for i, done in enumerate(dones):
            assert isinstance(done, bool), \
                f"Done {i} should be boolean, got {type(done)}"
        
        # Check that values are numeric values
        for i, value in enumerate(values):
            assert isinstance(value, (int, float)), \
                f"Value {i} should be numeric, got {type(value)}"
        
        # Check that log_probs are numeric values
        for i, log_prob in enumerate(log_probs):
            assert isinstance(log_prob, (int, float)), \
                f"Log probability {i} should be numeric, got {type(log_prob)}"
        
        # Check that states are numpy arrays
        for i, state in enumerate(states):
            assert isinstance(state, np.ndarray), \
                f"State {i} should be numpy array, got {type(state)}"
        
        # Check that actions are numpy arrays
        for i, action in enumerate(actions):
            assert isinstance(action, np.ndarray), \
                f"Action {i} should be numpy array, got {type(action)}"

    def test_collect_trajectories_episode_reset(self, 
                                              cstr_state_dim: int,
                                              cstr_action_dim: int,
                                              mock_cstr_environment):
        """
        Test that collect_trajectories handles episode resets correctly.
        """
        # Given: A model and environment that can trigger episode resets
        state_dim = cstr_state_dim  # 3 for CSTR: [Ca, Cb, T]
        action_dim = cstr_action_dim  # 1 for cooling temperature adjustment
        model = ActorCriticNet(state_dim=state_dim, action_dim=action_dim)
        env = mock_cstr_environment
        
        # Modify environment to trigger episode end after few steps
        env.max_steps = 5  # Short episodes to test reset behavior
        
        # When: Collecting trajectories with episodes that end
        states, actions, rewards, dones, values, log_probs = collect_trajectories(
            model,
            env,
            steps=20)
        
        # Then: Function should handle resets and continue collecting data
        # Should collect all requested steps despite episode resets
        assert len(states) == 20, \
            "Should collect all requested steps despite episode resets"
        
        # Should have some done=True flags (episode endings)
        assert any(dones), \
            "Should have some episode endings (done=True)"
        
        # Should have some done=False flags (episode continuations)
        assert not all(dones), \
            "Should have some episode continuations (done=False)"

    def test_collect_trajectories_no_gradients(self, 
                                             cstr_state_dim: int,
                                             cstr_action_dim: int,
                                             mock_cstr_environment):
        """
        Test that collect_trajectories runs without gradient computation.
        """
        # Given: A model and environment
        state_dim = cstr_state_dim  # 3 for CSTR: [Ca, Cb, T]
        action_dim = cstr_action_dim  # 1 for cooling temperature adjustment
        model = ActorCriticNet(state_dim=state_dim, action_dim=action_dim)
        env = mock_cstr_environment
        
        # When: Collecting trajectories
        states, actions, rewards, dones, values, log_probs = collect_trajectories(
            model, env, steps=10)
        
        # Then: Function should run without accumulating gradients
        # Check that no gradients were computed
        for param in model.parameters():
            assert param.grad is None, \
                "No gradients should be computed during trajectory collection"
        
        # Verify that trajectory data was collected successfully
        assert len(states) == 10, \
            "Should collect trajectory data successfully without gradients"

    def test_collect_trajectories_action_sampling(self, 
                                                cstr_state_dim: int,
                                                cstr_action_dim: int,
                                                mock_cstr_environment):
        """
        Test that collect_trajectories performs proper action sampling.
        """
        # Given: A model and environment
        state_dim = cstr_state_dim  # 3 for CSTR: [Ca, Cb, T]
        action_dim = cstr_action_dim  # 1 for cooling temperature adjustment
        model = ActorCriticNet(state_dim=state_dim, action_dim=action_dim)
        env = mock_cstr_environment
        
        # When: Collecting multiple trajectories
        states1, actions1, rewards1, dones1, values1, log_probs1 = collect_trajectories(model, env, steps=10)
        states2, actions2, rewards2, dones2, values2, log_probs2 = collect_trajectories(model, env, steps=10)
        
        # Then: Actions should be sampled from the policy distribution
        # Actions should be different between runs (stochastic sampling)
        actions_different = any(not np.allclose(a1, a2) for a1, a2 in zip(actions1, actions2))
        assert actions_different, \
            "Actions should be different between runs due to stochastic sampling"
        
        # Log probabilities should be finite and reasonable
        for log_prob in log_probs1 + log_probs2:
            assert np.isfinite(log_prob), \
                "Log probabilities should be finite"
            assert -20 < log_prob < 20, \
                f"Log probabilities should be reasonable, got {log_prob}"

    def test_collect_trajectories_value_estimation(self, 
                                                 cstr_state_dim: int,
                                                 cstr_action_dim: int,
                                                 mock_cstr_environment):
        """
        Test that collect_trajectories provides proper value estimates.
        
        Given: A model and environment
        When: Collecting trajectories
        Then: Value estimates should be consistent with the critic network
        """
        # Given: A model and environment
        state_dim = cstr_state_dim  # 3 for CSTR: [Ca, Cb, T]
        action_dim = cstr_action_dim  # 1 for cooling temperature adjustment
        model = ActorCriticNet(state_dim=state_dim, action_dim=action_dim)
        env = mock_cstr_environment
        
        # When: Collecting trajectories
        states, actions, rewards, dones, values, log_probs = collect_trajectories(model, env, steps=10)
        
        # Then: Value estimates should be consistent with the critic network
        # Check that values are finite
        for value in values:
            assert np.isfinite(value), \
                "Value estimates should be finite"
        
        # Check that values are reasonable (not extreme)
        for value in values:
            assert -100 < value < 100, \
                f"Value estimates should be reasonable, got {value}"
        
        # Verify that values match critic predictions for the same states
        for i, state in enumerate(states):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                _, _, predicted_value = model(state_tensor)
                predicted_value = predicted_value.item()
            
            # Values should be close (allowing for small numerical differences)
            assert abs(values[i] - predicted_value) < 1e-6, \
                f"Value estimate should match critic prediction, got {values[i]} vs {predicted_value}"


class TestComputeGAE:
    """Test suite for compute_gae function from base_agent.py"""
    
    def test_compute_gae_basic(self, sample_trajectory_data):
        """
        Test that compute_gae computes GAE correctly for a simple trajectory.
        """
        # Given: Sample trajectory data (10 timesteps)
        rewards, dones, values = sample_trajectory_data
        
        # When: Computing GAE with default parameters
        gae_lambda = 0.95
        gae_values, gae_advantages = compute_gae(rewards, dones, values, gae_lambda)
        
        # Then: GAE values and advantages should be computed correctly
        # Check that gae_values has the correct length
        assert len(gae_values) == len(rewards), \
            f"gae_values length mismatch: expected {len(rewards)}, got {len(gae_values)}"
        # Check that gae_advantages has the correct length
        assert len(gae_advantages) == len(rewards), \
            f"gae_advantages length mismatch: expected {len(rewards)}, got {len(gae_advantages)}"
        
        # Check that gae_values are finite
        assert torch.all(torch.isfinite(gae_values)), \
            "gae_values should be finite"
        # Check that gae_advantages are finite
        assert torch.all(torch.isfinite(gae_advantages)), \
            "gae_advantages should be finite"
        
        # Check that gae_values and gae_advantages are not all zeros
        assert torch.any(gae_values != 0), \
            "gae_values should not be all zeros"
        assert torch.any(gae_advantages != 0), \
            "gae_advantages should not be all zeros"

    def test_compute_gae_short_trajectory(self, short_trajectory_data):
        """
        Test that compute_gae handles a short trajectory correctly.
        """
        # Given: Short trajectory data (3 timesteps)
        rewards, dones, values = short_trajectory_data
        
        # When: Computing GAE with default parameters
        gae_lambda = 0.95
        gae_values, gae_advantages = compute_gae(rewards, dones, values, gae_lambda)
        
        # Then: GAE values and advantages should be computed correctly
        # Check that gae_values has the correct length
        assert len(gae_values) == len(rewards), \
            f"gae_values length mismatch: expected {len(rewards)}, got {len(gae_values)}"
        # Check that gae_advantages has the correct length
        assert len(gae_advantages) == len(rewards), \
            f"gae_advantages length mismatch: expected {len(rewards)}, got {len(gae_advantages)}"
        
        # Check that gae_values are finite
        assert torch.all(torch.isfinite(gae_values)), \
            "gae_values should be finite"
        # Check that gae_advantages are finite
        assert torch.all(torch.isfinite(gae_advantages)), \
            "gae_advantages should be finite"
        
        # Check that gae_values and gae_advantages are not all zeros
        assert torch.any(gae_values != 0), \
            "gae_values should not be all zeros"
        assert torch.any(gae_advantages != 0), \
            "gae_advantages should not be all zeros"

        assert torch.all(gae_advantages == 0), \
            "gae_advantages should be all zeros for zero rewards"

    def test_compute_gae_normalization(self, sample_trajectory_data):
        """
        Test that compute_gae properly normalizes advantages to mean=0, std=1.
        
        This test verifies that the GAE function correctly normalizes advantages
        as specified in the implementation: (advantages - mean) / (std + 1e-8)
        """
        # Given: Sample trajectory data (10 timesteps)
        rewards, dones, values = sample_trajectory_data
        
        # When: Computing GAE with default parameters
        gamma = 0.99
        gae_lambda = 0.95
        gae_advantages, expected_future_rewards = compute_gae(rewards, dones, values, gamma, gae_lambda)
        
        # Then: Advantages should be properly normalized
        # Check that advantages have mean=0 (within numerical precision)
        advantages_mean = gae_advantages.mean().item()
        assert abs(advantages_mean) < 1e-6, \
            f"Normalized advantages should have mean=0, got {advantages_mean}"
        
        # Check that advantages have std=1 (within numerical precision)
        advantages_std = gae_advantages.std().item()
        assert abs(advantages_std - 1.0) < 1e-6, \
            f"Normalized advantages should have std=1, got {advantages_std}"
        
        # Check that advantages are finite
        assert torch.all(torch.isfinite(gae_advantages)), \
            "Normalized advantages should be finite"
        
        # Check that advantages are not all zeros (unless input was all zeros)
        assert torch.any(gae_advantages != 0), \
            "Normalized advantages should not be all zeros for non-zero input"
        
        # Check that advantages have reasonable range (should be mostly within ±3 std)
        advantages_abs = torch.abs(gae_advantages)
        assert torch.all(advantages_abs < 10), \
            "Normalized advantages should be within reasonable range"

    # def test_compute_gae_lambda_0(self, sample_trajectory_data):
    #     """
    #     Test that compute_gae with lambda=0.0 computes Time Difference TD(0) advantages.
        
    #     When λ=0.0, GAE reduces to Time Difference TD(0) advantages:
    #     - Advantages = rewards + γ*V(next_state) - V(current_state)
    #     - Returns = advantages + V(current_state)
    #     """
    #     # Given: Sample trajectory data (10 timesteps)
    #     rewards, dones, values = sample_trajectory_data
        
    #     # When: Computing GAE with lambda=0.0
    #     gamma = 0.99
    #     gae_lambda = 0.0
    #     gae_values, gae_advantages = compute_gae(rewards, dones, values, gamma, gae_lambda)
        
    #     # Then: GAE should compute TD(0) advantages correctly
    #     # Check that outputs have correct length
    #     assert len(gae_values) == len(rewards), \
    #         "gae_values length should match rewards length"
    #     assert len(gae_advantages) == len(rewards), \
    #         "gae_advantages length should match rewards length"
        
    #     # Check that outputs are finite
    #     assert torch.all(torch.isfinite(gae_values)), \
    #         "gae_values should be finite"
    #     assert torch.all(torch.isfinite(gae_advantages)), \
    #         "gae_advantages should be finite"
        
    #     # Check that advantages are normalized (mean close to 0, std close to 1)
    #     # The GAE function normalizes advantages to mean=0, std=1
    #     advantages_mean = gae_advantages.mean().item()
    #     advantages_std = gae_advantages.std().item()
        
    #     # Allow for small numerical precision errors
    #     assert abs(advantages_mean) < 1e-6, \
    #         f"Normalized advantages should have mean close to 0, got {advantages_mean}"
    #     assert abs(advantages_std - 1.0) < 1e-6, \
    #         f"Normalized advantages should have std close to 1, got {advantages_std}"
        
    #     # Check that returns are reasonable (should be close to rewards for λ=0.0)
    #     # Returns = advantages + values, so they should be in reasonable range
    #     assert torch.all(gae_values > -100) and torch.all(gae_values < 100), \
    #         "Returns should be in reasonable range"

    # def test_compute_gae_lambda_1(self, sample_trajectory_data):
    #     """
    #     Test that compute_gae with lambda=1.0 computes Monte Carlo advantages.
        
    #     When λ=1.0, GAE reduces to Monte Carlo advantages:
    #     - Uses actual returns from each point onward
    #     - More variance but less bias than TD(0)
    #     """
    #     # Given: Sample trajectory data (10 timesteps)
    #     rewards, dones, values = sample_trajectory_data
        
    #     # When: Computing GAE with lambda=1.0
    #     gamma = 0.99
    #     gae_lambda = 1.0
    #     gae_values, gae_advantages = compute_gae(rewards, dones, values, gamma, gae_lambda)
        
    #     # Then: GAE should compute Monte Carlo advantages correctly
    #     # Check that outputs have correct length
    #     assert len(gae_values) == len(rewards), \
    #         "gae_values length should match rewards length"
    #     assert len(gae_advantages) == len(rewards), \
    #         "gae_advantages length should match rewards length"
        
    #     # Check that outputs are finite
    #     assert torch.all(torch.isfinite(gae_values)), \
    #         "gae_values should be finite"
    #     assert torch.all(torch.isfinite(gae_advantages)), \
    #         "gae_advantages should be finite"
        
    #     # Check that advantages are normalized (mean close to 0, std close to 1)
    #     advantages_mean = gae_advantages.mean().item()
    #     advantages_std = gae_advantages.std().item()
    #     assert abs(advantages_mean) < 1e-6, \
    #         f"Normalized advantages should have mean close to 0, got {advantages_mean}"
    #     assert abs(advantages_std - 1.0) < 1e-6, \
    #         f"Normalized advantages should have std close to 1, got {advantages_std}"

    # def test_compute_gae_lambda_0_5(self, sample_trajectory_data):
    #     """
    #     Test that compute_gae with lambda=0.5 computes intermediate GAE.
        
    #     When λ=0.5, GAE is a weighted combination of TD(0) and Monte Carlo:
    #     - Balances bias and variance
    #     - Intermediate between λ=0.0 and λ=1.0
    #     """
    #     # Given: Sample trajectory data (10 timesteps)
    #     rewards, dones, values = sample_trajectory_data
        
    #     # When: Computing GAE with lambda=0.5
    #     gamma = 0.99
    #     gae_lambda = 0.5
    #     gae_values, gae_advantages = compute_gae(rewards, dones, values, gamma, gae_lambda)
        
    #     # Then: GAE should compute intermediate advantages correctly
    #     # Check that outputs have correct length
    #     assert len(gae_values) == len(rewards), \
    #         "gae_values length should match rewards length"
    #     assert len(gae_advantages) == len(rewards), \
    #         "gae_advantages length should match rewards length"
        
    #     # Check that outputs are finite
    #     assert torch.all(torch.isfinite(gae_values)), \
    #         "gae_values should be finite"
    #     assert torch.all(torch.isfinite(gae_advantages)), \
    #         "gae_advantages should be finite"
        
    #     # Check that advantages are normalized (mean close to 0, std close to 1)
    #     advantages_mean = gae_advantages.mean().item()
    #     advantages_std = gae_advantages.std().item()
    #     assert abs(advantages_mean) < 1e-6, \
    #         f"Normalized advantages should have mean close to 0, got {advantages_mean}"
    #     assert abs(advantages_std - 1.0) < 1e-6, \
    #         f"Normalized advantages should have std close to 1, got {advantages_std}"

    # def test_compute_gae_episode_termination(self, sample_trajectory_data):
    #     """
    #     Test that compute_gae handles episode termination correctly.
    #     """
    #     # Given: Sample trajectory data with episode termination
    #     rewards, dones, values = sample_trajectory_data
        
    #     # When: Computing GAE with default parameters
    #     gamma = 0.99
    #     gae_lambda = 0.95
    #     gae_values, gae_advantages = compute_gae(rewards, dones, values, gamma, gae_lambda)
        
    #     # Then: GAE should handle episode termination correctly
    #     # Check that the last timestep has a done flag
    #     assert dones[-1] == True, \
    #         "Last timestep should have done=True"
        
    #     # Check that gae_values and gae_advantages are computed correctly
    #     assert len(gae_values) == len(rewards), \
    #         "gae_values length should match rewards length"
    #     assert len(gae_advantages) == len(rewards), \
    #         "gae_advantages length should match rewards length"
        
    #     # Check that gae_values and gae_advantages are finite
    #     assert torch.all(torch.isfinite(gae_values)), \
    #         "gae_values should be finite"
    #     assert torch.all(torch.isfinite(gae_advantages)), \
    #         "gae_advantages should be finite"

    # def test_compute_gae_different_gamma(self, sample_trajectory_data):
    #     """
    #     Test that compute_gae works with different gamma values.
    #     """
    #     # Given: Sample trajectory data
    #     rewards, dones, values = sample_trajectory_data
        
    #     # When: Computing GAE with different gamma values
    #     gamma_0_9 = 0.9
    #     gamma_0_99 = 0.99
    #     gae_lambda = 0.95
        
    #     gae_values_0_9, gae_advantages_0_9 = compute_gae(rewards, dones, values, gamma_0_9, gae_lambda)
    #     gae_values_0_99, gae_advantages_0_99 = compute_gae(rewards, dones, values, gamma_0_99, gae_lambda)
        
    #     # Then: GAE should work with different gamma values
    #     # Check that both computations produce valid results
    #     assert torch.all(torch.isfinite(gae_values_0_9)), \
    #         "gae_values with gamma=0.9 should be finite"
    #     assert torch.all(torch.isfinite(gae_advantages_0_9)), \
    #         "gae_advantages with gamma=0.9 should be finite"
    #     assert torch.all(torch.isfinite(gae_values_0_99)), \
    #         "gae_values with gamma=0.99 should be finite"
    #     assert torch.all(torch.isfinite(gae_advantages_0_99)), \
    #         "gae_advantages with gamma=0.99 should be finite"
        
    #     # Check that results are different for different gamma values
    #     assert not torch.allclose(gae_values_0_9, gae_values_0_99), \
    #         "Different gamma values should produce different results"

    # def test_compute_gae_constant_rewards(self):
    #     """
    #     Test that compute_gae works with constant rewards.
    #     """
    #     # Given: Trajectory with constant rewards
    #     rewards = [10.0, 10.0, 10.0, 10.0, 10.0]
    #     dones = [False, False, False, False, True]
    #     values = [10.0, 10.0, 10.0, 10.0, 10.0]
        
    #     # When: Computing GAE with default parameters
    #     gamma = 0.99
    #     gae_lambda = 0.95
    #     gae_values, gae_advantages = compute_gae(rewards, dones, values, gamma, gae_lambda)
        
    #     # Then: GAE should handle constant rewards correctly
    #     # Check that gae_values and gae_advantages have correct length
    #     assert len(gae_values) == len(rewards), \
    #         "gae_values length should match rewards length"
    #     assert len(gae_advantages) == len(rewards), \
    #         "gae_advantages length should match rewards length"
        
    #     # Check that gae_values and gae_advantages are finite
    #     assert torch.all(torch.isfinite(gae_values)), \
    #         "gae_values should be finite"
    #     assert torch.all(torch.isfinite(gae_advantages)), \
    #         "gae_advantages should be finite"

    # def test_compute_gae_increasing_rewards(self):
    #     """
    #     Test that compute_gae works with increasing rewards.
    #     """
    #     # Given: Trajectory with increasing rewards
    #     rewards = [5.0, 10.0, 15.0, 20.0, 25.0]
    #     dones = [False, False, False, False, True]
    #     values = [5.0, 10.0, 15.0, 20.0, 25.0]
        
    #     # When: Computing GAE with default parameters
    #     gamma = 0.99
    #     gae_lambda = 0.95
    #     gae_values, gae_advantages = compute_gae(rewards, dones, values, gamma, gae_lambda)
        
    #     # Then: GAE should handle increasing rewards correctly
    #     # Check that gae_values and gae_advantages have correct length
    #     assert len(gae_values) == len(rewards), \
    #         "gae_values length should match rewards length"
    #     assert len(gae_advantages) == len(rewards), \
    #         "gae_advantages length should match rewards length"
        
    #     # Check that gae_values and gae_advantages are finite
    #     assert torch.all(torch.isfinite(gae_values)), \
    #         "gae_values should be finite"
    #     assert torch.all(torch.isfinite(gae_advantages)), \
    #         "gae_advantages should be finite"

    # def test_compute_gae_decreasing_rewards(self):
    #     """
    #     Test that compute_gae works with decreasing rewards.
    #     """
    #     # Given: Trajectory with decreasing rewards
    #     rewards = [25.0, 20.0, 15.0, 10.0, 5.0]
    #     dones = [False, False, False, False, True]
    #     values = [25.0, 20.0, 15.0, 10.0, 5.0]
        
    #     # When: Computing GAE with default parameters
    #     gamma = 0.99
    #     gae_lambda = 0.95
    #     gae_values, gae_advantages = compute_gae(rewards, dones, values, gamma, gae_lambda)
        
    #     # Then: GAE should handle decreasing rewards correctly
    #     # Check that gae_values and gae_advantages have correct length
    #     assert len(gae_values) == len(rewards), \
    #         "gae_values length should match rewards length"
    #     assert len(gae_advantages) == len(rewards), \
    #         "gae_advantages length should match rewards length"
        
    #     # Check that gae_values and gae_advantages are finite
    #     assert torch.all(torch.isfinite(gae_values)), \
    #         "gae_values should be finite"
    #     assert torch.all(torch.isfinite(gae_advantages)), \
    #         "gae_advantages should be finite"

    # def test_compute_gae_negative_rewards(self):
    #     """
    #     Test that compute_gae works with negative rewards.
    #     """
    #     # Given: Trajectory with negative rewards
    #     rewards = [-5.0, -10.0, -15.0, -20.0, -25.0]
    #     dones = [False, False, False, False, True]
    #     values = [-5.0, -10.0, -15.0, -20.0, -25.0]
        
    #     # When: Computing GAE with default parameters
    #     gamma = 0.99
    #     gae_lambda = 0.95
    #     gae_values, gae_advantages = compute_gae(rewards, dones, values, gamma, gae_lambda)
        
    #     # Then: GAE should handle negative rewards correctly
    #     # Check that gae_values and gae_advantages have correct length
    #     assert len(gae_values) == len(rewards), \
    #         "gae_values length should match rewards length"
    #     assert len(gae_advantages) == len(rewards), \
    #         "gae_advantages length should match rewards length"
        
    #     # Check that gae_values and gae_advantages are finite
    #     assert torch.all(torch.isfinite(gae_values)), \
    #         "gae_values should be finite"
    #     assert torch.all(torch.isfinite(gae_advantages)), \
    #         "gae_advantages should be finite"

    # def test_compute_gae_mixed_rewards(self):
    #     """
    #     Test that compute_gae works with mixed positive and negative rewards.
    #     """
    #     # Given: Trajectory with mixed rewards
    #     rewards = [5.0, -10.0, 15.0, -20.0, 25.0]
    #     dones = [False, False, False, False, True]
    #     values = [5.0, -10.0, 15.0, -20.0, 25.0]
        
    #     # When: Computing GAE with default parameters
    #     gamma = 0.99
    #     gae_lambda = 0.95
    #     gae_values, gae_advantages = compute_gae(rewards, dones, values, gamma, gae_lambda)
        
    #     # Then: GAE should handle mixed rewards correctly
    #     # Check that gae_values and gae_advantages have correct length
    #     assert len(gae_values) == len(rewards), \
    #         "gae_values length should match rewards length"
    #     assert len(gae_advantages) == len(rewards), \
    #         "gae_advantages length should match rewards length"
        
    #     # Check that gae_values and gae_advantages are finite
    #     assert torch.all(torch.isfinite(gae_values)), \
    #         "gae_values should be finite"
    #     assert torch.all(torch.isfinite(gae_advantages)), \
    #         "gae_advantages should be finite"

    # def test_compute_gae_zero_rewards(self):
    #     """
    #     Test that compute_gae works with zero rewards.
    #     """
    #     # Given: Trajectory with zero rewards
    #     rewards = [0.0, 0.0, 0.0, 0.0, 0.0]
    #     dones = [False, False, False, False, True]
    #     values = [0.0, 0.0, 0.0, 0.0, 0.0]
        
    #     # When: Computing GAE with default parameters
    #     gamma = 0.99
    #     gae_lambda = 0.95
    #     gae_values, gae_advantages = compute_gae(rewards, dones, values, gamma, gae_lambda)
        
    #     # Then: GAE should handle zero rewards correctly
    #     # Check that gae_values and gae_advantages have correct length
    #     assert len(gae_values) == len(rewards), \
    #         "gae_values length should match rewards length"
    #     assert len(gae_advantages) == len(rewards), \
    #         "gae_advantages length should match rewards length"
        
    #     # Check that gae_values and gae_advantages are finite
    #     assert torch.all(torch.isfinite(gae_values)), \
    #         "gae_values should be finite"
    #     assert torch.all(torch.isfinite(gae_advantages)), \
    #         "gae_advantages should be finite"
        
    #     # Check that gae_values and gae_advantages are all zeros
    #     assert torch.all(gae_values == 0), \
    #         "gae_values should be all zeros for zero rewards"
    #     assert torch.all(gae_advantages == 0), \
    #         "gae_advantages should be all zeros for zero rewards"
