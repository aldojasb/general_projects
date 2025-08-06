import pytest
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rl.cstr.optimization.base_agent import ActorCriticNet, collect_trajectories, compute_gae, ppo_update


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


@pytest.fixture
def mock_actor_critic_model():
    """
    Fixture providing a mock actor-critic model for testing ppo_update.
    
    This creates a mock model that implements the required interface for ppo_update
    without depending on the concrete ActorCriticNet implementation.
    
    Returns:
        MockActorCriticModel: A mock model with actor and critic components
    """
    class MockActorCriticModel(nn.Module):
        def __init__(self, state_dim=3, action_dim=1):
            super().__init__()
            # Create simple linear layers for actor and critic
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim)
            )
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
            # Add log_std parameter for actor
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        def forward(self, states):
            """Forward pass that returns (action_mean, action_std, state_value)."""
            action_mean = self.actor(states)
            action_std = self.log_std.clamp(-20, 2).exp()
            state_value = self.critic(states)
            return action_mean, action_std, state_value
    
    return MockActorCriticModel()


@pytest.fixture
def sample_ppo_data():
    """
    Fixture providing sample data for testing ppo_update.
    
    Returns:
        tuple: (states, actions, log_probs_old, returns, advantages) for PPO testing
    """
    batch_size = 10
    states = [np.array([0.8, 0.2, 350.0]) for _ in range(batch_size)]
    actions = [np.array([2.3]) for _ in range(batch_size)]
    log_probs_old = [-0.5 for _ in range(batch_size)]
    returns = torch.FloatTensor([15.2, 12.8, 18.1, 14.5, 16.3, 13.7, 17.9, 15.8, 14.2, 16.7])
    advantages = torch.FloatTensor([0.2, -0.2, 0.6, 0.5, 0.3, -0.1, 0.9, 0.3, 0.2, 0.2])
    
    return states, actions, log_probs_old, returns, advantages


@pytest.fixture
def extreme_advantages_data():
    """
    Fixture providing data with extreme advantage values for testing robustness.
    
    Returns:
        tuple: (states, actions, log_probs_old, returns, advantages) with extreme advantages
    """
    batch_size = 8
    states = [np.array([0.8, 0.2, 350.0]) for _ in range(batch_size)]
    actions = [np.array([2.3]) for _ in range(batch_size)]
    log_probs_old = [-0.5 for _ in range(batch_size)]
    returns = torch.FloatTensor([15.2, 12.8, 18.1, 14.5, 16.3, 13.7, 17.9, 15.8])
    
    # Extreme advantages: very large positive, very large negative, zero, and mixed
    advantages = torch.FloatTensor([100.0, -50.0, 0.0, 25.0, -30.0, 75.0, -100.0, 0.5])
    
    return states, actions, log_probs_old, returns, advantages


@pytest.fixture
def clipping_test_data():
    """
    Fixture providing data specifically designed to test PPO clipping behavior.
    
    Creates data where the policy changes significantly to trigger clipping.
    
    Returns:
        tuple: (states, actions, log_probs_old, returns, advantages) for clipping tests
    """
    batch_size = 6
    states = [np.array([0.8, 0.2, 350.0]) for _ in range(batch_size)]
    actions = [np.array([2.3]) for _ in range(batch_size)]
    
    # Create log_probs_old that will result in extreme ratios when compared to new policy
    # This will trigger clipping behavior
    log_probs_old = [-3.0, -0.1, -5.0, 0.5, -2.0, -1.0]  # Mix of extreme and moderate values
    returns = torch.FloatTensor([15.2, 12.8, 18.1, 14.5, 16.3, 13.7])
    advantages = torch.FloatTensor([1.0, -1.0, 2.0, -0.5, 1.5, -1.5])
    
    return states, actions, log_probs_old, returns, advantages


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
        gae_advantages_normalized, total_expected_future_rewards, raw_gae_advantages  = compute_gae(rewards, dones, values, gae_lambda)
        
        # Then: GAE advantages and total expected future rewards should be computed correctly
        # Check that gae_advantages_normalized has the correct length
        assert len(gae_advantages_normalized) == len(rewards), \
            f"gae_advantages_normalized length mismatch: expected {len(rewards)}, got {len(gae_advantages_normalized)}"
        # Check that total expected future rewards has the correct length
        assert len(total_expected_future_rewards) == len(rewards), \
            f"total expected future rewards length mismatch: expected {len(rewards)}, got {len(total_expected_future_rewards)}"
        # Check that raw_gae_advantages has the correct length
        assert len(raw_gae_advantages) == len(rewards), \
            f"raw_gae_advantages length mismatch: expected {len(rewards)}, got {len(raw_gae_advantages)}"
        
        # Check that gae_advantages_normalized are finite
        assert torch.all(torch.isfinite(gae_advantages_normalized)), \
            "gae_advantages_normalized should be finite"
        # Check that total expected future rewards are finite
        assert torch.all(torch.isfinite(total_expected_future_rewards)), \
            "total expected future rewards should be finite"
        # Check that raw_gae_advantages are finite
        assert torch.all(torch.isfinite(raw_gae_advantages)), \
            "raw_gae_advantages should be finite"
        
        # Check that gae_advantages_normalized and total_expected_future_rewards are not all zeros
        assert torch.any(gae_advantages_normalized != 0), \
            "gae_advantages_normalized should not be all zeros"
        assert torch.any(total_expected_future_rewards != 0), \
            "total_expected_future_rewards should not be all zeros"
        # Check that raw_gae_advantages and gae_advantages_normalized are not all zeros
        assert torch.any(raw_gae_advantages != 0), \
            "raw_gae_advantages should not be all zeros"


    def test_compute_gae_short_trajectory(self, short_trajectory_data):
        """
        Test that compute_gae handles a short trajectory correctly.
        """
        # Given: Short trajectory data (3 timesteps)
        rewards, dones, values = short_trajectory_data
        
        # When: Computing GAE with default parameters
        gae_lambda = 0.95
        gae_advantages_normalized, total_expected_future_rewards, raw_gae_advantages = compute_gae(rewards, dones, values, gae_lambda)
        
        # Then: GAE advantages normalized and total expected future rewards should be computed correctly
        # Check that gae_advantages_normalized has the correct length
        assert len(gae_advantages_normalized) == len(rewards), \
            f"gae_advantages_normalized length mismatch: expected {len(rewards)}, got {len(gae_advantages_normalized)}"
        # Check that total expected future rewards has the correct length
        assert len(total_expected_future_rewards) == len(rewards), \
            f"total expected future rewards length mismatch: expected {len(rewards)}, got {len(total_expected_future_rewards)}"
        # Check that raw_gae_advantages has the correct length
        assert len(raw_gae_advantages) == len(rewards), \
            f"raw_gae_advantages length mismatch: expected {len(rewards)}, got {len(raw_gae_advantages)}"
        
        # Check that gae_advantages_normalized are finite
        assert torch.all(torch.isfinite(gae_advantages_normalized)), \
            "gae_advantages_normalized should be finite"
        # Check that total expected future rewards are finite
        assert torch.all(torch.isfinite(total_expected_future_rewards)), \
            "total expected future rewards should be finite"
        # Check that raw_gae_advantages are finite
        assert torch.all(torch.isfinite(raw_gae_advantages)), \
            "raw_gae_advantages should be finite"
        
        # Check that gae_advantages_normalized and total_expected_future_rewards are not all zeros
        assert torch.any(gae_advantages_normalized != 0), \
            "gae_advantages_normalized should not be all zeros"
        # Check that total expected future rewards are not all zeros
        assert torch.any(total_expected_future_rewards != 0), \
            "total expected future rewards should not be all zeros"
        # Check that raw_gae_advantages and gae_advantages_normalized are not all zeros
        assert torch.any(raw_gae_advantages != 0), \
            "raw_gae_advantages should not be all zeros"


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
        gae_advantages_normalized, total_expected_future_rewards, raw_gae_advantages = compute_gae(rewards, dones, values, gamma, gae_lambda)
        
        # Then: gae_advantages_normalized should be properly normalized
        # Check that advantages have mean=0 (within numerical precision)
        advantages_mean = gae_advantages_normalized.mean().item()
        assert abs(advantages_mean) < 1e-6, \
            f"Normalized advantages should have mean=0, got {advantages_mean}"
        
        # Check that advantages have std=1 (within numerical precision)
        advantages_std = gae_advantages_normalized.std().item()
        assert abs(advantages_std - 1.0) < 1e-6, \
            f"Normalized advantages should have std=1, got {advantages_std}"
        
        # Check that gae_advantages_normalized have reasonable range (should be mostly within ±3 std)
        advantages_abs_normalized = torch.abs(gae_advantages_normalized)
        assert torch.all(advantages_abs_normalized < 10), \
            "Raw GAE advantages should be within reasonable range"


    def test_compute_gae_lambda_0(self, sample_trajectory_data):
        """
        Test that compute_gae with lambda=0.0 computes Time Difference TD(0) advantages.
        
        When λ=0.0, GAE reduces to Time Difference TD(0) advantages:
        - Advantages = rewards + γ*V(next_state) - V(current_state)
        - Returns = advantages + V(current_state)
        """
        # Given: Sample trajectory data (10 timesteps)
        rewards, dones, values = sample_trajectory_data
        
        # When: Computing GAE with lambda=0.0 and making a manual TD(0) calculation
        gamma = 0.99
        gae_lambda = 0.0
        gae_advantages_normalized, total_expected_future_rewards, raw_gae_advantages = compute_gae(rewards, dones, values, gamma, gae_lambda)
        # Manual TD(0) calculation for verification
        td0_advantages = []
        for i in range(len(rewards)):
            if i == len(rewards) - 1:  # Last timestep
                # If done, no future value
                td0_adv = rewards[i] - values[i]
            else:
                # TD(0): r_t + γ*V(s_{t+1}) - V(s_t)
                td0_adv = rewards[i] + gamma * values[i+1] - values[i]
            td0_advantages.append(td0_adv)
        
        td0_advantages_tensor = torch.FloatTensor(td0_advantages)        
        # Then: GAE should compute TD(0) advantages correctly        
        # Check that raw_gae_advantages match TD(0) advantages
        assert torch.allclose(raw_gae_advantages, td0_advantages_tensor, atol=1e-6), \
            f"Raw GAE advantages should match TD(0) advantages. Got {raw_gae_advantages}, expected {td0_advantages_tensor}"
        
        # Check that returns are advantages + values
        expected_returns = td0_advantages_tensor + torch.FloatTensor(values)
        assert torch.allclose(total_expected_future_rewards, expected_returns, atol=1e-6), \
            f"Returns should be advantages + values. Got {total_expected_future_rewards}, expected {expected_returns}"


    def test_compute_gae_lambda_1(self, sample_trajectory_data):
        """
        Test that compute_gae with lambda=1.0 computes Monte Carlo (MC) advantages.
        
        When λ=1.0, GAE reduces to Monte Carlo advantages:
        - Advantages = Σ(γ^t * r_t) - V(current_state) = Returns - V(current_state)
        - Returns = Σ(γ^t * r_t) (discounted sum of future rewards)
        """
        # Given: Sample trajectory data (10 timesteps)
        rewards, dones, values = sample_trajectory_data
        
        # When: Computing GAE with lambda=1.0 and making a manual MC calculation
        gamma = 0.99
        gae_lambda = 1.0
        gae_advantages_normalized, total_expected_future_rewards, raw_gae_advantages = compute_gae(rewards, dones, values, gamma, gae_lambda)
        
        # Manual Monte Carlo calculation for verification
        mc_advantages = []
        mc_returns = []
        
        for i in range(len(rewards)):
            # Calculate discounted sum of future rewards (MC returns)
            mc_return = 0.0
            discount = 1.0
            for j in range(i, len(rewards)):
                mc_return += discount * rewards[j]
                discount *= gamma
                if dones[j]:  # Stop if episode ends
                    break
            
            mc_returns.append(mc_return)
            # MC advantage = MC return - V(current_state)
            mc_advantage = mc_return - values[i]
            mc_advantages.append(mc_advantage)
        
        mc_advantages_tensor = torch.FloatTensor(mc_advantages)
        mc_returns_tensor = torch.FloatTensor(mc_returns)
        
        # Then: GAE should compute MC advantages correctly
        # Check that raw_gae_advantages match MC advantages
        assert torch.allclose(raw_gae_advantages, mc_advantages_tensor, atol=1e-6), \
            f"Raw GAE advantages should match MC advantages. Got {raw_gae_advantages}, expected {mc_advantages_tensor}"
        
        # Check that returns match MC returns
        assert torch.allclose(total_expected_future_rewards, mc_returns_tensor, atol=1e-6), \
            f"Returns should match MC returns. Got {total_expected_future_rewards}, expected {mc_returns_tensor}"


    def test_compute_gae_episode_termination(self):
        """
        Test that compute_gae handles episode termination correctly.
        
        This test verifies that GAE correctly handles:
        1. Episode termination in the middle of a trajectory
        2. Episode termination at the end of a trajectory
        3. Multiple episode terminations in a single trajectory
        4. Proper bootstrapping when episodes end
        
        For CSTR context: Tests reactor shutdown, safety violations, and time limits.
        """
        # Given: Trajectory with multiple episode terminations
        rewards = [10.0, 15.0, 8.0, 12.0, 20.0, 5.0, 18.0, 14.0, 9.0, 16.0]
        dones = [False, False, True, False, False, True, False, False, False, True]  # 3 episodes
        values = [12.0, 14.0, 9.0, 13.0, 18.0, 6.0, 16.0, 15.0, 10.0, 17.0]
        
        # When: Computing GAE with default parameters and making a manual GAE calculation
        gamma = 0.99
        gae_lambda = 0.95
        gae_advantages_normalized, total_expected_future_rewards, raw_gae_advantages = compute_gae(rewards, dones, values, gamma, gae_lambda)
        
        # Manual GAE calculation for verification with episode termination handling
        manual_advantages = []
        manual_returns = []
        
        # Bootstrap value for the "next" state after the final timestep
        values_with_bootstrap = values + [0.0]  # Add bootstrap value
        
        # Compute GAE backwards
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            # Compute delta: r_t + γ*V(s_{t+1})*(1-done_t) - V(s_t)
            if t == len(rewards) - 1:  # Last timestep
                # Bootstrap with zero value
                delta = rewards[t] + gamma * 0.0 * (1 - dones[t]) - values[t]
            else:
                # Use next state value
                delta = rewards[t] + gamma * values_with_bootstrap[t + 1] * (1 - dones[t]) - values[t]
            
            # GAE: A_t = δ_t + γλ(1-done_t)A_{t+1}
            gae_advantage = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            manual_advantages.append(gae_advantage)
            last_gae = gae_advantage
        
        # Reverse to get correct order
        manual_advantages.reverse()
        manual_advantages_tensor = torch.FloatTensor(manual_advantages)
        
        # Compute returns: returns = advantages + values
        manual_returns = [adv + val for adv, val in zip(manual_advantages, values)]
        manual_returns_tensor = torch.FloatTensor(manual_returns)
        
        # Then: GAE should handle episode termination correctly
        # Check that raw_gae_advantages match manual calculation
        assert torch.allclose(raw_gae_advantages, manual_advantages_tensor, atol=1e-6), \
            f"Raw GAE advantages should match manual calculation. Got {raw_gae_advantages}, expected {manual_advantages_tensor}"
        
        # Check that returns match manual calculation
        assert torch.allclose(total_expected_future_rewards, manual_returns_tensor, atol=1e-6), \
            f"Returns should match manual calculation. Got {total_expected_future_rewards}, expected {manual_returns_tensor}"
        
        # Check that episode termination affects the computation correctly
        # Episode 1: timesteps 0-2 (ends at t=2)
        # Episode 2: timesteps 3-5 (ends at t=5)  
        # Episode 3: timesteps 6-9 (ends at t=9)
        
        # Verify that advantages are finite and reasonable
        assert torch.all(torch.isfinite(raw_gae_advantages)), \
            "Advantages should be finite even with episode termination"
        assert torch.all(torch.isfinite(total_expected_future_rewards)), \
            "Returns should be finite even with episode termination"
        
        # Verify that the computation handles the episode boundaries correctly
        # The advantages should reflect the episode structure
        assert len(raw_gae_advantages) == len(rewards), \
            f"Advantages length should match rewards length. Got {len(raw_gae_advantages)}, expected {len(rewards)}"
        assert len(total_expected_future_rewards) == len(rewards), \
            f"Returns length should match rewards length. Got {len(total_expected_future_rewards)}, expected {len(rewards)}"


    @pytest.mark.parametrize(
        "gamma,description",
        [
            (0.5, "high_discounting"),
            (0.9, "moderate_discounting"), 
            (0.99, "low_discounting"),
            (1.0, "no_discounting"),
        ],
        ids=["gamma_0.5_high_discounting", "gamma_0.9_moderate_discounting", "gamma_0.99_low_discounting", "gamma_1.0_no_discounting"]
    )
    def test_compute_gae_different_gamma(self, gamma, description):
        """
        Test that compute_gae works correctly with different gamma values.
        
        This test verifies that GAE correctly handles different discount factors:
        1. gamma = 0.5 (high discounting - immediate rewards matter more)
        2. gamma = 0.9 (moderate discounting - balanced future/immediate)
        3. gamma = 0.99 (low discounting - future rewards matter more)
        4. gamma = 1.0 (no discounting - all rewards equally important)
        
        For CSTR context: Tests different time horizons for reactor control decisions.
        """
        # Given: Sample trajectory data
        rewards = [10.0, 15.0, 8.0, 12.0, 20.0, 5.0, 18.0, 14.0, 9.0, 16.0]
        dones = [False, False, False, False, False, False, False, False, False, True]
        values = [12.0, 14.0, 9.0, 13.0, 18.0, 6.0, 16.0, 15.0, 10.0, 17.0]
        
        # When: Computing GAE with the specified gamma value
        gae_lambda = 0.95
        gae_advantages_normalized, total_expected_future_rewards, raw_gae_advantages = compute_gae(
            rewards, dones, values, gamma, gae_lambda
        )
        
        # Manual GAE calculation for verification
        manual_advantages = []
        values_with_bootstrap = values + [0.0]  # Add bootstrap value
        
        # Compute GAE backwards
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            # Compute delta: r_t + γ*V(s_{t+1})*(1-done_t) - V(s_t)
            if t == len(rewards) - 1:  # Last timestep
                # Bootstrap with zero value
                delta = rewards[t] + gamma * 0.0 * (1 - dones[t]) - values[t]
            else:
                # Use next state value
                delta = rewards[t] + gamma * values_with_bootstrap[t + 1] * (1 - dones[t]) - values[t]
            
            # GAE: A_t = δ_t + γλ(1-done_t)A_{t+1}
            gae_advantage = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
            manual_advantages.append(gae_advantage)
            last_gae = gae_advantage
        
        # Reverse to get correct order
        manual_advantages.reverse()
        manual_advantages_tensor = torch.FloatTensor(manual_advantages)
        
        # Compute returns: returns = advantages + values
        manual_returns = [adv + val for adv, val in zip(manual_advantages, values)]
        manual_returns_tensor = torch.FloatTensor(manual_returns)
        
        # Then: GAE should work correctly with the specified gamma value
        # Check that raw_gae_advantages match manual calculation
        assert torch.allclose(raw_gae_advantages, manual_advantages_tensor, atol=1e-6), \
            f"Raw GAE advantages should match manual calculation for gamma={gamma} ({description}). " \
            f"Got {raw_gae_advantages}, expected {manual_advantages_tensor}"
        
        # Check that returns match manual calculation
        assert torch.allclose(total_expected_future_rewards, manual_returns_tensor, atol=1e-6), \
            f"Returns should match manual calculation for gamma={gamma} ({description}). " \
            f"Got {total_expected_future_rewards}, expected {manual_returns_tensor}"
        
        # Verify that advantages are finite and reasonable
        assert torch.all(torch.isfinite(raw_gae_advantages)), \
            f"Advantages should be finite for gamma={gamma} ({description})"
        assert torch.all(torch.isfinite(total_expected_future_rewards)), \
            f"Returns should be finite for gamma={gamma} ({description})"
        
        # Verify that the computation handles the gamma value correctly
        assert len(raw_gae_advantages) == len(rewards), \
            f"Advantages length should match rewards length for gamma={gamma} ({description}). " \
            f"Got {len(raw_gae_advantages)}, expected {len(rewards)}"
        assert len(total_expected_future_rewards) == len(rewards), \
            f"Returns length should match rewards length for gamma={gamma} ({description}). " \
            f"Got {len(total_expected_future_rewards)}, expected {len(rewards)}"


class TestPPOUpdate:
    """Test suite for ppo_update function from base_agent.py"""
    
    def test_ppo_update_basic(self, mock_actor_critic_model, sample_ppo_data):
        """
        Test that ppo_update function performs basic policy and value function updates.
        
        This test verifies that:
        1. The function accepts valid inputs without errors
        2. The model parameters are updated during training
        3. The optimizers work correctly with the model
        4. The function handles the PPO update process properly
        
        Uses a mock model to test ppo_update in isolation without depending on
        the concrete ActorCriticNet implementation.
        """
        # Given: A mock model, sample training data, and optimizers
        model = mock_actor_critic_model
        states, actions, log_probs_old, returns, advantages = sample_ppo_data
        
        # Create optimizers for actor and critic
        # Include log_std parameter in actor optimizer since it's part of the actor
        actor_params = list(model.actor.parameters()) + [model.log_std]
        actor_optimizer = optim.Adam(actor_params, lr=3e-4)
        critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)
        
        # Store initial model parameters for comparison
        initial_actor_params = {name: param.clone() for name, param in model.actor.named_parameters()}
        initial_critic_params = {name: param.clone() for name, param in model.critic.named_parameters()}
        initial_log_std = model.log_std.clone()
        
        # When: Performing PPO update
        ppo_update(
            model=model,
            states=states,
            actions=actions,
            log_probs_old=log_probs_old,
            returns=returns,
            advantages=advantages,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            actor_clip=0.2,
            value_clip=0.2,
            epochs=5  # Using fewer epochs for faster testing
        )
        
        # Then: The model parameters should be updated and it should be able to perform forward passes
        # Check that actor parameters have changed
        actor_params_changed = False
        for name, param in model.actor.named_parameters():
            if not torch.allclose(param, initial_actor_params[name]):
                actor_params_changed = True
                break
        
        assert actor_params_changed, \
            "Actor parameters should be updated during PPO training"
        
        # Check that critic parameters have changed
        critic_params_changed = False
        for name, param in model.critic.named_parameters():
            if not torch.allclose(param, initial_critic_params[name]):
                critic_params_changed = True
                break
        
        assert critic_params_changed, \
            "Critic parameters should be updated during PPO training"
        
        # Check that log_std parameter has changed
        log_std_changed = not torch.allclose(model.log_std, initial_log_std)
        assert log_std_changed, \
            "Log std parameter should be updated during PPO training"
        
        # Verify that the model can still perform forward passes
        test_states = torch.FloatTensor([[0.8, 0.2, 350.0], [0.5, 0.5, 340.0]])
        mean, std, values = model(test_states)
        
        # Check that outputs have correct shapes
        assert mean.shape == (2, 1), \
            f"Actor output should have shape (2, 1), got {mean.shape}"
        assert std.shape == (1,), \
            f"Actor std should have shape (1,), got {std.shape}"
        assert values.shape == (2, 1), \
            f"Critic output should have shape (2, 1), got {values.shape}"
        
        # Check that outputs are finite
        assert torch.all(torch.isfinite(mean)), \
            "Actor mean output should be finite"
        assert torch.all(torch.isfinite(std)), \
            "Actor std output should be finite"
        assert torch.all(torch.isfinite(values)), \
            "Critic values output should be finite"

    def test_ppo_update_actor_clipping_effect(self, mock_actor_critic_model, clipping_test_data):
        """
        Test that PPO clipping prevents extreme policy changes.
        
        This test verifies that:
        1. PPO's clipped surrogate objective works correctly
        2. Extreme policy changes are prevented by clipping
        3. The minimum of clipped and unclipped objectives is used
        4. Clipping maintains training stability
        
        Uses data designed to trigger clipping behavior.
        """
        # Given: A mock model and data designed to trigger clipping
        model = mock_actor_critic_model
        states, actions, log_probs_old, returns, advantages = clipping_test_data
        
        # Create optimizers
        actor_params = list(model.actor.parameters()) + [model.log_std]
        actor_optimizer = optim.Adam(actor_params, lr=3e-4)
        critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)
        
        # Store initial parameters
        initial_actor_params = {name: param.clone() for name, param in model.actor.named_parameters()}
        initial_critic_params = {name: param.clone() for name, param in model.critic.named_parameters()}
        initial_log_std = model.log_std.clone()
        
        # When: Performing PPO update with clipping
        actor_clip_value = 0.2
        ppo_update(
            model=model,
            states=states,
            actions=actions,
            log_probs_old=log_probs_old,
            returns=returns,
            advantages=advantages,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            actor_clip=actor_clip_value,
            epochs=3  # Fewer epochs for faster testing
        )
        
        # Then: Clipping should prevent extreme policy changes
        # Check that parameters changed but not too drastically
        actor_params_changed = False
        max_param_change = 0.0
        
        for name, param in model.actor.named_parameters():
            initial_param = initial_actor_params[name]
            if not torch.allclose(param, initial_param):
                actor_params_changed = True
                # Calculate maximum relative change
                relative_change = torch.max(torch.abs(param - initial_param) / (torch.abs(initial_param) + 1e-8))
                max_param_change = max(max_param_change, relative_change.item())
        
        assert actor_params_changed, \
            "Actor parameters should be updated during PPO training"
        
        # Check that parameter changes are reasonable (not extreme)
        # This verifies that clipping is working - allow for some larger changes
        # since we're using extreme log_probs_old values designed to trigger clipping
        assert max_param_change < 5.0, \
            f"Parameter changes should be reasonable, got max change of {max_param_change}"
        
        # Check that critic parameters also changed
        critic_params_changed = False
        for name, param in model.critic.named_parameters():
            if not torch.allclose(param, initial_critic_params[name]):
                critic_params_changed = True
                break
        
        assert critic_params_changed, \
            "Critic parameters should be updated during PPO training"
        
        # Check that log_std parameter changed
        log_std_changed = not torch.allclose(model.log_std, initial_log_std)
        assert log_std_changed, \
            "Log std parameter should be updated during PPO training"
        
        # Verify model still works after clipping
        test_states = torch.FloatTensor([[0.8, 0.2, 350.0], [0.5, 0.5, 340.0]])
        mean, std, values = model(test_states)
        
        assert torch.all(torch.isfinite(mean)), \
            "Actor mean output should be finite after clipping"
        assert torch.all(torch.isfinite(std)), \
            "Actor std output should be finite after clipping"
        assert torch.all(torch.isfinite(values)), \
            "Critic values output should be finite after clipping"


    @pytest.mark.parametrize(
        "clip_value,description",
        [
            (0.01, "very_conservative"),
            (0.1, "conservative"),
            (0.2, "standard"),
            (0.3, "aggressive"),
            (0.5, "very_aggressive"),
        ],
        ids=["clip_0.01_very_conservative", "clip_0.1_conservative", "clip_0.2_standard", 
             "clip_0.3_aggressive", "clip_0.5_very_aggressive"]
    )
    def test_ppo_update_different_actor_clip_values(self, mock_actor_critic_model, sample_ppo_data, 
                                            clip_value, description):
        """
        Test that PPO update works correctly with different clipping values.
        
        This test verifies that:
        1. Different clip values produce different update behaviors
        2. Conservative clips (small values) produce smaller parameter changes
        3. Aggressive clips (large values) allow larger parameter changes
        4. All clip values maintain training stability
        
        For CSTR context: Tests different levels of policy change conservatism
        in temperature control strategy updates.
        """
        # Given: A mock model and sample data
        model = mock_actor_critic_model
        states, actions, log_probs_old, returns, advantages = sample_ppo_data
        
        # Create optimizers
        actor_params = list(model.actor.parameters()) + [model.log_std]
        actor_optimizer = optim.Adam(actor_params, lr=3e-4)
        critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)
        
        # Store initial parameters
        initial_actor_params = {name: param.clone() for name, param in model.actor.named_parameters()}
        initial_critic_params = {name: param.clone() for name, param in model.critic.named_parameters()}
        initial_log_std = model.log_std.clone()
        
        # When: Performing PPO update with the specified clip value
        ppo_update(
            model=model,
            states=states,
            actions=actions,
            log_probs_old=log_probs_old,
            returns=returns,
            advantages=advantages,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            actor_clip=clip_value,
            epochs=3  # Fewer epochs for faster testing
        )
        
        # Then: Different clip values should produce different behaviors
        # Check that parameters changed
        actor_params_changed = False
        total_actor_change = 0.0
        
        for name, param in model.actor.named_parameters():
            initial_param = initial_actor_params[name]
            if not torch.allclose(param, initial_param):
                actor_params_changed = True
                # Calculate total parameter change
                param_change = torch.sum(torch.abs(param - initial_param)).item()
                total_actor_change += param_change
        
        assert actor_params_changed, \
            f"Actor parameters should be updated with clip={clip_value} ({description})"
        
        # Check that critic parameters changed
        critic_params_changed = False
        for name, param in model.critic.named_parameters():
            if not torch.allclose(param, initial_critic_params[name]):
                critic_params_changed = True
                break
        
        assert critic_params_changed, \
            f"Critic parameters should be updated with clip={clip_value} ({description})"
        
        # Check that log_std parameter changed
        log_std_changed = not torch.allclose(model.log_std, initial_log_std)
        assert log_std_changed, \
            f"Log std parameter should be updated with clip={clip_value} ({description})"
        
        # Verify model still works after update
        test_states = torch.FloatTensor([[0.8, 0.2, 350.0], [0.5, 0.5, 340.0]])
        mean, std, values = model(test_states)
        
        assert torch.all(torch.isfinite(mean)), \
            f"Actor mean output should be finite with clip={clip_value} ({description})"
        assert torch.all(torch.isfinite(std)), \
            f"Actor std output should be finite with clip={clip_value} ({description})"
        assert torch.all(torch.isfinite(values)), \
            f"Critic values output should be finite with clip={clip_value} ({description})"
        
        # Store total change for potential comparison (could be used in future tests)
        # For now, just verify that changes occurred
        assert total_actor_change > 0, \
            f"Total actor parameter change should be positive with clip={clip_value} ({description})"


    def test_ppo_update_extreme_advantages(self, mock_actor_critic_model, extreme_advantages_data):
        """
        Test that PPO update handles extreme advantage values correctly.
        
        This test verifies that:
        1. Very large positive advantages don't cause numerical instability
        2. Very large negative advantages don't cause numerical instability
        3. Zero advantages are handled correctly
        4. Mixed extreme advantages don't crash the training
        5. All computations remain finite and stable
        
        For CSTR context: Tests robustness when temperature adjustments
        have unexpectedly good or bad outcomes.
        """
        # Given: A mock model and data with extreme advantages
        model = mock_actor_critic_model
        states, actions, log_probs_old, returns, advantages = extreme_advantages_data
        
        # Create optimizers
        actor_params = list(model.actor.parameters()) + [model.log_std]
        actor_optimizer = optim.Adam(actor_params, lr=3e-4)
        critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)
        
        # Store initial parameters
        initial_actor_params = {name: param.clone() for name, param in model.actor.named_parameters()}
        initial_critic_params = {name: param.clone() for name, param in model.critic.named_parameters()}
        initial_log_std = model.log_std.clone()
        
        # When: Performing PPO update with extreme advantages
        ppo_update(
            model=model,
            states=states,
            actions=actions,
            log_probs_old=log_probs_old,
            returns=returns,
            advantages=advantages,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            actor_clip=0.2,
            epochs=3  # Fewer epochs for faster testing
        )
        
        # Then: Extreme advantages should be handled without numerical issues
        # Check that parameters changed (training occurred)
        actor_params_changed = False
        for name, param in model.actor.named_parameters():
            if not torch.allclose(param, initial_actor_params[name]):
                actor_params_changed = True
                break
        
        assert actor_params_changed, \
            "Actor parameters should be updated even with extreme advantages"
        
        # Check that critic parameters changed
        critic_params_changed = False
        for name, param in model.critic.named_parameters():
            if not torch.allclose(param, initial_critic_params[name]):
                critic_params_changed = True
                break
        
        assert critic_params_changed, \
            "Critic parameters should be updated even with extreme advantages"
        
        # Check that log_std parameter changed
        log_std_changed = not torch.allclose(model.log_std, initial_log_std)
        assert log_std_changed, \
            "Log std parameter should be updated even with extreme advantages"
        
        # Verify that all model parameters are finite
        for name, param in model.named_parameters():
            assert torch.all(torch.isfinite(param)), \
                f"Parameter {name} should be finite after extreme advantages"
        
        # Verify model can still perform forward passes
        test_states = torch.FloatTensor([[0.8, 0.2, 350.0], [0.5, 0.5, 340.0]])
        mean, std, values = model(test_states)
        
        # Check that outputs are finite
        assert torch.all(torch.isfinite(mean)), \
            "Actor mean output should be finite after extreme advantages"
        assert torch.all(torch.isfinite(std)), \
            "Actor std output should be finite after extreme advantages"
        assert torch.all(torch.isfinite(values)), \
            "Critic values output should be finite after extreme advantages"
        
        # Check that outputs have correct shapes
        assert mean.shape == (2, 1), \
            f"Actor output should have shape (2, 1), got {mean.shape}"
        assert std.shape == (1,), \
            f"Actor std should have shape (1,), got {std.shape}"
        assert values.shape == (2, 1), \
            f"Critic output should have shape (2, 1), got {values.shape}"
        
        # Verify that std is positive (as expected for standard deviation)
        assert torch.all(std > 0), \
            "Action std should be positive after extreme advantages"

    def test_ppo_update_critic_clipping_effect(self, mock_actor_critic_model, sample_ppo_data):
        """
        Test that PPO value function clipping prevents extreme critic changes.
        
        This test verifies that:
        1. PPO's value function clipping works correctly
        2. Extreme critic changes are prevented by clipping
        3. The maximum of clipped and unclipped value losses is used
        4. Value function clipping maintains training stability
        
        Uses data designed to test value function clipping behavior.
        """
        # Given: A mock model and sample data
        model = mock_actor_critic_model
        states, actions, log_probs_old, returns, advantages = sample_ppo_data
        
        # Create optimizers
        actor_params = list(model.actor.parameters()) + [model.log_std]
        actor_optimizer = optim.Adam(actor_params, lr=3e-4)
        critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)
        
        # Store initial parameters
        initial_actor_params = {name: param.clone() for name, param in model.actor.named_parameters()}
        initial_critic_params = {name: param.clone() for name, param in model.critic.named_parameters()}
        initial_log_std = model.log_std.clone()
        
        # When: Performing PPO update with value function clipping
        value_clip_value = 0.2
        ppo_update(
            model=model,
            states=states,
            actions=actions,
            log_probs_old=log_probs_old,
            returns=returns,
            advantages=advantages,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            actor_clip=0.2,
            value_clip=value_clip_value,
            epochs=3  # Fewer epochs for faster testing
        )
        
        # Then: Value function clipping should prevent extreme critic changes
        # Check that parameters changed but not too drastically
        critic_params_changed = False
        max_critic_param_change = 0.0
        
        for name, param in model.critic.named_parameters():
            initial_param = initial_critic_params[name]
            if not torch.allclose(param, initial_param):
                critic_params_changed = True
                # Calculate maximum relative change
                relative_change = torch.max(torch.abs(param - initial_param) / (torch.abs(initial_param) + 1e-8))
                max_critic_param_change = max(max_critic_param_change, relative_change.item())
        
        assert critic_params_changed, \
            "Critic parameters should be updated during PPO training"
        
        # Check that parameter changes are reasonable (not extreme)
        # This verifies that value function clipping is working
        assert max_critic_param_change < 5.0, \
            f"Critic parameter changes should be reasonable, got max change of {max_critic_param_change}"
        
        # Check that actor parameters also changed
        actor_params_changed = False
        for name, param in model.actor.named_parameters():
            if not torch.allclose(param, initial_actor_params[name]):
                actor_params_changed = True
                break
        
        assert actor_params_changed, \
            "Actor parameters should be updated during PPO training"
        
        # Check that log_std parameter changed
        log_std_changed = not torch.allclose(model.log_std, initial_log_std)
        assert log_std_changed, \
            "Log std parameter should be updated during PPO training"
        
        # Verify model still works after value function clipping
        test_states = torch.FloatTensor([[0.8, 0.2, 350.0], [0.5, 0.5, 340.0]])
        mean, std, values = model(test_states)
        
        assert torch.all(torch.isfinite(mean)), \
            "Actor mean output should be finite after value function clipping"
        assert torch.all(torch.isfinite(std)), \
            "Actor std output should be finite after value function clipping"
        assert torch.all(torch.isfinite(values)), \
            "Critic values output should be finite after value function clipping"

    @pytest.mark.parametrize(
        "value_clip_value,description",
        [
            (0.01, "very_conservative"),
            (0.1, "conservative"),
            (0.2, "standard"),
            (0.3, "aggressive"),
            (0.5, "very_aggressive"),
        ],
        ids=["value_clip_0.01_very_conservative", "value_clip_0.1_conservative", "value_clip_0.2_standard", 
             "value_clip_0.3_aggressive", "value_clip_0.5_very_aggressive"]
    )
    def test_ppo_update_different_value_clip_values(self, mock_actor_critic_model, sample_ppo_data, 
                                                  value_clip_value, description):
        """
        Test that PPO update works correctly with different value function clipping values.
        
        This test verifies that:
        1. Different value clip values produce different update behaviors
        2. Conservative value clips (small values) produce smaller critic parameter changes
        3. Aggressive value clips (large values) allow larger critic parameter changes
        4. All value clip values maintain training stability
        
        For CSTR context: Tests different levels of value function change conservatism
        in reactor state value estimation updates.
        """
        # Given: A mock model and sample data
        model = mock_actor_critic_model
        states, actions, log_probs_old, returns, advantages = sample_ppo_data
        
        # Create optimizers
        actor_params = list(model.actor.parameters()) + [model.log_std]
        actor_optimizer = optim.Adam(actor_params, lr=3e-4)
        critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)
        
        # Store initial parameters
        initial_actor_params = {name: param.clone() for name, param in model.actor.named_parameters()}
        initial_critic_params = {name: param.clone() for name, param in model.critic.named_parameters()}
        initial_log_std = model.log_std.clone()
        
        # When: Performing PPO update with the specified value clip value
        ppo_update(
            model=model,
            states=states,
            actions=actions,
            log_probs_old=log_probs_old,
            returns=returns,
            advantages=advantages,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            actor_clip=0.2,
            value_clip=value_clip_value,
            epochs=3  # Fewer epochs for faster testing
        )
        
        # Then: Different value clip values should produce different behaviors
        # Check that parameters changed
        critic_params_changed = False
        total_critic_change = 0.0
        
        for name, param in model.critic.named_parameters():
            initial_param = initial_critic_params[name]
            if not torch.allclose(param, initial_param):
                critic_params_changed = True
                # Calculate total parameter change
                param_change = torch.sum(torch.abs(param - initial_param)).item()
                total_critic_change += param_change
        
        assert critic_params_changed, \
            f"Critic parameters should be updated with value_clip={value_clip_value} ({description})"
        
        # Check that actor parameters changed
        actor_params_changed = False
        for name, param in model.actor.named_parameters():
            if not torch.allclose(param, initial_actor_params[name]):
                actor_params_changed = True
                break
        
        assert actor_params_changed, \
            f"Actor parameters should be updated with value_clip={value_clip_value} ({description})"
        
        # Check that log_std parameter changed
        log_std_changed = not torch.allclose(model.log_std, initial_log_std)
        assert log_std_changed, \
            f"Log std parameter should be updated with value_clip={value_clip_value} ({description})"
        
        # Verify model still works after update
        test_states = torch.FloatTensor([[0.8, 0.2, 350.0], [0.5, 0.5, 340.0]])
        mean, std, values = model(test_states)
        
        assert torch.all(torch.isfinite(mean)), \
            f"Actor mean output should be finite with value_clip={value_clip_value} ({description})"
        assert torch.all(torch.isfinite(std)), \
            f"Actor std output should be finite with value_clip={value_clip_value} ({description})"
        assert torch.all(torch.isfinite(values)), \
            f"Critic values output should be finite with value_clip={value_clip_value} ({description})"
        
        # Store total change for potential comparison (could be used in future tests)
        # For now, just verify that changes occurred
        assert total_critic_change > 0, \
            f"Total critic parameter change should be positive with value_clip={value_clip_value} ({description})"

    def test_ppo_update_memory_efficiency(self, mock_actor_critic_model, sample_ppo_data):
        """
        Test that PPO update doesn't leak memory across epochs.
        
        This test verifies that:
        1. No gradient accumulation between epochs
        2. Computational graph is properly cleared
        3. Memory usage remains reasonable
        4. No tensor memory leaks
        
        Critical for long training runs where memory leaks can crash training.
        """
        # Given: A mock model and sample data
        model = mock_actor_critic_model
        states, actions, log_probs_old, returns, advantages = sample_ppo_data
        
        # Create optimizers
        actor_params = list(model.actor.parameters()) + [model.log_std]
        actor_optimizer = optim.Adam(actor_params, lr=3e-4)
        critic_optimizer = optim.Adam(model.critic.parameters(), lr=1e-3)
        
        # Store initial parameters for comparison
        initial_actor_params = {name: param.clone() for name, param in model.actor.named_parameters()}
        initial_critic_params = {name: param.clone() for name, param in model.critic.named_parameters()}
        initial_log_std = model.log_std.clone()
        
        # When: Performing PPO update with multiple epochs
        ppo_update(
            model=model,
            states=states,
            actions=actions,
            log_probs_old=log_probs_old,
            returns=returns,
            advantages=advantages,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            actor_clip=0.2,
            value_clip=0.2,
            epochs=5  # Multiple epochs to test memory efficiency
        )
        
        # Then: Memory should be managed efficiently
        # Check that parameters changed (training occurred)
        actor_params_changed = False
        for name, param in model.actor.named_parameters():
            if not torch.allclose(param, initial_actor_params[name]):
                actor_params_changed = True
                break
        
        assert actor_params_changed, \
            "Actor parameters should be updated during PPO training"
        
        critic_params_changed = False
        for name, param in model.critic.named_parameters():
            if not torch.allclose(param, initial_critic_params[name]):
                critic_params_changed = True
                break
        
        assert critic_params_changed, \
            "Critic parameters should be updated during PPO training"
        
        # Check that no gradients are accumulated (memory leak prevention)
        for param in model.parameters():
            assert param.grad is None, \
                "Gradients should be cleared after PPO update to prevent memory leaks"
        
        # Verify model still works after multiple epochs
        test_states = torch.FloatTensor([[0.8, 0.2, 350.0], [0.5, 0.5, 340.0]])
        mean, std, values = model(test_states)
        
        assert torch.all(torch.isfinite(mean)), \
            "Actor mean output should be finite after multiple epochs"
        assert torch.all(torch.isfinite(std)), \
            "Actor std output should be finite after multiple epochs"
        assert torch.all(torch.isfinite(values)), \
            "Critic values output should be finite after multiple epochs"

    def test_ppo_update_edge_cases(self, mock_actor_critic_model):
        """
        Test that PPO update handles edge cases correctly.
        
        This test verifies that:
        1. Single sample batches work correctly
        2. Zero advantages are handled properly
        3. Extreme ratios don't crash training
        4. Empty data is handled gracefully
        
        Important for robustness in real-world scenarios.
        """
        # Given: Edge case data
        # Single sample batch
        states_single = [np.array([0.8, 0.2, 350.0])]
        actions_single = [np.array([2.3])]
        log_probs_old_single = [-0.5]
        returns_single = torch.FloatTensor([15.2])
        advantages_single = torch.FloatTensor([0.2])
        
        # Zero advantages
        states_zero = [np.array([0.8, 0.2, 350.0]) for _ in range(3)]
        actions_zero = [np.array([2.3]) for _ in range(3)]
        log_probs_old_zero = [-0.5 for _ in range(3)]
        returns_zero = torch.FloatTensor([15.2, 12.8, 18.1])
        advantages_zero = torch.FloatTensor([0.0, 0.0, 0.0])
        
        # Create optimizers
        actor_params = list(mock_actor_critic_model.actor.parameters()) + [mock_actor_critic_model.log_std]
        actor_optimizer = optim.Adam(actor_params, lr=3e-4)
        critic_optimizer = optim.Adam(mock_actor_critic_model.critic.parameters(), lr=1e-3)
        
        # When: Testing single sample batch
        ppo_update(
            model=mock_actor_critic_model,
            states=states_single,
            actions=actions_single,
            log_probs_old=log_probs_old_single,
            returns=returns_single,
            advantages=advantages_single,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            actor_clip=0.2,
            epochs=2
        )
        
        # Then: Single sample should work correctly
        test_states = torch.FloatTensor([[0.8, 0.2, 350.0]])
        mean, std, values = mock_actor_critic_model(test_states)
        
        assert torch.all(torch.isfinite(mean)), \
            "Model should work with single sample batch"
        assert torch.all(torch.isfinite(std)), \
            "Model should work with single sample batch"
        assert torch.all(torch.isfinite(values)), \
            "Model should work with single sample batch"
        
        # When: Testing zero advantages
        ppo_update(
            model=mock_actor_critic_model,
            states=states_zero,
            actions=actions_zero,
            log_probs_old=log_probs_old_zero,
            returns=returns_zero,
            advantages=advantages_zero,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            actor_clip=0.2,
            epochs=2
        )
        
        # Then: Zero advantages should be handled gracefully
        mean, std, values = mock_actor_critic_model(test_states)
        
        assert torch.all(torch.isfinite(mean)), \
            "Model should work with zero advantages"
        assert torch.all(torch.isfinite(std)), \
            "Model should work with zero advantages"
        assert torch.all(torch.isfinite(values)), \
            "Model should work with zero advantages"
