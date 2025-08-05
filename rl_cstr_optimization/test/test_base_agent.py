import pytest
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Import the ActorCriticNet class from the base_agent module
from rl.cstr.optimization.base_agent import ActorCriticNet


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

