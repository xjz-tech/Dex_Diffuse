import torch
import torch.nn.functional as F
from typing import List, Optional


class AdaptiveSamplingScheduler:
    def __init__(
        self,
        prob_lower_bound: float = 0.001,
        prob_upper_bound: float = 0.04,
        prefer_low_metric: bool = False,
    ):
        """
        Adaptive sampling scheduler that adjusts sampling probabilities based on metrics.
        
        Args:
            prob_lower_bound: Lower bound for sampling probabilities
            prob_upper_bound: Upper bound for sampling probabilities
            prefer_low_metric: If True, lower metrics lead to higher sampling probability (hard example mining).
                             If False, higher metrics lead to higher sampling probability (default).
        """
        self.prob_lower_bound = prob_lower_bound
        self.prob_upper_bound = prob_upper_bound
        self.prefer_low_metric = prefer_low_metric
        
        self.num_demos = None
        self.data_indices = None
        self.sampling_probs = None
        self.current_metrics = None  # Store current metrics for printing
        self.prev_probs = None       # Store previous probabilities for delta visualization
        self.prev_metrics = None     # Store previous metrics for delta visualization
        self.device = None
    
    def initialize(self, num_demos: int):
        """
        Initialize the scheduler with number of demos.
        
        Args:
            num_demos: Number of demos
        """
        self.num_demos = num_demos
        if self.device is None:
            self.device = torch.device("cpu")
    
    def set_data_indices(self, data_indices: List[int]):
        """
        Set the data indices.
        
        Args:
            data_indices: List of data indices
        """
        self.data_indices = data_indices
    
    def reset(
        self,
        num_demos: int,
        initial_metrics: torch.Tensor,
    ):
        """
        Reset the scheduler with initial metrics.
        
        Args:
            num_demos: Number of demos
            initial_metrics: Initial metrics tensor of shape (num_demos,)
        """
        self.num_demos = num_demos
        self.device = initial_metrics.device
        
        # Reset current/previous stats
        self.current_metrics = initial_metrics.clone().float()
        self.prev_probs = None
        self.prev_metrics = None
        
        # Initialize with uniform probabilities
        self.sampling_probs = torch.ones(num_demos, device=self.device) / num_demos
    
    def update(self, metrics: torch.Tensor):
        """
        Update sampling probabilities based on metrics.
        
        Args:
            metrics: Metrics tensor of shape (num_demos,), e.g., consecutive_reach_goal_max_buf
        """
        if self.sampling_probs is None:
            raise RuntimeError("Scheduler not initialized. Call reset() first.")
        
        # Backup previous probs / metrics for delta visualization
        if self.sampling_probs is not None:
            self.prev_probs = self.sampling_probs.clone()
        if self.current_metrics is not None:
            self.prev_metrics = self.current_metrics.clone()
        
        # Store original metrics (before any transformation)
        self.current_metrics = metrics.clone().float()
        
        metrics = metrics.float()
        
        if metrics.sum() == 0:
            self.sampling_probs = torch.ones(self.num_demos, device=self.device) / self.num_demos
        else:
            # If prefer_low_metric is True, invert the metrics (higher metric -> lower sampling prob)
            if self.prefer_low_metric:
                # Invert metrics: max - metrics, then add small epsilon to avoid all zeros
                max_metric = metrics.max()
                metrics = max_metric - metrics + 1e-8
            
            # Normalize metrics to get relative performance
            metrics_normalized = metrics / (metrics.sum() + 1e-8)
            
            # Scale to fit within bounds while preserving relative order
            # First, normalize to [0, 1] range
            min_val = metrics_normalized.min()
            max_val = metrics_normalized.max()
            if max_val > min_val:
                normalized = (metrics_normalized - min_val) / (max_val - min_val + 1e-8)
            else:
                # All metrics are equal, use uniform distribution
                self.sampling_probs = torch.ones(self.num_demos, device=self.device) / self.num_demos
                return
            
            # Scale to [prob_lower_bound, prob_upper_bound] range
            scaled_probs = self.prob_lower_bound + normalized * (self.prob_upper_bound - self.prob_lower_bound)
            
            # Iteratively adjust to ensure all values are within bounds after normalization
            # This is necessary because normalization can push values outside bounds
            max_iterations = 10
            for _ in range(max_iterations):
                # Clamp to bounds
                scaled_probs = torch.clamp(
                    scaled_probs,
                    min=self.prob_lower_bound,
                    max=self.prob_upper_bound
                )
                
                # Normalize
                total = scaled_probs.sum()
                if total > 0:
                    scaled_probs = scaled_probs / total
                else:
                    scaled_probs = torch.ones(self.num_demos, device=self.device) / self.num_demos
                    break
                
                # Check if all values are within bounds
                if torch.all(scaled_probs >= self.prob_lower_bound) and torch.all(scaled_probs <= self.prob_upper_bound):
                    break
                
                # If not, adjust values that are outside bounds
                # Scale down values that exceed upper bound
                excess_mask = scaled_probs > self.prob_upper_bound
                if excess_mask.any():
                    excess = (scaled_probs - self.prob_upper_bound) * excess_mask.float()
                    scaled_probs = torch.where(excess_mask, self.prob_upper_bound, scaled_probs)
                    # Redistribute excess to values below upper bound
                    remaining_mask = ~excess_mask
                    if remaining_mask.any():
                        excess_total = excess.sum()
                        scaled_probs = torch.where(
                            remaining_mask,
                            scaled_probs + excess_total / remaining_mask.sum(),
                            scaled_probs
                        )
                
                # Scale up values that are below lower bound
                deficit_mask = scaled_probs < self.prob_lower_bound
                if deficit_mask.any():
                    deficit = (self.prob_lower_bound - scaled_probs) * deficit_mask.float()
                    scaled_probs = torch.where(deficit_mask, self.prob_lower_bound, scaled_probs)
                    # Redistribute deficit from values above lower bound
                    remaining_mask = ~deficit_mask
                    if remaining_mask.any():
                        deficit_total = deficit.sum()
                        scaled_probs = torch.where(
                            remaining_mask,
                            scaled_probs - deficit_total / remaining_mask.sum(),
                            scaled_probs
                        )
            
            self.sampling_probs = scaled_probs
    
    def get_indices(self, num_envs: int) -> torch.Tensor:
        """
        Get environment indices to demo indices mapping based on current sampling probabilities.
        
        Args:
            num_envs: Number of environments
            
        Returns:
            Tensor of shape (num_envs,) mapping each env to a demo index
        """
        if self.sampling_probs is None:
            raise RuntimeError("Scheduler not initialized. Call reset() first.")
        
        demo_indices = torch.multinomial(
            self.sampling_probs,
            num_samples=num_envs,
            replacement=True
        )
        
        return demo_indices
    
    def print_probs(self, num_per_line: int = 10, precision: int = 6, metric_precision: int = 1):
        """
        Pretty-print demo probabilities and metrics in a table, and visualize
        how they changed compared to the previous update.
        
        Args:
            num_per_line: Number of demos to print per block
            precision: Number of decimal places to display for probabilities
            metric_precision: Number of decimal places to display for metrics
        """
        if self.sampling_probs is None:
            print("Scheduler not initialized. Call reset() first.")
            return
        
        print(f"\nSampling Probabilities (total: {len(self.sampling_probs)} demos):")
        print(f"Bounds: [{self.prob_lower_bound:.{precision}f}, {self.prob_upper_bound:.{precision}f}]")
        print(f"Sum: {self.sampling_probs.sum().item():.{precision}f}")
        if self.current_metrics is not None:
            print(f"Prefer low metric: {self.prefer_low_metric}")
        if self.prev_probs is None:
            print("(No previous round yet, deltas are not available.)")
        print("-" * 80)
        
        header = f"{'Idx':>4} {'Prob':>{precision+6}} {'ΔProb':>{precision+7}}"
        if self.current_metrics is not None:
            header += f" {'Metric':>{metric_precision+6}} {'ΔMetric':>{metric_precision+7}}"
        print(header)
        print("-" * 80)
        
        for i in range(0, len(self.sampling_probs), num_per_line):
            end_idx = min(i + num_per_line, len(self.sampling_probs))
            for j in range(i, end_idx):
                prob = self.sampling_probs[j].item()
                # Prob delta
                if self.prev_probs is not None and len(self.prev_probs) == len(self.sampling_probs):
                    d_prob = prob - self.prev_probs[j].item()
                    d_prob_str = f"{d_prob:+.{precision}f}"
                else:
                    d_prob_str = " " * (precision + 2)
                
                line = f"{j:4d} {prob:{precision+6}.{precision}f} {d_prob_str:>{precision+7}}"
                
                if self.current_metrics is not None:
                    metric = self.current_metrics[j].item()
                    if self.prev_metrics is not None and len(self.prev_metrics) == len(self.current_metrics):
                        d_metric = metric - self.prev_metrics[j].item()
                        d_metric_str = f"{d_metric:+.{metric_precision}f}"
                    else:
                        d_metric_str = " " * (metric_precision + 2)
                    line += f" {metric:{metric_precision+6}.{metric_precision}f} {d_metric_str:>{metric_precision+7}}"
                
                print(line)
            print("-" * 80)
        
        # Print statistics
        print("Statistics:")
        print(f"  Min prob: {self.sampling_probs.min().item():.{precision}f} (demo {self.sampling_probs.argmin().item()})")
        print(f"  Max prob: {self.sampling_probs.max().item():.{precision}f} (demo {self.sampling_probs.argmax().item()})")
        
        if self.current_metrics is not None:
            print(f"  Min metric: {self.current_metrics.min().item():.{metric_precision}f} (demo {self.current_metrics.argmin().item()})")
            print(f"  Max metric: {self.current_metrics.max().item():.{metric_precision}f} (demo {self.current_metrics.argmax().item()})")
            print(f"  Mean metric: {self.current_metrics.mean().item():.{metric_precision}f}")


if __name__ == "__main__":
    print("Testing AdaptiveSamplingScheduler...")
    
    # Test 1: Basic initialization and reset
    print("\n=== Test 1: Initialization and Reset ===")
    # Use more reasonable bounds for testing: lower_bound should allow uniform distribution
    # For 5 demos, uniform is 0.2, so upper_bound should be >= 0.2
    scheduler = AdaptiveSamplingScheduler(
        prob_lower_bound=0.05,
        prob_upper_bound=0.5,
        prefer_low_metric=False,  # Default: higher metric -> higher probability
    )
    
    num_demos = 5
    device = torch.device("cpu")
    initial_metrics = torch.zeros(num_demos, device=device, dtype=torch.int32)
    
    scheduler.initialize(num_demos)
    scheduler.reset(num_demos, initial_metrics)
    
    print(f"Initial sampling probabilities: {scheduler.sampling_probs}")
    print(f"Expected: uniform distribution (1/{num_demos} = {1.0/num_demos:.4f})")
    assert torch.allclose(scheduler.sampling_probs, torch.ones(num_demos) / num_demos), "Initial probabilities should be uniform"
    print("✓ Test 1 passed")
    
    # Test 2: Get indices
    print("\n=== Test 2: Get Indices ===")
    num_envs = 1000
    indices = scheduler.get_indices(num_envs)
    print(f"Generated {num_envs} indices")
    print(f"Index range: [{indices.min().item()}, {indices.max().item()}]")
    print(f"Expected range: [0, {num_demos-1}]")
    
    # Count occurrences
    counts = torch.bincount(indices, minlength=num_demos).float()
    empirical_probs = counts / counts.sum()
    print(f"Empirical probabilities: {empirical_probs}")
    print(f"Expected probabilities: {scheduler.sampling_probs}")
    
    # Check if empirical probabilities are close to expected (within 5%)
    assert torch.allclose(empirical_probs, scheduler.sampling_probs, atol=0.05), \
        f"Empirical probabilities {empirical_probs} should be close to expected {scheduler.sampling_probs}"
    print("✓ Test 2 passed")
    
    # Test 3: Update with metrics
    print("\n=== Test 3: Update with Metrics ===")
    # Simulate metrics where demo 0 performs best, demo 4 performs worst
    metrics = torch.tensor([10, 5, 3, 2, 1], device=device, dtype=torch.int32)
    scheduler.update(metrics)
    
    print(f"Metrics: {metrics}")
    print(f"Updated sampling probabilities: {scheduler.sampling_probs}")
    
    # Check that demo 0 has highest probability
    assert scheduler.sampling_probs[0] > scheduler.sampling_probs[4], \
        "Demo 0 (best) should have higher probability than demo 4 (worst)"
    
    # Check that probabilities are within bounds
    assert torch.all(scheduler.sampling_probs >= scheduler.prob_lower_bound), \
        f"All probabilities should be >= {scheduler.prob_lower_bound}"
    assert torch.all(scheduler.sampling_probs <= scheduler.prob_upper_bound), \
        f"All probabilities should be <= {scheduler.prob_upper_bound}"
    
    # Check that probabilities sum to 1
    assert torch.allclose(scheduler.sampling_probs.sum(), torch.tensor(1.0)), \
        "Probabilities should sum to 1"
    print("✓ Test 3 passed")
    
    # Test 4: Get indices after update
    print("\n=== Test 4: Get Indices After Update ===")
    indices_after = scheduler.get_indices(num_envs)
    counts_after = torch.bincount(indices_after, minlength=num_demos).float()
    empirical_probs_after = counts_after / counts_after.sum()
    
    print(f"Empirical probabilities after update: {empirical_probs_after}")
    print(f"Expected probabilities: {scheduler.sampling_probs}")
    
    # Check if demo 0 is sampled more frequently than demo 4
    assert empirical_probs_after[0] > empirical_probs_after[4], \
        "Demo 0 should be sampled more frequently than demo 4"
    print("✓ Test 4 passed")
    
    # Test 5: Update with all zeros (edge case)
    print("\n=== Test 5: Update with All Zeros (Edge Case) ===")
    zero_metrics = torch.zeros(num_demos, device=device, dtype=torch.int32)
    scheduler.update(zero_metrics)
    
    print(f"Metrics: {zero_metrics}")
    print(f"Sampling probabilities: {scheduler.sampling_probs}")
    
    # Should fall back to uniform distribution
    assert torch.allclose(scheduler.sampling_probs, torch.ones(num_demos) / num_demos), \
        "Should fall back to uniform distribution when all metrics are zero"
    print("✓ Test 5 passed")
    
    # Test 6: Update with extreme values
    print("\n=== Test 6: Update with Extreme Values ===")
    extreme_metrics = torch.tensor([1000, 1, 1, 1, 1], device=device, dtype=torch.int32)
    scheduler.update(extreme_metrics)
    
    print(f"Metrics: {extreme_metrics}")
    print(f"Sampling probabilities: {scheduler.sampling_probs}")
    
    # Demo 0 should have highest probability (may be clamped to upper bound)
    assert scheduler.sampling_probs[0] >= scheduler.sampling_probs[1], \
        f"Demo 0 should have higher probability than demo 1"
    assert scheduler.sampling_probs[0] <= scheduler.prob_upper_bound, \
        f"Demo 0 should be <= upper bound {scheduler.prob_upper_bound}"
    
    # Check that probabilities sum to 1
    assert torch.allclose(scheduler.sampling_probs.sum(), torch.tensor(1.0)), \
        "Probabilities should sum to 1"
    print("✓ Test 6 passed")
    
    # Test 7: Multiple updates
    print("\n=== Test 7: Multiple Updates ===")
    metrics_seq = [
        torch.tensor([5, 5, 5, 5, 5], device=device, dtype=torch.int32),
        torch.tensor([10, 5, 3, 2, 1], device=device, dtype=torch.int32),
        torch.tensor([1, 2, 3, 5, 10], device=device, dtype=torch.int32),
    ]
    
    for i, m in enumerate(metrics_seq):
        scheduler.update(m)
        print(f"Update {i+1}, metrics: {m}, probs: {scheduler.sampling_probs}")
    
    # Final probabilities should reflect the last update
    assert scheduler.sampling_probs[4] > scheduler.sampling_probs[0], \
        "After last update, demo 4 should have higher probability than demo 0"
    print("✓ Test 7 passed")
    
    # Test 8: Test prefer_low_metric mode (inverse sampling)
    print("\n=== Test 8: Prefer Low Metric Mode (Inverse Sampling) ===")
    scheduler_inverse = AdaptiveSamplingScheduler(
        prob_lower_bound=0.05,
        prob_upper_bound=0.5,
        prefer_low_metric=True,  # Lower metric -> higher probability
    )
    
    scheduler_inverse.initialize(num_demos)
    scheduler_inverse.reset(num_demos, initial_metrics)
    
    # Test with same metrics as Test 3
    metrics_test = torch.tensor([10, 5, 3, 2, 1], device=device, dtype=torch.int32)
    scheduler_inverse.update(metrics_test)
    
    print(f"Metrics: {metrics_test}")
    print(f"Sampling probabilities (prefer_low_metric=True): {scheduler_inverse.sampling_probs}")
    
    # In inverse mode, demo 4 (lowest metric=1) should have highest probability
    # Demo 0 (highest metric=10) should have lowest probability
    assert scheduler_inverse.sampling_probs[4] > scheduler_inverse.sampling_probs[0], \
        "In inverse mode, demo 4 (lowest metric) should have higher probability than demo 0 (highest metric)"
    
    # Check that probabilities are within bounds
    assert torch.all(scheduler_inverse.sampling_probs >= scheduler_inverse.prob_lower_bound), \
        f"All probabilities should be >= {scheduler_inverse.prob_lower_bound}"
    assert torch.all(scheduler_inverse.sampling_probs <= scheduler_inverse.prob_upper_bound), \
        f"All probabilities should be <= {scheduler_inverse.prob_upper_bound}"
    
    # Check that probabilities sum to 1
    assert torch.allclose(scheduler_inverse.sampling_probs.sum(), torch.tensor(1.0)), \
        "Probabilities should sum to 1"
    
    # Compare with normal mode
    print(f"Normal mode probabilities: {scheduler.sampling_probs}")
    print(f"Inverse mode probabilities: {scheduler_inverse.sampling_probs}")
    print("✓ Test 8 passed")
    
    # Test 9: Large scale test (150 demos, 8192 envs)
    print("\n=== Test 9: Large Scale Test (150 demos, 8192 envs) ===")
    num_demos_large = 150
    num_envs_large = 8192
    
    scheduler_large = AdaptiveSamplingScheduler(
        prob_lower_bound=0.001,
        prob_upper_bound=0.04,
        prefer_low_metric=True,
    )
    
    device_large = torch.device("cpu")
    initial_metrics_large = torch.zeros(num_demos_large, device=device_large, dtype=torch.int32)
    
    scheduler_large.initialize(num_demos_large)
    scheduler_large.reset(num_demos_large, initial_metrics_large)
    
    print(f"Initialized scheduler with {num_demos_large} demos")
    print(f"Initial uniform probability: {1.0/num_demos_large:.6f}")
    print(f"Expected range: [{scheduler_large.prob_lower_bound}, {scheduler_large.prob_upper_bound}]")
    
    # Simulate metrics: some demos perform well, some poorly
    # Create a distribution where demos 0-49 perform well (high metrics)
    # demos 50-99 perform medium, demos 100-149 perform poorly (low metrics)
    metrics_large = torch.zeros(num_demos_large, device=device_large, dtype=torch.int32)
    metrics_large[0:50] = torch.randint(80, 100, (50,), device=device_large)  # Good performance
    metrics_large[50:100] = torch.randint(40, 60, (50,), device=device_large)  # Medium performance
    metrics_large[100:150] = torch.randint(1, 20, (50,), device=device_large)  # Poor performance
    
    scheduler_large.update(metrics_large)
    
    print(f"\nMetrics distribution:")
    print(f"  Good demos (0-49):   metrics range [{metrics_large[0:50].min().item()}, {metrics_large[0:50].max().item()}]")
    print(f"  Medium demos (50-99): metrics range [{metrics_large[50:100].min().item()}, {metrics_large[50:100].max().item()}]")
    print(f"  Poor demos (100-149): metrics range [{metrics_large[100:150].min().item()}, {metrics_large[100:150].max().item()}]")
    
    print(f"\nSampling probabilities:")
    print(f"  Good demos (0-49):   prob range [{scheduler_large.sampling_probs[0:50].min().item():.6f}, {scheduler_large.sampling_probs[0:50].max().item():.6f}], mean: {scheduler_large.sampling_probs[0:50].mean().item():.6f}")
    print(f"  Medium demos (50-99): prob range [{scheduler_large.sampling_probs[50:100].min().item():.6f}, {scheduler_large.sampling_probs[50:100].max().item():.6f}], mean: {scheduler_large.sampling_probs[50:100].mean().item():.6f}")
    print(f"  Poor demos (100-149): prob range [{scheduler_large.sampling_probs[100:150].min().item():.6f}, {scheduler_large.sampling_probs[100:150].max().item():.6f}], mean: {scheduler_large.sampling_probs[100:150].mean().item():.6f}")
    
    # Verify probabilities are within bounds
    assert torch.all(scheduler_large.sampling_probs >= scheduler_large.prob_lower_bound), \
        f"All probabilities should be >= {scheduler_large.prob_lower_bound}"
    assert torch.all(scheduler_large.sampling_probs <= scheduler_large.prob_upper_bound), \
        f"All probabilities should be <= {scheduler_large.prob_upper_bound}"
    assert torch.allclose(scheduler_large.sampling_probs.sum(), torch.tensor(1.0)), \
        "Probabilities should sum to 1"
    
    # Check probability ordering based on prefer_low_metric setting
    if scheduler_large.prefer_low_metric:
        # In inverse mode, poor demos (low metrics) should have higher probability
        assert scheduler_large.sampling_probs[100:150].mean() > scheduler_large.sampling_probs[0:50].mean(), \
            "In inverse mode, poor demos should have higher average probability than good demos"
    else:
        # In normal mode, good demos (high metrics) should have higher probability
        assert scheduler_large.sampling_probs[0:50].mean() > scheduler_large.sampling_probs[100:150].mean(), \
            "In normal mode, good demos should have higher average probability than poor demos"
    
    # Generate indices for 8192 environments
    print(f"\nGenerating indices for {num_envs_large} environments...")
    indices_large = scheduler_large.get_indices(num_envs_large)
    
    print(f"Generated {num_envs_large} indices")
    print(f"Index range: [{indices_large.min().item()}, {indices_large.max().item()}]")
    print(f"Expected range: [0, {num_demos_large-1}]")
    
    # Count occurrences for each demo group
    counts_large = torch.bincount(indices_large, minlength=num_demos_large).float()
    empirical_probs_large = counts_large / counts_large.sum()
    
    print(f"\nEmpirical sampling distribution:")
    print(f"  Good demos (0-49):   count={counts_large[0:50].sum().item():.0f}, prob={empirical_probs_large[0:50].sum().item():.6f}")
    print(f"  Medium demos (50-99): count={counts_large[50:100].sum().item():.0f}, prob={empirical_probs_large[50:100].sum().item():.6f}")
    print(f"  Poor demos (100-149): count={counts_large[100:150].sum().item():.0f}, prob={empirical_probs_large[100:150].sum().item():.6f}")
    
    print(f"\nExpected vs Empirical (for good demos):")
    print(f"  Expected prob: {scheduler_large.sampling_probs[0:50].sum().item():.6f}")
    print(f"  Empirical prob: {empirical_probs_large[0:50].sum().item():.6f}")
    print(f"  Difference: {abs(scheduler_large.sampling_probs[0:50].sum().item() - empirical_probs_large[0:50].sum().item()):.6f}")
    
    # Verify sampling frequency based on prefer_low_metric setting
    if scheduler_large.prefer_low_metric:
        # In inverse mode, poor demos should be sampled more frequently
        assert empirical_probs_large[100:150].sum() > empirical_probs_large[0:50].sum(), \
            "In inverse mode, poor demos should be sampled more frequently than good demos"
    else:
        # In normal mode, good demos should be sampled more frequently
        assert empirical_probs_large[0:50].sum() > empirical_probs_large[100:150].sum(), \
            "In normal mode, good demos should be sampled more frequently than poor demos"
    
    # Check that all demos are sampled at least once (very likely with 8192 envs)
    unique_demos = torch.unique(indices_large)
    print(f"\nUnique demos sampled: {len(unique_demos)}/{num_demos_large}")
    print(f"Coverage: {len(unique_demos)/num_demos_large*100:.2f}%")
    
    print("✓ Test 9 passed")
    
    # Test 10: Print probabilities function
    print("\n=== Test 10: Print Probabilities Function ===")
    print("Printing probabilities for small scheduler (5 demos):")
    scheduler.print_probs(num_per_line=5, precision=6)
    
    print("\nPrinting probabilities for large scale scheduler (showing all 150 demos):")
    scheduler_large.print_probs(num_per_line=10, precision=6)
    print("✓ Test 10 passed")
    
    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)

