from typing import Dict

from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class NullLowdimRunner(BaseLowdimRunner):
    """Offline runner for low-dimensional datasets without an environment."""

    def run(self, policy: BaseLowdimPolicy) -> Dict:
        return {}
