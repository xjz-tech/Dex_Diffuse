"""Runtime import for the formal SDEdit controller."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_policy.SDEdit.reference_action_editor import (  # noqa: E402
    ReferenceActionEditor,
    ddim_transition,
)

__all__ = ["ReferenceActionEditor", "ddim_transition"]
