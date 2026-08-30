from .base import DexHand
from .inspire import Inspire

import os
from .factory import DexHandFactory

current_dir = os.path.dirname(os.path.abspath(__file__))
base_package = __name__

DexHandFactory.auto_register_hands(current_dir, base_package)
