"""View a bulb2 initialization demonstration using the existing Viser player.

Run with the decv2 Python environment; accepts --index 000 and --port 8098.
This replays recorded poses on CPU, without running a policy rollout.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, default=0, choices=range(150))
    parser.add_argument('--port', type=int, default=8098)
    parser.add_argument('--controller-root', type=Path,
                        default=Path('/home/carus/Program/dex-controller'))
    args = parser.parse_args()
    root = args.controller_root.resolve()
    index = f'{args.index:03d}'
    pkl = root / f'data/retargeting/NOKOV-v3/mano2sharpa_rh/bulb2/{index}.pkl'
    h5 = root / f'data/NOKOV-v3/data/bulb2/{index}.h5'
    urdf = root / 'maniptrans_envs/assets/sharpa_hand/v3right_sharpa_wave-forhammer5.urdf'
    for path in (pkl, h5, urdf):
        if not path.is_file():
            parser.error(f'Missing file: {path}')
    os.chdir(root)
    sys.path.insert(0, str(root))
    from isaacgym import gymapi  # Must precede any torch import.
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from maniptrans_envs.lib.envs.dexhands.factory import DexHandFactory
    original = DexHandFactory._registry['sharpa_rh']

    class LocalSharpa(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._urdf_path = str(urdf)

    DexHandFactory._registry['sharpa_rh'] = LocalSharpa
    sys.argv = [str(root / 'main/dataset/vis_retargeting_pkl.py'),
                '--pkl_path', str(pkl), '--object_h5', str(h5),
                '--object_urdf', str(root / 'data/NOKOV-v3/object/bulb2.urdf'),
                '--dexhand', 'sharpa', '--side', 'right', '--port', str(args.port)]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
