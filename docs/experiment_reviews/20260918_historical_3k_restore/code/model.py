import sys,runpy
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'eval'))
sys.path.insert(0,str(HERE))
import inference_dp_controller
runpy.run_path(str(ROOT/'eval/xjz_eval_strong_prior.py'),run_name='__main__')
