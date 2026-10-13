"""CPU-only checks of delayed replies, action order and the video frame cap."""
import ast
import importlib.util
from pathlib import Path
import threading
import time
import types
import numpy as np

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('wait0_plan', root/'code/eval/wait0_plan.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
gate = threading.Event()
entered = threading.Event()
seen = []
def predict(history):
    seen.append(history.copy())
    entered.set()
    assert gate.wait(2)
    return np.array([[[1., 2.], [3., 4.]]]), .01

initial = np.array([[7., 8.]])
history = np.zeros((1, 4, 6))
plan = module.HoldLastPlan(predict, initial)
try:
    assert np.array_equal(plan.next_target(history), initial)
    assert entered.wait(2)
    history[:] = 99
    for _ in range(5):
        assert np.array_equal(plan.next_target(history), initial)
    assert np.all(seen[0] == 0), 'in-flight observation must be immutable'
    gate.set()
    plan.pending.result(timeout=2)
    assert np.array_equal(plan.next_target(history), [[1., 2.]])
    assert np.array_equal(plan.next_target(history), [[3., 4.]])
    assert plan.hold_steps == 6 and plan.action_steps == 2 and plan.calls == 1
    gate.clear()
    assert np.array_equal(plan.next_target(history), [[3., 4.]])
finally:
    gate.set()
    plan.close()

# Execute the real recorder class with fake transport; no Isaac Gym or GPU.
tree = ast.parse((root/'code/eval/recording.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Mp4Recorder')
future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
namespace = dict(np=np, time=types.SimpleNamespace(monotonic=lambda: 305.),
                 os=types.SimpleNamespace(environ={'MAX_VIDEO_SECONDS':'300'}))
exec(compile(ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[])), '<recorder-test>', 'exec'), namespace)
rec = namespace['Mp4Recorder'].__new__(namespace['Mp4Recorder'])
rec.config = types.SimpleNamespace(width=2, height=2, fps=30)
rec._start_monotonic = 0.
rec._pending_payload = b'previous'
rec.frame_count = 8999
rec.process = object()
rec._process = object()
rec._writer_error = None
def enqueue(payload, count):
    assert count >= 0
    rec.frame_count += count
rec._enqueue_payload = enqueue
rec.write(np.zeros((2,2,3), dtype=np.uint8), captured_at=305.)
assert rec.frame_count == 9000
print('PASS: nonblocking hold, immutable history, ordered two-step execution, 300-second frame cap')
