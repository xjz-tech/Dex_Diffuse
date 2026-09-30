import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('handoff', Path(__file__).with_name('resume_after_video.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class HandoffTests(unittest.TestCase):
    def test_end_conditions(self):
        with tempfile.TemporaryDirectory(prefix='video-handoff-test-') as folder:
            original_root = m.ROOT
            m.ROOT = Path(folder)
            try:
                video = m.ROOT/'test.mp4'
                video.write_bytes(b'path-check-only; decode is a separate mandatory gate')
                props = dict(InvocationID=m.INVOCATION, ActiveState='failed', ExecMainStatus='139')
                log = f'[sim] reached max failure episodes: 1\n[record] STOP | frames=20 duration=1.0s saved={video}\n'
                rows = [dict(reason='failure', episode=0)]
                self.assertEqual(m.validate_terminal(props, log, rows), video)
                for replacement in (dict(ActiveState='active'), dict(InvocationID='different'), dict(ExecMainStatus='137')):
                    with self.assertRaises(RuntimeError):
                        m.validate_terminal(dict(props, **replacement), log, rows)
                for bad_rows in ([], [dict(reason='timeout', episode=0)], rows+rows):
                    with self.assertRaises(RuntimeError):
                        m.validate_terminal(props, log, bad_rows)
                with self.assertRaises(RuntimeError):
                    m.validate_terminal(props, '', rows)
            finally:
                m.ROOT = original_root


if __name__ == '__main__':
    unittest.main()
