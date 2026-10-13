"""Show the real start image beside the two actual static-run camera feeds."""

from pathlib import Path
import os
import cv2
import numpy as np

P = Path(__file__).resolve().parent
OUT = Path(os.environ.get('STATIC_RENDER_OUTPUT_DIR', str(P / 'static_unperturbed')))
DATA = Path('/home/carus/Data/Object_state_data')


def main():
    for episode, frame in [(51, 105), (53, 90)]:
        source = [cv2.imread(str(DATA / f'episode_{episode}' / view / f'{frame:06d}.png'))
                  for view in ['front', 'wrist']]
        source = [cv2.resize(im, (640, 480)) for im in source]
        cap = cv2.VideoCapture(str(OUT / f'episode{episode}_static.mp4'))
        assert cap.isOpened()
        writer = cv2.VideoWriter(str(OUT / f'episode{episode}_source_vs_static.mp4'),
                                 cv2.VideoWriter_fourcc(*'mp4v'), 30, (2560, 544))
        assert writer.isOpened()
        count = 0
        try:
            while True:
                ok, sim = cap.read()
                if not ok:
                    break
                assert sim.shape[:2] == (544, 1280)
                panel = np.zeros((544, 2560, 3), dtype=np.uint8)
                panel[64:, :640] = source[0]
                panel[64:, 640:1280] = source[1]
                panel[:, 1280:] = sim
                cv2.putText(panel, f'REAL ep{episode} f{frame} | frozen start frame',
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .68, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(panel, 'front', (10, 52), cv2.FONT_HERSHEY_SIMPLEX, .6,
                            (190, 230, 255), 1, cv2.LINE_AA)
                cv2.putText(panel, 'wrist', (650, 52), cv2.FONT_HERSHEY_SIMPLEX, .6,
                            (190, 230, 255), 1, cv2.LINE_AA)
                writer.write(panel)
                count += 1
        finally:
            cap.release()
            writer.release()
        assert count == 60, (episode, count)
        print(episode, count)


if __name__ == '__main__':
    main()
