"""Record Episode 54 at 170 g / mu=2.0 using the verified three-way runner."""
import run_episode54_m130_mu24_threeway_video as experiment

experiment.BASE = (experiment.R / 'm170_mu20_guidance4_vs_edit015_20260928' /
                   'episode_54')
experiment.OUT = experiment.R / 'm170_mu20_episode54_fourway_video_20260928'
experiment.MASS = .170
experiment.FRICTION = 2.0

if __name__ == '__main__':
    experiment.main()
