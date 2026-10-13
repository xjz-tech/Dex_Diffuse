"""Run the matched comparison at 130 g and friction 2.4."""
import run_m170_mu20_guidance4_vs_edit015 as experiment

experiment.O = experiment.R / 'm130_mu24_guidance4_vs_edit015_20260928'
experiment.MASS = .130
experiment.FRICTION = 2.4

if __name__ == '__main__':
    experiment.main()
