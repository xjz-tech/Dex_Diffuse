from diffusion_policy.env_runner.null_lowdim_runner import NullLowdimRunner


def test_null_lowdim_runner_has_no_rollout_metrics(tmp_path):
    runner = NullLowdimRunner(output_dir=str(tmp_path))

    assert runner.run(policy=object()) == {}
