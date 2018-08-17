import json
import unittest
import os

os.environ["TEST_FLAG"] = "True"

import ray
import ray.rllib.ppo as ppo
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder

# use this to specify the environment to run

# number of rollouts per training iteration
N_ROLLOUTS = 2
# number of parallel workers
PARALLEL_ROLLOUTS = 2

class TestRllibBenchmarks(unittest.TestCase):
    """Test that the benchmarks run"""
    def test_grid(self):
        from flow.benchmarks.grid0 import flow_params
        # get the env name and a creator for the environment
        create_env, env_name = make_create_env(params=flow_params, version=0)

        # initialize a ray instance
        ray.init(redirect_output=False)

        horizon = 64
        config = ppo.DEFAULT_CONFIG.copy()
        config["num_workers"] = PARALLEL_ROLLOUTS
        config["min_steps_per_task"] = horizon
        config["timesteps_per_batch"] = horizon * N_ROLLOUTS
        config["vf_loss_coeff"] = 1.0
        config["kl_target"] = 0.02
        config["use_gae"] = True
        config["horizon"] = horizon
        config["clip_param"] = 0.2
        config["sgd_batchsize"] = horizon

        # save the flow params for replay
        flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                               indent=4)
        config['env_config']['flow_params'] = flow_json

        # Register as rllib env
        register_env(env_name, create_env)

        trials = run_experiments({
            flow_params["exp_tag"]: {
                "run": "PPO",
                "env": env_name,
                "config": {
                    **config
                },
                "checkpoint_freq": 5,
                "max_failures": 999,
                "stop": {"training_iteration": 1},
                "repeat": 1,
                "trial_resources": {
                    "cpu": 1,
                    "gpu": 0,
                    "extra_cpu": PARALLEL_ROLLOUTS - 1,
                },
            },
        })


if __name__ == '__main__':
    unittest.main()
