"""
(description)
"""
import logging
import numpy as np

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.gym_env import GymEnv

from flow.core.params import *
from flow.core.vehicles import Vehicles

from flow.controllers.rlcontroller import RLController
from flow.controllers.car_following_models import *
from flow.controllers.lane_change_controllers import *
from flow.controllers.routing_controllers import ContinuousRouter

from flow.scenarios.two_loops_one_merging_new.gen import TwoLoopOneMergingGenerator
from flow.scenarios.two_loops_one_merging_new.scenario \
    import TwoLoopsOneMergingScenario


def run_task(*_):
    logging.basicConfig(level=logging.INFO)

    sumo_params = SumoParams(sim_step=0.1, sumo_binary="sumo-gui")

    # note that the vehicles are added sequentially by the generator,
    # so place the merging vehicles after the vehicles in the ring
    vehicles = Vehicles()
    vehicles.add_vehicles(veh_id="human",
                          acceleration_controller=(IDMController, {"noise": 0.2}),
                          lane_change_controller=(SumoLaneChangeController, {}),
                          routing_controller=(ContinuousRouter, {}),
                          num_vehicles=6,
                          sumo_car_following_params=SumoCarFollowingParams(minGap=0.0, tau=0.5),
                          sumo_lc_params=SumoLaneChangeParams())

    vehicles.add_vehicles(veh_id="merge-rl",
                          acceleration_controller=(RLController, {"fail_safe": "safe_velocity"}),
                          lane_change_controller=(SumoLaneChangeController, {}),
                          routing_controller=(ContinuousRouter, {}),
                          speed_mode="no_collide",
                          num_vehicles=10,
                          sumo_car_following_params=SumoCarFollowingParams(minGap=0.01, tau=0.5),
                          sumo_lc_params=SumoLaneChangeParams())

    additional_env_params = {"target_velocity": 20, "max-deacc": -1.5,
                             "max-acc": 1, "num_steps": 1000}
    env_params = EnvParams(additional_params=additional_env_params)

    additional_net_params = {"ring_radius": 50, "lanes": 1, "lane_length": 75,
                             "speed_limit": 30, "resolution": 40}
    net_params = NetParams(
        no_internal_links=False,
        additional_params=additional_net_params
    )

    initial_config = InitialConfig(
        x0=50,
        spacing="custom",
        additional_params={"merge_bunching": 0}
    )

    scenario = TwoLoopsOneMergingScenario(
        name="two-loop-one-merging",
        generator_class=TwoLoopOneMergingGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config
    )

    env_name = "TwoLoopsMergeEnv"
    pass_params = (env_name, sumo_params, vehicles, env_params, net_params,
                   initial_config, scenario)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
    env = normalize(env)

    logging.info("Experiment Set Up complete")

    print("experiment initialized")

    env = normalize(env)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 50, 25)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = TRPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=30000,
        max_path_length=horizon,
        # whole_paths=True,
        n_itr=2000,
        discount=0.999,
        # step_size=0.01,
    )
    algo.train()

exp_tag = "merges-mixed-rl"

for seed in [56]:  # , 1, 5, 10, 73]:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a
        # random seed will be used
        seed=seed,
        mode="local",
        exp_prefix=exp_tag,
        # python_command="/home/aboudy/anaconda2/envs/rllab-multiagent/bin/python3.5"
        # plot=True,
    )