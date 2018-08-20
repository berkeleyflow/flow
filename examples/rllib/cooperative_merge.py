"""
Cooperative merging example, consisting of 1 learning agent and 6 additional
vehicles in an inner ring, and 10 vehicles in an outer ring attempting to
merge into the inner ring.
"""

import json

import ray
import ray.rllib.ppo as ppo
from ray.tune import run_experiments
from ray.tune.registry import register_env

from flow.controllers import RLController, IDMController, ContinuousRouter, \
    SumoLaneChangeController
from flow.core.params import SumoCarFollowingParams, SumoLaneChangeParams, \
    SumoParams, EnvParams, InitialConfig, NetParams
from flow.utils.registry import make_create_env
from flow.utils.rllib import FlowParamsEncoder
from flow.core.vehicles import Vehicles

# time horizon of a single rollout
HORIZON = 100
# number of rollouts per training iteration
N_ROLLOUTS = 10
# number of parallel workers
PARALLEL_ROLLOUTS = 2

RING_RADIUS = 100
NUM_MERGE_HUMANS = 9
NUM_MERGE_RL = 1

# note that the vehicles are added sequentially by the generator,
# so place the merging vehicles after the vehicles in the ring
vehicles = Vehicles()
# Inner ring vehicles
vehicles.add(veh_id="human",
             acceleration_controller=(IDMController, {"noise": 0.2}),
             lane_change_controller=(SumoLaneChangeController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=6,
             sumo_car_following_params=SumoCarFollowingParams(
                 minGap=0.0,
                 tau=0.5
             ),
             sumo_lc_params=SumoLaneChangeParams())
# A single learning agent in the inner ring
vehicles.add(veh_id="rl",
             acceleration_controller=(RLController, {}),
             lane_change_controller=(SumoLaneChangeController, {}),
             routing_controller=(ContinuousRouter, {}),
             speed_mode="no_collide",
             num_vehicles=1,
             sumo_car_following_params=SumoCarFollowingParams(
                 minGap=0.01,
                 tau=0.5
             ),
             sumo_lc_params=SumoLaneChangeParams())
# Outer ring vehicles
vehicles.add(veh_id="merge-human",
             acceleration_controller=(IDMController, {"noise": 0.2}),
             lane_change_controller=(SumoLaneChangeController, {}),
             routing_controller=(ContinuousRouter, {}),
             num_vehicles=10,
             sumo_car_following_params=SumoCarFollowingParams(
                 minGap=0.0,
                 tau=0.5
             ),
             sumo_lc_params=SumoLaneChangeParams())

flow_params = dict(
    # name of the experiment
    exp_tag="cooperative_merge",

    # name of the flow environment the experiment is running on
    env_name="TwoLoopsMergePOEnv",

    # name of the scenario class the experiment is running on
    scenario="TwoLoopsOneMergingScenario",

    # name of the generator used to create/modify network configuration files
    generator="TwoLoopOneMergingGenerator",

    # sumo-related parameters (see flow.core.params.SumoParams)
    sumo=SumoParams(
        sim_step=0.1,
        sumo_binary="sumo",
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "max_accel": 3,
            "max_decel": 3,
            "target_velocity": 10,
            "n_preceding": 2,
            "n_following": 2,
            "n_merging_in": 2,
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # scenario's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        no_internal_links=False,
        additional_params={
            "ring_radius": 50,
            "lane_length": 75,
            "inner_lanes": 1,
            "outer_lanes": 1,
            "speed_limit": 30,
            "resolution": 40,
        },
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.vehicles.Vehicles)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        x0=50,
        spacing="uniform",
        additional_params={
            "merge_bunching": 0,
        },
    ),
)


if __name__ == "__main__":
    ray.init(num_cpus=PARALLEL_ROLLOUTS, redirect_output=False)

    config = ppo.DEFAULT_CONFIG.copy()
    config["num_workers"] = PARALLEL_ROLLOUTS
    config["timesteps_per_batch"] = HORIZON * N_ROLLOUTS
    config["gamma"] = 0.999  # discount rate
    config["model"].update({"fcnet_hiddens": [16, 16, 16]})
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["sgd_batchsize"] = min(16 * 1024, config["timesteps_per_batch"])
    config["kl_target"] = 0.02
    config["num_sgd_iter"] = 10
    config["horizon"] = HORIZON

    # save the flow params for replay
    flow_json = json.dumps(flow_params, cls=FlowParamsEncoder, sort_keys=True,
                           indent=4)
    config['env_config']['flow_params'] = flow_json

    create_env, env_name = make_create_env(params=flow_params, version=0)

    # Register as rllib env
    register_env(env_name, create_env)

    trials = run_experiments({
        flow_params["exp_tag"]: {
            "run": "PPO",
            "env": env_name,
            "config": {
                **config
            },
            "checkpoint_freq": 20,
            "max_failures": 999,
            "stop": {
                "training_iteration": 200,
            },
            "trial_resources": {
                "cpu": 1,
                "gpu": 0,
                "extra_cpu": PARALLEL_ROLLOUTS - 1,
            },
        }
    })
