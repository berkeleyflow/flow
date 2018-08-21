"""
Cooperative merging example, consisting of 1 learning agent and 6 additional
vehicles in an inner ring, and 10 vehicles in an outer ring attempting to
merge into the inner ring. rllab version.
"""

from rllab.algos.trpo import TRPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from flow.controllers import RLController, IDMController, \
    SumoLaneChangeController, ContinuousRouter
from flow.core.params import SumoParams, EnvParams, NetParams, InitialConfig, \
    SumoCarFollowingParams, SumoLaneChangeParams
from flow.core.vehicles import Vehicles
from flow.scenarios.loop_merge.gen import TwoLoopOneMergingGenerator
from flow.scenarios.loop_merge.scenario import TwoLoopsOneMergingScenario, \
    ADDITIONAL_NET_PARAMS

HORIZON = 300


def run_task(*_):
    sumo_params = SumoParams(sim_step=0.2, sumo_binary="sumo-gui")

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

    env_params = EnvParams(
        horizon=HORIZON,
        additional_params={
            "max_accel": 3,
            "max_decel": 3,
            "target_velocity": 10,
            "n_preceding": 2,
            "n_following": 2,
            "n_merging_in": 2,
        }
    )

    additional_net_params = ADDITIONAL_NET_PARAMS.copy()
    additional_net_params["ring_radius"] = 50
    additional_net_params["inner_lanes"] = 1
    additional_net_params["outer_lanes"] = 1
    additional_net_params["lane_length"] = 75
    net_params = NetParams(
        no_internal_links=False,
        additional_params=additional_net_params
    )

    initial_config = InitialConfig(
        x0=50,
        spacing="uniform",
        additional_params={"merge_bunching": 0}
    )

    scenario = TwoLoopsOneMergingScenario(
        name=exp_tag,
        generator_class=TwoLoopOneMergingGenerator,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config
    )

    env_name = "TwoLoopsMergePOEnv"
    pass_params = (env_name, sumo_params, vehicles, env_params,
                   net_params, initial_config, scenario)

    env = GymEnv(env_name, record_video=False, register_params=pass_params)
    horizon = env.horizon
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
        batch_size=64 * 3 * horizon,
        max_path_length=horizon,
        # whole_paths=True,
        n_itr=1000,
        discount=0.999,
        # step_size=0.01,
    )
    algo.train()


exp_tag = "cooperative_merge_example"  # experiment prefix

for seed in [1]:  # , 5, 10, 56, 73]:
    run_experiment_lite(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="all",
        # Specifies the seed for the experiment. If this is not provided, a
        # random seed will be used
        seed=seed,
        mode="local",  # "ec2"
        exp_prefix=exp_tag,
        # plot=True,
    )
