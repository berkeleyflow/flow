from flow.envs.base_env import Env
from gym.spaces.box import Box
import numpy as np
import collections
import random
import traci.constants as tc

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 2,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 2,
    # time delay for the execution of actions, in seconds. Note that actions
    # can only be executing in increments of the time step.
    "accel_delay": 0,
    # standard deviation of the gaussian exogenous noise added to the
    # acceleration actions, in m/s^2
    "accel_noise": 1,
    # standard deviation of the gaussian exogenous noise added to the speed
    # observations, in m/s
    "speed_noise": 1,
    # standard deviation of the gaussian exogenous noise added to the position
    # observations, in meters
    "position_noise": 0,
    # desired time headway for the autonomous vehicles, in seconds
    "target_headway": 3,
    # probability of a vehicle cutting in at any time step
    "cut_in_prob": 0.01,
    # probability of a cut-in leaving at any time step
    "cut_out_prob": 0.01,
}


class PlatooningEnv(Env):
    """Fully observable, single agent platooning environment

    A leading vehicle is provided a trajectory, and the platooning vehicles are
    told to match the speed of the leading vehicle as well as maintain a
    desirable headway with its immediate leader.

    Note that, in this environment, vehicle cut-in and cut-out behavior is
    simulated by periodically reducing simulating new vehicles in front of the
    autonomous vehicles through the observation.

    Required from env_params

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * accel_delay: time delay for the execution of actions, in seconds. Note
      that actions can only be executing in increments of the time step.
    * accel_noise: standard deviation of the gaussian exogenous noise added to
      the acceleration actions, in m/s^2
    * speed_noise: standard deviation of the gaussian exogenous noise added to
      the speed observations, in m/s
    * position_noise: standard deviation of the gaussian exogenous noise added
      to the position observations, in meters
    * target_headway: desired time headway for the autonomous vehicles, in
      seconds

    States
        States are the headways for all autonomous vehicles as well as the
        speeds of the autonomous vehicles and the leader.

    Actions
        Actions are a list of accelerations for each autonomous vehicle.

    Rewards
        The reward is the two-norm from the desired time headway. Note that the
        reward is always positive to penalize early terminations of a rollout
        due vehicle-to-vehicle collisions.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

    def __init__(self, env_params, sumo_params, scenario):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError('Environment parameter "{}" not supplied'.
                               format(p))

        super().__init__(env_params, sumo_params, scenario)

        # initialize an empty acceleration queue
        self.accel_queue = collections.deque()

        # number of actions performed before the most recent set of actions
        self.delay_size = int(self.env_params.additional_params["accel_delay"]
                              / self.sim_step)

        # number of cut-in vehicles during the current rollout
        self.num_cut_in = 0

        # list of names for vehicle that are currently cut-in in front of an RL
        # vehicle. If None, then there is no cut-in vehicle
        self.cut_in = [None for _ in range(self.vehicles.num_rl_vehicles)]

        # vehicle cut-in probability. If a vehicle has already cut in, no new
        # vehicles are assumed to also cut-in in this environment.
        self.cut_in_prob = self.env_params.additional_params["cut_in_prob"]

        # vehicle cut-out probability. Only applies to cut-in vehicles
        self.cut_out_prob = self.env_params.additional_params["cut_out_prob"]

    @property
    def action_space(self):
        return Box(low=-abs(self.env_params.additional_params["max_decel"]),
                   high=abs(self.env_params.additional_params["max_accel"]),
                   shape=(self.vehicles.num_rl_vehicles,),
                   dtype=np.float32)

    @property
    def observation_space(self):
        return Box(low=0,
                   high=float("inf"),
                   shape=(2 * self.vehicles.num_rl_vehicles + 1,),
                   dtype=np.float32)

    def _apply_rl_actions(self, rl_actions):
        # add the action to the queue of actions
        self.accel_queue.append(rl_actions)

        # get the next set of actions in the queue
        actions = self.accel_queue.pop()

        # add acceleration noise to the actions
        actions = np.array(actions) + np.random.normal(
            loc=0, scale=self.env_params.additional_params["accel_noise"],
            size=self.vehicles.num_rl_vehicles)

        # execute the above actions
        self.apply_acceleration(self.vehicles.get_rl_ids(), actions)

    def additional_command(self):
        # add noise to the position, headway, and velocity observations of all
        # vehicles in accordance with the noise parameters
        speed_dev = self.env_params.additional_params["speed_noise"]
        pos_dev = self.env_params.additional_params["position_noise"]
        for veh_id in self.vehicles.get_ids():
            self.vehicles.test_set_speed(
                veh_id=veh_id,
                speed=self.vehicles.get_speed(veh_id)
                + np.random.normal(0, speed_dev))
            self.vehicles.test_set_position(
                veh_id=veh_id,
                position=self.vehicles.get_position(veh_id)
                + np.random.normal(0, pos_dev))
            self.vehicles.set_headway(
                veh_id=veh_id,
                headway=self.vehicles.get_headway(veh_id)
                + np.random.normal(0, pos_dev))

        # list of sorted RL vehicles (this is used in the next step)
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
                         if veh_id in self.vehicles.get_rl_ids()]

        # simulate entering/exiting vehicles
        for i in range(self.vehicles.num_rl_vehicles):
            if self.cut_in[i] is None:
                # check a vehicle is cutting-in in this time step
                if random.uniform(0, 1) < self.cut_out_prob:
                    # name of the entering vehicle
                    self.cut_in[i] = "cut_in{}".format(self.num_cut_in)

                    # the entering vehicle is placed halfway between the
                    # lagging vehicle and the one ahead of it
                    lag_veh = sorted_rl_ids[i]
                    new_pos = (self.get_x_by_id(lag_veh) +
                               self.vehicles.get_headway(lag_veh) / 2) \
                        % self.scenario.length
                    edge, rel_pos = self.scenario.get_edge(new_pos)

                    self.traci_connection.vehicle.add(
                        "cut_in{}".format(self.num_cut_in),
                        "route{}".format(edge),
                        typeID="cut_in",
                        lane=self.vehicles.get_lane(lag_veh),
                        pos=rel_pos,
                        speed=tc.DEPARTFLAG_SPEED_RANDOM,
                    )

                    self.num_cut_in += 1

            else:
                # check if the cut-in vehicle is leaving in this time step
                if random.uniform(0, 1) < self.cut_out_prob:
                    veh_id = self.cut_in[i]
                    self.traci_connection.vehicle.remove(veh_id)
                    try:
                        self.traci_connection.vehicle.unsubscribe(veh_id)
                        self.vehicles.remove(veh_id)
                    except:
                        pass
                    self.cut_in[i] = None

    def compute_reward(self, state, rl_actions, **kwargs):
        headway = [self.vehicles.get_headway(veh_id)
                   / max(self.vehicles.get_speed(veh_id), 0.01)
                   for veh_id in self.vehicles.get_rl_ids()]

        target = self.env_params.additional_params["target_headway"]
        cost = np.linalg.norm(target - np.array(headway))
        max_cost = np.linalg.norm([target] * self.vehicles.num_rl_vehicles)

        return max(max_cost - cost, 0)

    def get_state(self, **kwargs):
        return np.array(
            [self.vehicles.get_headway(veh_id)
             for veh_id in self.vehicles.get_rl_ids()] +
            [self.vehicles.get_speed(veh_id)
             for veh_id in self.vehicles.get_rl_ids() + ["leader_0"]]
        )

    def reset(self):
        # reset the number of cut-in vehicles
        self.num_cut_in = 0

        # empty the acceleration queue
        self.accel_queue.clear()

        # fill the acceleration delay with actions for the first set of time
        # steps corresponding to the delay
        for _ in range(self.delay_size):
            self.accel_queue.append([0] * self.action_space.shape[0])

        return super().reset()
