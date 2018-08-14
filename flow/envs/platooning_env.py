from flow.envs.base_env import Env
from gym.spaces.box import Box
import numpy as np
import collections

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
    "accel_noise": 0,
    # standard deviation of the gaussian exogenous noise added to the speed
    # observations, in m/s
    "speed_noise": 0,
    # standard deviation of the gaussian exogenous noise added to the position
    # observations, in meters
    "position_noise": 0,
    # desired time headway for the autonomous vehicles, in seconds
    "target_headway": 3,
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

        # initialize an empty acceleration queue
        self.accel_queue = collections.deque()

        # number of actions performed before the most recent set of actions
        self.delay_size = int(self.env_params.additional_params["accel_delay"]
                              / self.sim_step)

        super().__init__(env_params, sumo_params, scenario)

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
        # add noise to the position and velocity observation of all vehicles in
        # accordance with the noise parameters
        if self.env_params.additional_params["speed_noise"] > 0:
            pass
        if self.env_params.additional_params["position_noise"] > 0:
            pass

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
        # empty the acceleration queue
        self.accel_queue.clear()

        # fill the acceleration delay with actions for the first set of time
        # steps corresponding to the delay
        for _ in range(self.delay_size):
            self.accel_queue.append([0] * self.action_space.shape[0])

        return super().reset()
