from flow.envs.base_env import Env
from flow.core import rewards

from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple
import numpy as np

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # lane change duration for autonomous vehicles, in s. Autonomous vehicles
    # reject new lane changing commands for this duration after successfully
    # changing lanes.
    "lane_change_duration": 5,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 10,
}


class LaneChangeAccelEnv(Env):
    """Environment used to train autonomous vehicles to improve traffic flows
    when lane-change and acceleration actions are permitted by the rl agent.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * lane_change_duration: lane change duration for autonomous vehicles, in s
    * target_velocity: desired velocity for all vehicles in the network, in m/s

    States
        The state consists of the velocities, absolute position, and lane index
        of all vehicles in the network. This assumes a constant number of
        vehicles.

    Actions
        Actions consist of:

        * a (continuous) acceleration from -abs(max_decel) to max_accel,
          specified in env_params
        * a (continuous) lane-change action from -1 to 1, used to determine the
          lateral direction the vehicle will take.

        Lane change actions are performed only if the vehicle has not changed
        lanes for the lane change duration specified in env_params.

    Rewards
        The reward function is the two-norm of the distance of the speed of the
        vehicles in the network from a desired speed, combined with a penalty
        to discourage excess lane changes by the rl vehicle.

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

    @property
    def action_space(self):
        max_decel = self.env_params.additional_params["max_decel"]
        max_accel = self.env_params.additional_params["max_accel"]

        lb = [-abs(max_decel), -1] * self.vehicles.num_rl_vehicles
        ub = [max_accel, 1] * self.vehicles.num_rl_vehicles

        return Box(np.array(lb), np.array(ub), dtype=np.float32)

    @property
    def observation_space(self):
        speed = Box(low=0, high=1, shape=(self.vehicles.num_vehicles,),
                    dtype=np.float32)
        lane = Box(low=0, high=1, shape=(self.vehicles.num_vehicles,),
                   dtype=np.float32)
        pos = Box(low=0., high=1, shape=(self.vehicles.num_vehicles,),
                  dtype=np.float32)
        return Tuple((speed, pos, lane))

    def compute_reward(self, state, rl_actions, **kwargs):
        # compute the system-level performance of vehicles from a velocity
        # perspective
        reward = rewards.desired_velocity(self, fail=kwargs["fail"])

        # punish excessive lane changes by reducing the reward by a set value
        # every time an rl car changes lanes (10% of max reward)
        for veh_id in self.vehicles.get_rl_ids():
            if self.vehicles.get_state(veh_id, "last_lc") == self.time_counter:
                reward -= 0.1

        return reward

    def get_state(self):
        # normalizers
        max_speed = self.scenario.max_speed
        length = self.scenario.length
        max_lanes = max(self.scenario.num_lanes(edge)
                        for edge in self.scenario.get_edge_list())

        return np.array([[self.vehicles.get_speed(veh_id) / max_speed,
                          self.get_x_by_id(veh_id) / length,
                          self.vehicles.get_lane(veh_id) / max_lanes]
                         for veh_id in self.sorted_ids])

    def _apply_rl_actions(self, actions):
        acceleration = actions[::2]
        direction = actions[1::2]

        # re-arrange actions according to mapping in observation space
        sorted_rl_ids = [veh_id for veh_id in self.sorted_ids
                         if veh_id in self.vehicles.get_rl_ids()]

        # represents vehicles that are allowed to change lanes
        non_lane_changing_veh = \
            [self.time_counter <=
             self.env_params.additional_params["lane_change_duration"]
             + self.vehicles.get_state(veh_id, 'last_lc')
             for veh_id in sorted_rl_ids]
        # vehicle that are not allowed to change have their directions set to 0
        direction[non_lane_changing_veh] = \
            np.array([0] * sum(non_lane_changing_veh))

        print(direction)

        self.apply_acceleration(sorted_rl_ids, acc=acceleration)
        self.apply_lane_change(sorted_rl_ids, direction=direction)

    def additional_command(self):
        # specify observed vehicles
        if self.vehicles.num_rl_vehicles > 0:
            for veh_id in self.vehicles.get_human_ids():
                self.vehicles.set_observed(veh_id)


class LaneChangeAccelPOEnv(LaneChangeAccelEnv):
    """POMDP version of LaneChangeAccelEnv.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * lane_change_duration: lane change duration for autonomous vehicles, in s
    * target_velocity: desired velocity for all vehicles in the network, in m/s

    States
        States are a list of rl vehicles speeds, as well as the speeds and
        bumper-to-bumper headawys between the rl vehicles and their
        leaders/followers in all lanes. There is no assumption on the number of
        vehicles in the network, so long as the number of rl vehicles is
        static.

    Actions
        See parent class.

    Rewards
        See parent class.

    Termination
        See parent class.
    """

    def __init__(self, env_params, sumo_params, scenario):
        # maximum number of lanes on any edge in the network
        self.num_lanes = max(scenario.num_lanes(edge)
                             for edge in scenario.get_edge_list())

        # lists of visible vehicles, used for visualization purposes
        self.visible = []

        super().__init__(env_params, sumo_params, scenario)

    @property
    def observation_space(self):
        return Box(low=0, high=1,
                   shape=(4 * self.vehicles.num_rl_vehicles * self.num_lanes
                          + self.vehicles.num_rl_vehicles,),
                   dtype=np.float32)

    def get_state(self):
        obs = [0 for _ in range(4 * self.vehicles.num_rl_vehicles
                                * self.num_lanes)]

        self.visible = []
        for i, rl_id in enumerate(self.vehicles.get_rl_ids()):
            # normalizers
            max_length = self.scenario.length
            max_speed = self.scenario.max_speed

            # set to 1000 since the absence of a vehicle implies a large
            # headway
            headway = [1] * self.num_lanes
            tailway = [1] * self.num_lanes
            vel_in_front = [0] * self.num_lanes
            vel_behind = [0] * self.num_lanes

            lane_leaders = self.vehicles.get_lane_leaders(rl_id)
            lane_followers = self.vehicles.get_lane_followers(rl_id)
            lane_headways = self.vehicles.get_lane_headways(rl_id)
            lane_tailways = self.vehicles.get_lane_tailways(rl_id)
            headway[0:len(lane_headways)] = lane_headways
            tailway[0:len(lane_tailways)] = lane_tailways

            for j, lane_leader in enumerate(lane_leaders):
                if lane_leader != '':
                    lane_headways[j] /= max_length
                    vel_in_front[j] = self.vehicles.get_speed(lane_leader) \
                        / max_speed
                    self.visible.extend([lane_leader])
            for j, lane_follower in enumerate(lane_followers):
                if lane_follower != '':
                    lane_headways[j] /= max_length
                    vel_behind[j] = self.vehicles.get_speed(lane_follower) \
                        / max_speed
                    self.visible.extend([lane_follower])

            # add the headways, tailways, and speed for all lane leaders
            # and followers
            obs[4*self.num_lanes*i:4*self.num_lanes*(i+1)] = \
                np.concatenate((headway, tailway, vel_in_front, vel_behind))

            # add the speed for the ego rl vehicle
            obs.append(self.vehicles.get_speed(rl_id))

            return np.array(obs)

    def additional_command(self):
        # specify observed vehicles
        for veh_id in self.visible:
            self.vehicles.set_observed(veh_id)
