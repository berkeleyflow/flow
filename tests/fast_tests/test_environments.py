import unittest
import os
import numpy as np

from flow.core.vehicles import Vehicles
from flow.core.params import NetParams, EnvParams, SumoParams, InitialConfig
from flow.controllers import RLController, IDMController
from flow.core.rewards import desired_velocity

from flow.scenarios import LoopScenario, CircleGenerator
from flow.scenarios.loop.loop_scenario import ADDITIONAL_NET_PARAMS \
    as LOOP_PARAMS

from flow.envs import TestEnv, AccelEnv, LaneChangeAccelEnv, \
    LaneChangeAccelPOEnv, WaveAttenuationEnv, WaveAttenuationPOEnv
from flow.envs.loop.loop_accel import ADDITIONAL_ENV_PARAMS as ACCELENV_PARAMS
from flow.envs.loop.lane_changing import ADDITIONAL_ENV_PARAMS as LCENV_PARAMS
from flow.envs.loop.wave_attenuation import ADDITIONAL_ENV_PARAMS as WAV_PARAMS

os.environ["TEST_FLAG"] = "True"


class TestTestEnv(unittest.TestCase):

    """Tests the TestEnv environment in flow/envs/test.py"""

    def setUp(self):
        vehicles = Vehicles()
        vehicles.add("test")
        net_params = NetParams(additional_params=LOOP_PARAMS)
        env_params = EnvParams()
        sumo_params = SumoParams()

        scenario = LoopScenario("test_loop",
                                generator_class=CircleGenerator,
                                vehicles=vehicles,
                                net_params=net_params)

        self.env = TestEnv(env_params, sumo_params, scenario)

    def tearDown(self):
        self.env.terminate()
        self.env = None

    def test_obs_space(self):
        self.assertEqual(self.env.observation_space.shape[0], 0)
        self.assertEqual(len(self.env.observation_space.high), 0)
        self.assertEqual(len(self.env.observation_space.low), 0)

    def test_action_space(self):
        self.assertEqual(self.env.action_space.shape[0], 0)
        self.assertEqual(len(self.env.action_space.high), 0)
        self.assertEqual(len(self.env.action_space.low), 0)

    def test_get_state(self):
        self.assertEqual(len(self.env.get_state()), 0)

    def test_compute_reward(self):
        # test the default
        self.assertEqual(self.env.compute_reward([], []), 0)

        # test if the "reward_fn" parameter is defined
        def reward_fn(*_):
            return 1
        self.env.env_params.additional_params["reward_fn"] = reward_fn
        self.assertEqual(self.env.compute_reward([], []), 1)


class TestAccelEnv(unittest.TestCase):

    """Tests the AccelEnv environment in flow/envs/loop/loop_accel.py"""

    def setUp(self):
        vehicles = Vehicles()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        net_params = NetParams(additional_params=LOOP_PARAMS)
        env_params = EnvParams(additional_params=ACCELENV_PARAMS)
        sumo_params = SumoParams()

        scenario = LoopScenario("test_loop",
                                generator_class=CircleGenerator,
                                vehicles=vehicles,
                                net_params=net_params)

        self.env = AccelEnv(env_params, sumo_params, scenario)

    def tearDown(self):
        self.env.terminate()
        self.env = None

    def test_observed_ids(self):
        self.env.additional_command()
        self.assertListEqual(self.env.vehicles.get_observed_ids(),
                             self.env.vehicles.get_human_ids())

    def test_action_space(self):
        self.assertEqual(self.env.action_space.shape[0],
                         self.env.vehicles.num_rl_vehicles)
        self.assertEqual(self.env.action_space.high,
                         self.env.env_params.additional_params["max_accel"])
        self.assertEqual(self.env.action_space.low,
                         -self.env.env_params.additional_params["max_decel"])

    def test_get_state(self):
        expected_state = np.array([[self.env.vehicles.get_speed(veh_id)
                                    / self.env.scenario.max_speed,
                                    self.env.get_x_by_id(veh_id) /
                                    self.env.scenario.length]
                                   for veh_id in self.env.sorted_ids])

        self.assertTrue((self.env.get_state() == expected_state).all())

    def test_compute_reward(self):
        rew = self.env.compute_reward([], [], fail=False)
        self.assertEqual(rew, desired_velocity(self.env))

    def test_apply_rl_actions(self):
        self.env.step(rl_actions=[1])
        self.assertAlmostEqual(self.env.vehicles.get_speed("rl_0"), 0.1, 2)


class TestLaneChangeAccelEnv(unittest.TestCase):

    """Tests the LaneChangeAccelEnv env in flow/envs/loop/lane_changing.py"""

    def setUp(self):
        vehicles = Vehicles()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        loop_params = LOOP_PARAMS.copy()
        loop_params["lanes"] = 2
        net_params = NetParams(additional_params=loop_params)
        env_params = EnvParams(additional_params=LCENV_PARAMS)
        sumo_params = SumoParams()
        initial_config = InitialConfig(lanes_distribution=1)

        scenario = LoopScenario("test_loop",
                                generator_class=CircleGenerator,
                                vehicles=vehicles,
                                net_params=net_params,
                                initial_config=initial_config)

        self.env = LaneChangeAccelEnv(env_params, sumo_params, scenario)

    def tearDown(self):
        self.env.terminate()
        self.env = None

    def test_observed_ids(self):
        self.env.additional_command()
        self.assertListEqual(self.env.vehicles.get_observed_ids(),
                             self.env.vehicles.get_human_ids())

    def test_action_space(self):
        self.assertEqual(self.env.action_space.shape[0],
                         2 * self.env.vehicles.num_rl_vehicles)
        self.assertTrue(
            (self.env.action_space.high ==
             np.array([self.env.env_params.additional_params["max_accel"], 1]))
            .all())
        self.assertTrue(
            (self.env.action_space.low ==
             np.array([-self.env.env_params.additional_params["max_decel"],
                       -1])).all())

    def test_get_state(self):
        # normalizers
        max_speed = self.env.scenario.max_speed
        length = self.env.scenario.length
        max_lanes = max(self.env.scenario.num_lanes(edge)
                        for edge in self.env.scenario.get_edge_list())

        expected = np.array([[self.env.vehicles.get_speed(veh_id) / max_speed,
                              self.env.get_x_by_id(veh_id) / length,
                              self.env.vehicles.get_lane(veh_id) / max_lanes]
                             for veh_id in self.env.sorted_ids])

        self.assertTrue((self.env.get_state() == expected).all())

    def test_compute_reward(self):
        rew = self.env.compute_reward([], [], fail=False)
        self.assertEqual(rew, desired_velocity(self.env))

    def test_apply_rl_actions(self):
        self.env.step(rl_actions=[1, 1])
        self.assertAlmostEqual(self.env.vehicles.get_speed("rl_0"), 0.1, 2)
        self.assertEqual(self.env.vehicles.get_lane("rl_0"), 1)


class TestLaneChangeAccelPOEnv(unittest.TestCase):

    """Tests the LaneChangeAccelPOEnv env in flow/envs/loop/lane_changing.py.
    Note that some tests are skipped here because they covered by its parent
    class: LaneChangeAccelEnv"""

    def setUp(self):
        vehicles = Vehicles()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        loop_params = LOOP_PARAMS.copy()
        loop_params["lanes"] = 2
        net_params = NetParams(additional_params=loop_params)
        env_params = EnvParams(additional_params=LCENV_PARAMS)
        sumo_params = SumoParams()
        initial_config = InitialConfig(lanes_distribution=1)

        scenario = LoopScenario("test_loop",
                                generator_class=CircleGenerator,
                                vehicles=vehicles,
                                net_params=net_params,
                                initial_config=initial_config)

        self.env = LaneChangeAccelPOEnv(env_params, sumo_params, scenario)

    def tearDown(self):
        self.env.terminate()
        self.env = None

    def test_observed_ids(self):
        self.env.step([])
        self.env.additional_command()
        self.assertListEqual(self.env.vehicles.get_observed_ids(),
                             self.env.vehicles.get_leader(
                                 self.env.vehicles.get_rl_ids()))

    def test_obs_space(self):
        self.assertEqual(self.env.observation_space.shape[0],
                         4 * self.env.vehicles.num_rl_vehicles *
                         self.env.num_lanes +
                         self.env.vehicles.num_rl_vehicles)
        self.assertTrue((np.array(self.env.observation_space.high) == 1).all())
        self.assertTrue((np.array(self.env.observation_space.low) == 0).all())


class TestWaveAttenuationEnv(unittest.TestCase):

    """Tests WaveAttenuationEnv in flow/envs/loop/wave_attenuation.py. Note
    that, besides the reward function and the reset method, it acts in a very
    similar manner as AccelEnv."""

    def setUp(self):
        vehicles = Vehicles()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        loop_params = LOOP_PARAMS.copy()
        # picking a number outside the ring range to test the reset in a later
        # portion of the class
        loop_params["length"] = 2000
        net_params = NetParams(additional_params=LOOP_PARAMS)

        env_params = EnvParams(additional_params=WAV_PARAMS)
        sumo_params = SumoParams()

        scenario = LoopScenario("test_loop",
                                generator_class=CircleGenerator,
                                vehicles=vehicles,
                                net_params=net_params)

        self.env = WaveAttenuationEnv(env_params, sumo_params, scenario)

    def tearDown(self):
        self.env.terminate()
        self.env = None

    def test_observed_ids(self):
        self.env.additional_command()
        self.assertListEqual(self.env.vehicles.get_observed_ids(),
                             self.env.vehicles.get_human_ids())

    def test_action_space(self):
        self.assertEqual(self.env.action_space.shape[0],
                         self.env.vehicles.num_rl_vehicles)
        self.assertEqual(self.env.action_space.high,
                         self.env.env_params.additional_params["max_accel"])
        self.assertEqual(self.env.action_space.low,
                         -self.env.env_params.additional_params["max_decel"])

    def test_get_state(self):
        expected_state = np.array([[self.env.vehicles.get_speed(veh_id)
                                    / self.env.scenario.max_speed,
                                    self.env.get_x_by_id(veh_id) /
                                    self.env.scenario.length]
                                   for veh_id in self.env.sorted_ids])

        self.assertTrue((self.env.get_state() == expected_state).all())

    def test_compute_reward(self):
        # explicitly copied over the reward here to make sure we never lose it
        # (this is only reward that has manage to solve for the partially
        # observable ring with varying lengths, at least when using policy
        # gradient)
        vel = np.array([self.env.vehicles.get_speed(veh_id)
                        for veh_id in self.env.vehicles.get_ids()])
        eta_2 = 4.
        reward = eta_2 * np.mean(vel) / 20
        eta = 8  # 0.25
        rl_actions = np.array([1])
        accel_threshold = 0

        if np.mean(np.abs(rl_actions)) > accel_threshold:
            reward += eta * (accel_threshold - np.mean(np.abs(rl_actions)))
        expected_rew = float(reward)

        rew = self.env.compute_reward([], rl_actions=rl_actions, fail=False)
        self.assertEqual(rew, expected_rew)

    def test_apply_rl_actions(self):
        self.env.step(rl_actions=[1])
        self.assertAlmostEqual(self.env.vehicles.get_speed("rl_0"), 0.1, 2)

    def test_reset(self):
        """Tests that the length of the ring road scenario during a reset is
        set between the ring_length range. For this reason, we start with a
        very large ring in this problem."""
        self.env.reset()
        self.assertGreaterEqual(self.env.scenario.length,
                                self.env.env_params.additional_params[
                                    "ring_length"][0])
        self.assertLessEqual(self.env.scenario.length,
                             self.env.env_params.additional_params[
                                 "ring_length"][1])


class TestWaveAttenuationPOEnv(unittest.TestCase):

    """Tests WaveAttenuationPOEnv in flow/envs/loop/wave_attenuation.py. Note
    that some tests are skipped here because they covered by its parent class:
    TestWaveAttenuationEnv."""

    def setUp(self):
        vehicles = Vehicles()
        vehicles.add("rl", acceleration_controller=(RLController, {}))
        vehicles.add("human", acceleration_controller=(IDMController, {}))

        loop_params = LOOP_PARAMS.copy()
        # picking a number outside the ring range to test the reset in a later
        # portion of the class
        loop_params["length"] = 2000
        net_params = NetParams(additional_params=LOOP_PARAMS)

        env_params = EnvParams(additional_params=WAV_PARAMS)
        sumo_params = SumoParams()

        scenario = LoopScenario("test_loop",
                                generator_class=CircleGenerator,
                                vehicles=vehicles,
                                net_params=net_params)

        self.env = WaveAttenuationPOEnv(env_params, sumo_params, scenario)

    def tearDown(self):
        self.env.terminate()
        self.env = None

    def test_observed_ids(self):
        self.env.additional_command()
        self.assertListEqual(self.env.vehicles.get_observed_ids(),
                             self.env.vehicles.get_leader(
                                 self.env.vehicles.get_rl_ids()))

    def test_get_state(self):
        rl_id = self.env.vehicles.get_rl_ids()[0]
        lead_id = self.env.vehicles.get_leader(rl_id) or rl_id
        max_speed = 15.
        max_length = self.env.env_params.additional_params["ring_length"][1]

        expected_state = np.array([
            self.env.vehicles.get_speed(rl_id) / max_speed,
            (self.env.vehicles.get_speed(lead_id) - self.env.vehicles.
             get_speed(rl_id)) / max_speed,
            self.env.vehicles.get_headway(rl_id) / max_length
        ])

        self.assertTrue((self.env.get_state() == expected_state).all())


if __name__ == '__main__':
    unittest.main()
