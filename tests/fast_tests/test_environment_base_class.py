import unittest

from flow.core.params import SumoParams, EnvParams, InitialConfig, \
    NetParams, SumoCarFollowingParams
from flow.core.vehicles import Vehicles

from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import IDMController
from flow.envs.loop.loop_accel import ADDITIONAL_ENV_PARAMS

from tests.setup_scripts import ring_road_exp_setup
import os
import numpy as np

os.environ["TEST_FLAG"] = "True"


class TestStartingPositionShuffle(unittest.TestCase):
    """
    Tests that, at resets, the starting position of vehicles changes while
    keeping the ordering and relative spacing between vehicles.
    """

    def setUp(self):
        # turn on starting position shuffle
        env_params = EnvParams(starting_position_shuffle=True,
                               additional_params=ADDITIONAL_ENV_PARAMS)

        # place 5 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add(veh_id="test",
                     acceleration_controller=(IDMController, {}),
                     routing_controller=(ContinuousRouter, {}),
                     num_vehicles=5)

        initial_config = InitialConfig(x0=5)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(env_params=env_params,
                                                 initial_config=initial_config,
                                                 vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_starting_pos(self):
        ids = self.env.vehicles.get_ids()

        # position of vehicles before reset
        before_reset = \
            np.array([self.env.get_x_by_id(veh_id) for veh_id in ids])

        # reset the environment
        self.env.reset()

        # position of vehicles after reset
        after_reset = \
            np.array([self.env.get_x_by_id(veh_id) for veh_id in ids])

        offset = after_reset[0] - before_reset[0]

        # remove the new offset from the original positions after reset
        after_reset = np.mod(after_reset - offset, self.env.scenario.length)

        np.testing.assert_array_almost_equal(before_reset, after_reset)


class TestVehicleArrangementShuffle(unittest.TestCase):
    """
    Tests that, at resets, the ordering of vehicles changes while the starting
    position values stay the same.
    """

    def setUp(self):
        # turn on vehicle arrangement shuffle
        env_params = EnvParams(vehicle_arrangement_shuffle=True,
                               additional_params=ADDITIONAL_ENV_PARAMS)

        # place 5 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add(veh_id="test",
                     acceleration_controller=(IDMController, {}),
                     routing_controller=(ContinuousRouter, {}),
                     num_vehicles=5)

        initial_config = InitialConfig(x0=5)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(env_params=env_params,
                                                 initial_config=initial_config,
                                                 vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_shuffle(self):
        ids = self.env.vehicles.get_ids()

        # position of vehicles before reset
        before_reset = [self.env.get_x_by_id(veh_id) for veh_id in ids]

        # reset the environment
        self.env.reset()

        # position of vehicles after reset
        after_reset = [self.env.get_x_by_id(veh_id) for veh_id in ids]

        self.assertCountEqual(before_reset, after_reset)


class TestEmissionPath(unittest.TestCase):
    """
    Tests that the default emission path of an environment is set to None.
    If it is not None, then sumo starts accumulating memory.
    """

    def setUp(self):
        # set sumo_params to default
        sumo_params = SumoParams()

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(sumo_params=sumo_params)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_emission(self):
        self.assertIsNone(self.env.sumo_params.emission_path)


class TestApplyingActionsWithSumo(unittest.TestCase):
    """
    Tests the apply_acceleration, apply_lane_change, and choose_routes
    functions in base_env.py
    """
    def setUp(self):
        # create a 2-lane ring road network
        additional_net_params = {"length": 230, "lanes": 3, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # turn on starting position shuffle
        env_params = EnvParams(starting_position_shuffle=True,
                               additional_params=ADDITIONAL_ENV_PARAMS)

        # place 5 vehicles in the network (we need at least more than 1)
        vehicles = Vehicles()
        vehicles.add(veh_id="test",
                     acceleration_controller=(IDMController, {}),
                     routing_controller=(ContinuousRouter, {}),
                     sumo_car_following_params=SumoCarFollowingParams(
                         accel=1000, decel=1000),
                     num_vehicles=5)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(net_params=net_params,
                                                 env_params=env_params,
                                                 vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_apply_acceleration(self):
        """
        Tests that, in the absence of all failsafes, the acceleration requested
        from sumo is equal to the acceleration witnessed in between steps. Also
        ensures that vehicles can never have velocities below zero given any
        acceleration.
        """
        ids = self.env.vehicles.get_ids()

        vel0 = np.array([self.env.vehicles.get_speed(veh_id)
                         for veh_id in ids])

        # apply a certain set of accelerations to the vehicles in the network
        accel_step0 = np.array([0, 1, 4, 9, 16])
        self.env.apply_acceleration(veh_ids=ids, acc=accel_step0)
        self.env.traci_connection.simulationStep()

        # compare the new velocity of the vehicles to the expected velocity
        # given the accelerations
        vel1 = np.array([self.env.traci_connection.vehicle.getSpeed(veh_id)
                         for veh_id in ids])
        expected_vel1 = (vel0 + accel_step0 * 0.1).clip(min=0)

        np.testing.assert_array_almost_equal(vel1, expected_vel1, 1)

        # collect information on the vehicle in the network from sumo
        veh_obs = self.env.traci_connection.vehicle.getSubscriptionResults()

        # get vehicle ids for the entering, exiting, and colliding vehicles
        id_list = self.env.traci_connection.simulation.getSubscriptionResults()

        # store the network observations in the vehicles class
        self.env.vehicles.update(veh_obs, id_list, self.env)

        # apply a set of decelerations
        accel_step1 = np.array([-16, -9, -4, -1, 0])
        self.env.apply_acceleration(veh_ids=ids, acc=accel_step1)
        self.env.traci_connection.simulationStep()

        # this time, some vehicles should be at 0 velocity (NOT less), and sum
        # are a result of the accelerations that took place
        vel2 = np.array([self.env.traci_connection.vehicle.getSpeed(veh_id)
                         for veh_id in ids])
        expected_vel2 = (vel1 + accel_step1 * 0.1).clip(min=0)

        np.testing.assert_array_almost_equal(vel2, expected_vel2, 1)

    def test_apply_lane_change_errors(self):
        """
        Ensures that apply_lane_change raises ValueErrors when it should
        """
        self.env.reset()
        ids = self.env.vehicles.get_ids()

        # make sure that running apply lane change with a invalid direction
        # values leads to a ValueError
        bad_directions = np.array([-1, 0, 1, 2, 3])

        self.assertRaises(
            ValueError,
            self.env.apply_lane_change, veh_ids=ids, direction=bad_directions)

    def test_apply_lane_change_direction(self):
        """
        Tests the direction method for apply_lane_change. Ensures that the lane
        change action requested from sumo is the same as the lane change that
        occurs, and that vehicles attempting do not issue lane changes in there
        is no lane in te requested direction.
        """
        self.env.reset()
        ids = self.env.vehicles.get_ids()
        lane0 = np.array([self.env.vehicles.get_lane(veh_id)
                          for veh_id in ids])

        # perform lane-changing actions using the direction method
        direction0 = np.array([0, 1, 0, 1, -1])
        self.env.apply_lane_change(ids, direction=direction0)
        self.env.traci_connection.simulationStep()

        # check that the lane vehicle lane changes to the correct direction
        # without skipping lanes
        lane1 = np.array(
            [self.env.traci_connection.vehicle.getLaneIndex(veh_id)
             for veh_id in ids])
        expected_lane1 = (lane0 + np.sign(direction0)).clip(
            min=0, max=self.env.scenario.lanes - 1)

        np.testing.assert_array_almost_equal(lane1, expected_lane1, 1)

        # collect information on the vehicle in the network from sumo
        veh_obs = self.env.traci_connection.vehicle.getSubscriptionResults()

        # get vehicle ids for the entering, exiting, and colliding vehicles
        id_list = self.env.traci_connection.simulation.getSubscriptionResults()

        # store the network observations in the vehicles class
        self.env.vehicles.update(veh_obs, id_list, self.env)

        # perform lane-changing actions using the direction method one more
        # time to test lane changes to the right
        direction1 = np.array([-1, -1, -1, -1, -1])
        self.env.apply_lane_change(ids, direction=direction1)
        self.env.traci_connection.simulationStep()

        # check that the lane vehicle lane changes to the correct direction
        # without skipping lanes
        lane2 = np.array(
            [self.env.traci_connection.vehicle.getLaneIndex(veh_id)
             for veh_id in ids])
        expected_lane2 = (lane1 + np.sign(direction1)).clip(
            min=0, max=self.env.scenario.lanes - 1)

        np.testing.assert_array_almost_equal(lane2, expected_lane2, 1)


class TestSorting(unittest.TestCase):
    """
    Tests that the sorting method returns a list of ids sorted by the
    get_absolute_position() method when sorting is requested, and does nothing
    if it is not requested
    """

    def test_sorting(self):
        # setup a environment with the "sort_vehicles" attribute set to True
        additional_env_params = ADDITIONAL_ENV_PARAMS
        env_params = EnvParams(additional_params=additional_env_params,
                               sort_vehicles=True)
        initial_config = InitialConfig(shuffle=True)
        vehicles = Vehicles()
        vehicles.add(veh_id="test", num_vehicles=5)
        self.env, scenario = ring_road_exp_setup(env_params=env_params,
                                                 initial_config=initial_config,
                                                 vehicles=vehicles)

        self.env.reset()

        sorted_ids = self.env.sorted_ids
        positions = self.env.vehicles.get_absolute_position(sorted_ids)

        # ensure vehicles ids are in sorted order by positions
        self.assertTrue(all(positions[i] <= positions[i + 1]
                            for i in range(len(positions) - 1)))

    def test_no_sorting(self):
        # setup a environment with the "sort_vehicles" attribute set to False,
        # and shuffling so that the vehicles are not sorted by their ids
        additional_env_params = ADDITIONAL_ENV_PARAMS
        env_params = EnvParams(additional_params=additional_env_params,
                               sort_vehicles=True)
        initial_config = InitialConfig(shuffle=True)
        vehicles = Vehicles()
        vehicles.add(veh_id="test", num_vehicles=5)
        self.env, scenario = ring_road_exp_setup(env_params=env_params,
                                                 initial_config=initial_config,
                                                 vehicles=vehicles)

        self.env.reset()

        sorted_ids = list(self.env.sorted_ids)
        ids = self.env.vehicles.get_ids()

        # ensure that the list of ids did not change
        self.assertListEqual(sorted_ids, ids)


class TestWarmUpSteps(unittest.TestCase):

    """Ensures that the appropriate number of warmup steps are run when using
    flow.core.params.EnvParams.warmup_steps"""

    def test_it_works(self):
        warmup_step = 5  # some value

        # start an environment with a number of simulations per step greater
        # than one
        env_params = EnvParams(warmup_steps=warmup_step,
                               additional_params=ADDITIONAL_ENV_PARAMS)
        env, scenario = ring_road_exp_setup(env_params=env_params)

        # time before running a reset
        t1 = env.time_counter
        # perform a reset
        env.reset()
        # time after a reset
        t2 = env.time_counter

        # ensure that the difference in time is equal to sims_per_step
        self.assertEqual(t2 - t1, warmup_step)


class TestSimsPerStep(unittest.TestCase):

    """Ensures that the appropriate number of simultaions are run at any given
    steps when using flow.core.params.EnvParams.sims_per_step"""

    def test_it_works(self):
        sims_per_step = 5  # some value

        # start an environment with a number of simulations per step greater
        # than one
        env_params = EnvParams(sims_per_step=sims_per_step,
                               additional_params=ADDITIONAL_ENV_PARAMS)
        env, scenario = ring_road_exp_setup(env_params=env_params)

        env.reset()
        # time before running a step
        t1 = env.time_counter
        # perform a step
        env.step(rl_actions=[])
        # time after a step
        t2 = env.time_counter

        # ensure that the difference in time is equal to sims_per_step
        self.assertEqual(t2 - t1, sims_per_step)


if __name__ == '__main__':
    unittest.main()
