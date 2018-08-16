import unittest
import os
import numpy as np

from flow.core.vehicles import Vehicles
from flow.core.params import SumoCarFollowingParams, NetParams, InitialConfig \
    , SumoParams, InFlows
from flow.controllers.car_following_models import IDMController, \
    SumoCarFollowingController
from flow.controllers.lane_change_controllers import StaticLaneChanger
from flow.controllers.rlcontroller import RLController
from flow.core.experiment import SumoExperiment
from flow.controllers.routing_controllers import GridRouter
from tests.setup_scripts import ring_road_exp_setup, grid_mxn_exp_setup


os.environ["TEST_FLAG"] = "True"


class TestVehiclesClass(unittest.TestCase):
    """
    Tests various functions in the vehicles class
    """

    def runSpeedLaneChangeModes(self):
        """
        Checks to make sure vehicle class correctly specifies lane change and
        speed modes
        """
        vehicles = Vehicles()
        vehicles.add("typeA",
                     acceleration_controller=(IDMController, {}),
                     speed_mode='no_collide',
                     lane_change_mode="no_lat_collide")

        self.assertEqual(vehicles.get_speed_mode("typeA_0"), 1)
        self.assertEqual(vehicles.get_lane_change_mode("typeA_0"), 256)

        vehicles.add("typeB",
                     acceleration_controller=(IDMController, {}),
                     speed_mode='aggressive',
                     lane_change_mode="strategic")

        self.assertEqual(vehicles.get_speed_mode("typeB_0"), 0)
        self.assertEqual(vehicles.get_lane_change_mode("typeB_0"), 853)

        vehicles.add("typeC",
                     acceleration_controller=(IDMController, {}),
                     speed_mode=31,
                     lane_change_mode=277)
        self.assertEqual(vehicles.get_speed_mode("typeC_0"), 31)
        self.assertEqual(vehicles.get_lane_change_mode("typeC_0"), 277)

    def test_controlled_id_params(self):
        """
        Ensures that, if a vehicle is not a sumo vehicle, then minGap is set to
        zero so that all headway values are correct.
        """
        # check that, if the vehicle is not a SumoCarFollowingController
        # vehicle, then its minGap is equal to 0
        vehicles = Vehicles()
        vehicles.add("typeA",
                     acceleration_controller=(IDMController, {}),
                     speed_mode='no_collide',
                     lane_change_mode="no_lat_collide")
        self.assertEqual(vehicles.types[0][1]["minGap"], 0)

        # check that, if the vehicle is a SumoCarFollowingController vehicle,
        # then its minGap, accel, and decel are set to default
        vehicles = Vehicles()
        vehicles.add("typeA",
                     acceleration_controller=(SumoCarFollowingController, {}),
                     speed_mode='no_collide',
                     lane_change_mode="no_lat_collide")
        default_mingap = SumoCarFollowingParams().controller_params["minGap"]
        self.assertEqual(vehicles.types[0][1]["minGap"], default_mingap)

    def test_add_vehicles_human(self):
        """
        Ensures that added human vehicles are placed in the current vehicle
        IDs, and that the number of vehicles is correct.
        """
        # generate a vehicles class
        vehicles = Vehicles()

        # vehicles whose acceleration and LC are controlled by sumo
        vehicles.add("test_1", num_vehicles=1)

        # vehicles whose acceleration are controlled by sumo
        vehicles.add("test_2", num_vehicles=2,
                     lane_change_controller=(StaticLaneChanger, {}))

        # vehicles whose LC are controlled by sumo
        vehicles.add("test_3", num_vehicles=4,
                     acceleration_controller=(IDMController, {}))

        self.assertEqual(vehicles.num_vehicles, 7)
        self.assertEqual(len(vehicles.get_ids()), 7)
        self.assertEqual(len(vehicles.get_rl_ids()), 0)
        self.assertEqual(len(vehicles.get_human_ids()), 7)
        self.assertEqual(len(vehicles.get_controlled_ids()), 4)
        self.assertEqual(len(vehicles.get_controlled_lc_ids()), 2)

    def test_add_vehicles_rl(self):
        """
        Ensures that added rl vehicles are placed in the current vehicle IDs,
        and that the number of vehicles is correct.
        """
        vehicles = Vehicles()
        vehicles.add("test_rl", num_vehicles=10,
                     acceleration_controller=(RLController, {}))

        self.assertEqual(vehicles.num_vehicles, 10)
        self.assertEqual(len(vehicles.get_ids()), 10)
        self.assertEqual(len(vehicles.get_rl_ids()), 10)
        self.assertEqual(len(vehicles.get_human_ids()), 0)
        self.assertEqual(len(vehicles.get_controlled_ids()), 0)
        self.assertEqual(len(vehicles.get_controlled_lc_ids()), 0)

    def test_remove(self):
        """
        Checks that there is no trace of the vehicle ID of the vehicle meant to
        be removed in the vehicles class.
        """
        # generate a vehicles class
        vehicles = Vehicles()
        vehicles.add("test", num_vehicles=10)
        vehicles.add("test_rl", num_vehicles=10,
                     acceleration_controller=(RLController, {}))

        # remove one human-driven vehicle and on rl vehicle
        vehicles.remove("test_0")
        vehicles.remove("test_rl_0")

        # ensure that the removed vehicle's ID is not in any lists of vehicles
        if "test_0" in vehicles.get_ids():
            raise AssertionError("vehicle still in get_ids()")
        if "test_0" in vehicles.get_human_ids():
            raise AssertionError("vehicle still in get_controlled_lc_ids()")
        if "test_0" in vehicles.get_controlled_lc_ids():
            raise AssertionError("vehicle still in get_controlled_lc_ids()")
        if "test_0" in vehicles.get_controlled_ids():
            raise AssertionError("vehicle still in get_controlled_ids()")
        if "test_rl_0" in vehicles.get_ids():
            raise AssertionError("RL vehicle still in get_ids()")
        if "test_rl_0" in vehicles.get_rl_ids():
            raise AssertionError("RL vehicle still in get_rl_ids()")

        # ensure that the vehicles are not storing extra information in the
        # vehicles.__vehicles dict
        error_state = vehicles.get_state('test_0', "type", error=None)
        self.assertIsNone(error_state)
        error_state_rl = vehicles.get_state('rl_test_0', "type", error=None)
        self.assertIsNone(error_state_rl)

        # ensure that the num_vehicles matches the actual number of vehicles
        self.assertEqual(vehicles.num_vehicles, len(vehicles.get_ids()))

        # ensures that then num_rl_vehicles matches the actual number of rl veh
        self.assertEqual(vehicles.num_rl_vehicles, len(vehicles.get_rl_ids()))


class TestMultiLaneData(unittest.TestCase):
    """
    Tests the functions get_lane_leaders(), get_lane_followers(),
    get_lane_headways(), and get_lane_footways() in the Vehicles class.
    """

    def test_no_junctions(self):
        """
        Tests the above mentioned methods in the absence of junctions.
        """
        # setup a network with no junctions and several vehicles
        # also, setup with a deterministic starting position to ensure that the
        # headways/lane leaders are what is expected
        additional_net_params = {"length": 230, "lanes": 3, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        vehicles = Vehicles()
        vehicles.add(veh_id="test",
                     acceleration_controller=(RLController, {}),
                     num_vehicles=21)

        initial_config = InitialConfig(lanes_distribution=float("inf"))

        env, scenario = ring_road_exp_setup(net_params=net_params,
                                            vehicles=vehicles,
                                            initial_config=initial_config)
        env.reset()

        # check the lane leaders method is outputting the right values
        actual_lane_leaders = env.vehicles.get_lane_leaders("test_0")
        expected_lane_leaders = ["test_3", "test_1", "test_2"]
        self.assertCountEqual(actual_lane_leaders, expected_lane_leaders)

        # check the lane headways is outputting the right values
        actual_lane_head = env.vehicles.get_lane_headways("test_0")
        expected_lane_head = [27.85714285714286, -5, -5]
        self.assertCountEqual(actual_lane_head, expected_lane_head)

        # check the lane followers method is outputting the right values
        actual_lane_followers = env.vehicles.get_lane_followers("test_0")
        expected_lane_followers = ["test_18", "test_19", "test_20"]
        self.assertCountEqual(actual_lane_followers, expected_lane_followers)

        # check the lane tailways is outputting the right values
        actual_lane_tail = env.vehicles.get_lane_tailways("test_0")
        expected_lane_tail = [27.85714285714286] * 3
        np.testing.assert_array_almost_equal(actual_lane_tail,
                                             expected_lane_tail)

    def test_junctions(self):
        """
        Tests the above mentioned methods in the presence of junctions.
        """
        # TODO(ak): add test
        pass


class TestIdsByEdge(unittest.TestCase):
    """
    Tests the ids_by_edge() method
    """

    def setUp(self):
        # create the environment and scenario classes for a figure eight
        vehicles = Vehicles()
        vehicles.add(veh_id="test", num_vehicles=20)

        self.env, scenario = ring_road_exp_setup(vehicles=vehicles)

    def tearDown(self):
        # free data used by the class
        self.env.terminate()
        self.env = None

    def test_ids_by_edge(self):
        self.env.reset()
        ids = self.env.vehicles.get_ids_by_edge("bottom")
        expected_ids = ["test_0", "test_1", "test_2", "test_3", "test_4"]
        self.assertListEqual(sorted(ids), sorted(expected_ids))


class TestObservedIDs(unittest.TestCase):
    """Tests the observed_ids methods, which are used for visualization."""

    def test_obs_ids(self):
        vehicles = Vehicles()
        vehicles.add(veh_id="test", num_vehicles=10)

        # test setting new observed values
        vehicles.set_observed("test_0")
        self.assertCountEqual(vehicles.get_observed_ids(), ["test_0"])

        vehicles.set_observed("test_1")
        self.assertCountEqual(vehicles.get_observed_ids(),
                              ["test_0", "test_1"])

        # ensures that setting vehicles twice doesn't add an element
        vehicles.set_observed("test_0")
        self.assertListEqual(vehicles.get_observed_ids(),
                             ["test_0", "test_1"])

        # test removing observed values
        vehicles.remove_observed("test_0")
        self.assertListEqual(vehicles.get_observed_ids(), ["test_1"])

        # ensures that removing a value that does not exist does not lead to
        # an error
        vehicles.remove_observed("test_0")
        self.assertCountEqual(vehicles.get_observed_ids(), ["test_1"])

class TestOutflowInfo(unittest.TestCase):
    """Tests that the out methods return the correct values"""
    def test_inflows(self):
        vehicles = Vehicles()
        self.assertEqual(0, vehicles.get_outflow_rate(10000),
                         "outflow should be zero when there are no vehicles")

        # now construct a scenario where a vehicles is going to leave
        # create the environment and scenario classes for a 1x1 grid

        sumo_params = SumoParams(sim_step=1, sumo_binary="sumo")
        total_vehicles = 4
        vehicles = Vehicles()
        vehicles.add(veh_id="krauss",
                     acceleration_controller=(RLController, {}),
                     routing_controller=(GridRouter, {}),
                     num_vehicles=total_vehicles,
                     speed_mode="all_checks")
        grid_array = {"short_length": 30, "inner_length": 30,
                      "long_length": 30, "row_num": 1,
                      "col_num": 1,
                      "cars_left": 1,
                      "cars_right": 1,
                      "cars_top": 1,
                      "cars_bot": 1}

        additional_net_params = {"speed_limit": 35, "grid_array": grid_array,
                                 "horizontal_lanes": 1, "vertical_lanes": 1}

        net_params = NetParams(no_internal_links=False,
                               additional_params=additional_net_params)

        self.env, self.scenario = grid_mxn_exp_setup(row_num=1, col_num=1,
                                                     sumo_params=sumo_params,
                                                     vehicles=vehicles,
                                                     net_params=net_params)

        # go through the env and set all the lights to green
        for i in range(self.env.rows * self.env.cols):
            self.env.traci_connection.trafficlight.setRedYellowGreenState(
                'center' + str(i), "gggrrrgggrrr")


        # instantiate an experiment class
        self.exp = SumoExperiment(self.env, self.scenario)

        self.exp.run(1, 15)
        self.assertAlmostEqual(2*3600.0/15.0, vehicles.get_outflow_rate(100))

if __name__=='__main__':
    unittest.main()
