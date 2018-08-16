import unittest
import os

from tests.setup_scripts import ring_road_exp_setup, grid_mxn_exp_setup
from flow.core.vehicles import Vehicles
from flow.core.params import NetParams
from flow.core.params import SumoCarFollowingParams
from flow.core.traffic_lights import TrafficLights
from flow.core.experiment import SumoExperiment
from flow.controllers.routing_controllers import GridRouter
from flow.controllers.car_following_models import IDMController

os.environ["TEST_FLAG"] = "True"


class TestUpdateGetState(unittest.TestCase):
    """
    Tests the update and get_state functions are working properly.
    """

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_single_lane(self):
        # add a traffic light to the top node
        traffic_lights = TrafficLights()
        traffic_lights.add("top")

        # create a ring road with one lane
        additional_net_params = {"length": 230, "lanes": 1, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(net_params=net_params,
                                                 traffic_lights=traffic_lights)

        self.env.reset()
        self.env.step([])

        state = self.env.traffic_lights.get_state("top")

        self.assertEqual(state, "G")

    def test_multi_lane(self):
        # add a traffic light to the top node
        traffic_lights = TrafficLights()
        traffic_lights.add("top")

        # create a ring road with two lanes
        additional_net_params = {"length": 230, "lanes": 2, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(net_params=net_params,
                                                 traffic_lights=traffic_lights)

        self.env.reset()
        self.env.step([])

        state = self.env.traffic_lights.get_state("top")

        self.assertEqual(state, "GG")


class TestSetState(unittest.TestCase):
    """
    Tests the set_state function
    """

    def setUp(self):
        # add a traffic light to the top node
        traffic_lights = TrafficLights()
        traffic_lights.add("top")

        # create a ring road with two lanes
        additional_net_params = {"length": 230, "lanes": 2, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(net_params=net_params,
                                                 traffic_lights=traffic_lights)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_all_lanes(self):
        # reset the environment
        self.env.reset()

        # set all states to something
        self.env.traffic_lights.set_state(node_id="top", env=self.env,
                                          state="rY")

        # run a new step
        self.env.step([])

        # check the new values
        state = self.env.traffic_lights.get_state("top")

        self.assertEqual(state, "rY")

    def test_single_lane(self):
        # reset the environment
        self.env.reset()

        # set all state of lane 1 to something
        self.env.traffic_lights.set_state(node_id="top", env=self.env,
                                          state="R", link_index=1)

        # run a new step
        self.env.step([])

        # check the new values
        state = self.env.traffic_lights.get_state("top")

        self.assertEqual(state[1], "R")


class TestPOEnv(unittest.TestCase):
    """
    Tests the set_state function
    """

    def setUp(self):
        vehicles = Vehicles()
        vehicles.add(veh_id="idm",
                     acceleration_controller=(IDMController, {}),
                     routing_controller=(GridRouter, {}),
                     sumo_car_following_params=SumoCarFollowingParams(
                         min_gap=2.5, tau=1.1),
                     num_vehicles=16)

        self.env, scenario = grid_mxn_exp_setup(row_num=1, col_num=3,
                                                vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def compare_ordering(self, ordering):
        # take in a list like [[bot0_0, right0_0, top0_1, left1_0], [bot....]
        # print(ordering)
        for x in ordering:
            # print(x)
            if not (x[0].startswith("bot") and x[1].startswith("right") and
                    x[2].startswith("top") and x[3].startswith("left")):
                return False
        return True

    def test_node_mapping(self):
        # reset the environment
        self.env.reset()

        node_mapping = self.env.scenario.get_node_mapping()
        nodes = [elem[0] for elem in node_mapping]
        ordering = [elem[1] for elem in node_mapping]

        self.assertEqual(nodes, sorted(nodes))
        self.assertTrue(self.compare_ordering(ordering))

    def test_k_closest(self):
        self.env.step(None)
        node_mapping = self.env.scenario.get_node_mapping()

        # get the node mapping for node center0
        c0_edges = node_mapping[0][1]
        k_closest = self.env.k_closest_to_intersection(c0_edges, 3)

        # check bot, right, top, left in that order
        self.assertEqual(self.env.k_closest_to_intersection(c0_edges[0], 3),
                         k_closest[0:2])
        self.assertEqual(self.env.k_closest_to_intersection(c0_edges[1], 3),
                         k_closest[2:4])
        self.assertEqual(
            len(self.env.k_closest_to_intersection(c0_edges[2], 3)), 0)
        self.assertEqual(self.env.k_closest_to_intersection(c0_edges[3], 3),
                         k_closest[4:6])

        for veh_id in k_closest:
            self.assertTrue(self.env.vehicles.get_edge(veh_id) in c0_edges)


class TestItRuns(unittest.TestCase):
    """
    Tests the set_state function
    """

    def setUp(self):
        vehicles = Vehicles()
        vehicles.add(veh_id="idm",
                     acceleration_controller=(IDMController, {}),
                     routing_controller=(GridRouter, {}),
                     sumo_car_following_params=SumoCarFollowingParams(
                         min_gap=2.5, tau=1.1),
                     num_vehicles=16)

        self.env, self.scenario = grid_mxn_exp_setup(row_num=1, col_num=3,
                                                     vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None
        self.scenario = None

    def test_it_runs(self):
        self.exp = SumoExperiment(self.env, self.scenario)
        self.exp.run(5, 50)


class TestIndividualLights(unittest.TestCase):
    """
    Tests the functionality of the the TrafficLights class in allowing
    for customization of specific nodes
    """

    def setUp(self):
        tl_logic = TrafficLights(baseline=False)
        phases = [{"duration": "31", "minDur": "8", "maxDur": "45",
                   "state": "GGGrrrGGGrrr"},
                  {"duration": "6", "minDur": "3", "maxDur": "6",
                   "state": "yyyrrryyyrrr"},
                  {"duration": "31", "minDur": "8", "maxDur": "45",
                   "state": "rrrGGGrrrGGG"},
                  {"duration": "6", "minDur": "3", "maxDur": "6",
                   "state": "rrryyyrrryyy"}]
        tl_logic.add("center0", phases=phases, programID=1)
        tl_logic.add("center1", phases=phases, programID=1, offset=1)
        tl_logic.add("center2", tls_type="actuated", phases=phases,
                     programID=1)
        tl_logic.add("center3", tls_type="actuated", phases=phases,
                     programID=1, maxGap=3.0, detectorGap=0.8,
                     showDetectors=True,
                     file="testindividuallights.xml", freq=100)

        self.env, self.scenario = grid_mxn_exp_setup(row_num=1, col_num=4,
                                                     tl_logic=tl_logic)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None
        self.scenario = None

    def test_it_runs(self):
        self.exp = SumoExperiment(self.env, self.scenario)
        self.exp.run(5, 50)


class TestCustomization(unittest.TestCase):

    def setUp(self):
        # add a traffic light to the top node
        traffic_lights = TrafficLights()

        # Phase durations in seconds
        self.green = 4
        self.yellow = 1
        self.red = 4
        phases = [{"duration": repr(self.green), "state": "G"},
                  {"duration": repr(self.yellow), "state": "y"},
                  {"duration": repr(self.red), "state": "r"}]

        traffic_lights.add("top", phases=phases)

        # create a ring road with two lanes
        additional_net_params = {"length": 230, "lanes": 1, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(net_params=net_params,
                                                 traffic_lights=traffic_lights)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_static_phases(self):
        # Reset the environment
        self.env.reset()

        # Calculate multiplier, because phases are in seconds 
        sim_multiplier = int(1 / self.env.sumo_params.sim_step)

        # Check that the phases occur for the correct amount of time
        for i in range(self.green * sim_multiplier - 1): # This is because env.reset() takes 1 step
            self.assertEqual(self.env.traffic_lights.get_state("top"), "G")
            self.env.step([])
        for i in range(self.yellow * sim_multiplier):
            self.assertEqual(self.env.traffic_lights.get_state("top"), "y")
            self.env.step([])
        for i in range(self.red * sim_multiplier):
            self.assertEqual(self.env.traffic_lights.get_state("top"), "r")
            self.env.step([])
        for i in range(3):
            for i in range(self.green * sim_multiplier):
                self.assertEqual(self.env.traffic_lights.get_state("top"), "G")
                self.env.step([])
            for i in range(self.yellow * sim_multiplier):
                self.assertEqual(self.env.traffic_lights.get_state("top"), "y")
                self.env.step([])
            for i in range(self.red * sim_multiplier):
                self.assertEqual(self.env.traffic_lights.get_state("top"), "r")
                self.env.step([])



if __name__ == '__main__':
    unittest.main()
