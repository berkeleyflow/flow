import unittest

from flow.core.experiment import SumoExperiment
from flow.core.params import EnvParams, InitialConfig, NetParams
from flow.core.vehicles import Vehicles
from flow.core.params import SumoCarFollowingParams

from flow.controllers.routing_controllers import ContinuousRouter
from flow.controllers.car_following_models import IDMController, \
    OVMController, BCMController, LinearOVM, CFMController
from tests.setup_scripts import ring_road_exp_setup
import os
import numpy as np

os.environ["TEST_FLAG"] = "True"


class TestCFMController(unittest.TestCase):
    """
    Tests that the CFM Controller returning mathematically accurate values.
    """
    def setUp(self):
        # add a few vehicles to the network using the requested model
        # also make sure that the input params are what is expected

        contr_params = {"time_delay": 0, "k_d": 1, "k_v": 1, "k_c": 1,
                        "d_des": 1, "v_des": 8, "noise": 0}

        vehicles = Vehicles()
        vehicles.add(
            veh_id="test_0",
            acceleration_controller=(CFMController, contr_params),
            routing_controller=(ContinuousRouter, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                accel=20, decel=5),
            num_vehicles=5
        )

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_get_action(self):
        self.env.reset()
        ids = self.env.vehicles.get_ids()

        test_headways = [5, 10, 15, 20, 25]
        for i, veh_id in enumerate(ids):
            self.env.vehicles.set_headway(veh_id, test_headways[i])

        requested_accel = [self.env.vehicles.get_acc_controller(
            veh_id).get_action(self.env) for veh_id in ids]

        expected_accel = [12., 17., 22., 27., 32.]

        np.testing.assert_array_almost_equal(requested_accel, expected_accel)


class TestBCMController(unittest.TestCase):
    """
    Tests that the BCM Controller returning mathematically accurate values.
    """
    def setUp(self):
        # add a few vehicles to the network using the requested model
        # also make sure that the input params are what is expected
        contr_params = \
            {"time_delay": 0, "k_d": 1, "k_v": 1, "k_c": 1, "d_des": 1,
             "v_des": 8, "noise": 0}

        vehicles = Vehicles()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(BCMController, contr_params),
            routing_controller=(ContinuousRouter, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                accel=15, decel=5),
            num_vehicles=5
        )

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_get_action(self):
        self.env.reset()
        ids = self.env.vehicles.get_ids()

        test_headways = [5, 10, 15, 20, 25]
        for i, veh_id in enumerate(ids):
            self.env.vehicles.set_headway(veh_id, test_headways[i])

        requested_accel = [self.env.vehicles.get_acc_controller(
            veh_id).get_action(self.env) for veh_id in ids]

        expected_accel = [-12., 13., 13., 13., 13.]

        np.testing.assert_array_almost_equal(requested_accel, expected_accel)


class TestOVMController(unittest.TestCase):
    """
    Tests that the OVM Controller returning mathematically accurate values.
    """
    def setUp(self):
        # add a few vehicles to the network using the requested model
        # also make sure that the input params are what is expected
        contr_params = {"time_delay": 0, "alpha": 1, "beta": 1, "h_st": 2,
                        "h_go": 15, "v_max": 30, "noise": 0}

        vehicles = Vehicles()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(OVMController, contr_params),
            routing_controller=(ContinuousRouter, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                accel=15, decel=5),
            num_vehicles=5
        )

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_get_action(self):
        self.env.reset()
        ids = self.env.vehicles.get_ids()

        test_headways = [0, 10, 5, 5, 5]
        for i, veh_id in enumerate(ids):
            self.env.vehicles.set_headway(veh_id, test_headways[i])

        requested_accel = [self.env.vehicles.get_acc_controller(
            veh_id).get_action(self.env) for veh_id in ids]

        expected_accel = [0., 20.319073, 3.772339, 3.772339, 3.772339]

        np.testing.assert_array_almost_equal(requested_accel, expected_accel)


class TestLinearOVM(unittest.TestCase):
    """
    Tests that the Linear OVM Controller returning mathematically accurate
    values.
    """
    def setUp(self):
        # add a few vehicles to the network using the requested model
        # also make sure that the input params are what is expected
        contr_params = \
            {"time_delay": 0, "v_max": 30, "adaptation": 0.65,
             "h_st": 5, "noise": 0}

        vehicles = Vehicles()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(LinearOVM, contr_params),
            routing_controller=(ContinuousRouter, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                accel=15, decel=5),
            num_vehicles=5
        )

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_get_action(self):
        self.env.reset()
        ids = self.env.vehicles.get_ids()

        test_headways = [5, 10, 10, 15, 0]
        for i, veh_id in enumerate(ids):
            self.env.vehicles.set_headway(veh_id, test_headways[i])

        requested_accel = [self.env.vehicles.get_acc_controller(
            veh_id).get_action(self.env) for veh_id in ids]

        expected_accel = [0., 12.992308, 12.992308, 25.984615, 0.]

        np.testing.assert_array_almost_equal(requested_accel, expected_accel)


class TestIDMController(unittest.TestCase):
    """
    Tests that the IDM Controller returning mathematically accurate values.
    """
    def setUp(self):
        # add a few vehicles to the network using the requested model
        # also make sure that the input params are what is expected
        contr_params = {"v0": 30, "b": 1.5, "delta": 4,
                        "s0": 2, "noise": 0}

        vehicles = Vehicles()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(IDMController, contr_params),
            routing_controller=(ContinuousRouter, {}),
            sumo_car_following_params=SumoCarFollowingParams(
                tau=1, accel=1, decel=5),
            num_vehicles=5
        )

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(vehicles=vehicles)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_get_action(self):
        self.env.reset()
        ids = self.env.vehicles.get_ids()

        test_headways = [10, 20, 30, 40, 50]
        for i, veh_id in enumerate(ids):
            self.env.vehicles.set_headway(veh_id, test_headways[i])

        requested_accel = [self.env.vehicles.get_acc_controller(
            veh_id).get_action(self.env) for veh_id in ids]

        expected_accel = [0.96, 0.99, 0.995556, 0.9975, 0.9984]

        np.testing.assert_array_almost_equal(requested_accel, expected_accel)

        # set the perceived headway to zero
        test_headways = [0, 0, 0, 0, 0]
        for i, veh_id in enumerate(ids):
            self.env.vehicles.set_headway(veh_id, test_headways[i])

        # make sure the controller doesn't return a ZeroDivisionError when the
        # headway is zero
        [self.env.vehicles.get_acc_controller(veh_id).get_action(self.env)
         for veh_id in ids]


class TestInstantaneousFailsafe(unittest.TestCase):
    """
    Tests that the instantaneous failsafe of the base acceleration controller
    does not allow vehicles to crash under situations where they otherwise
    would. This is tested on two crash-prone controllers: OVM and LinearOVM
    """
    def setUp_failsafe(self, vehicles):
        additional_env_params = {"target_velocity": 8, "max_accel": 3,
                                 "max_decel": 3}
        env_params = EnvParams(additional_params=additional_env_params)

        additional_net_params = {"length": 100, "lanes": 1, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        initial_config = InitialConfig(bunching=10)

        # create the environment and scenario classes for a ring road
        env, scenario = ring_road_exp_setup(vehicles=vehicles,
                                            env_params=env_params,
                                            net_params=net_params,
                                            initial_config=initial_config)

        # instantiate an experiment class
        self.exp = SumoExperiment(env, scenario)

    def tearDown_failsafe(self):
        # free data used by the class
        self.exp = None

    def test_no_crash_OVM(self):
        vehicles = Vehicles()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(OVMController,
                                     {"fail_safe": "instantaneous"}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=10,
        )

        self.setUp_failsafe(vehicles=vehicles)

        # run the experiment, see if it fails
        self.exp.run(1, 200)

        self.tearDown_failsafe()

    def test_no_crash_LinearOVM(self):
        vehicles = Vehicles()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(LinearOVM,
                                     {"fail_safe": "instantaneous"}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=10
        )

        self.setUp_failsafe(vehicles=vehicles)

        # run the experiment, see if it fails
        self.exp.run(1, 200)

        self.tearDown_failsafe()


class TestSafeVelocityFailsafe(TestInstantaneousFailsafe):
    """
    Tests that the safe velocity failsafe of the base acceleration controller
    does not fail under extreme conditions.
    """
    def test_no_crash_OVM(self):
        vehicles = Vehicles()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(OVMController,
                                     {"fail_safe": "safe_velocity"}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=10,
        )

        self.setUp_failsafe(vehicles=vehicles)

        # run the experiment, see if it fails
        self.exp.run(1, 200)

        self.tearDown_failsafe()

    def test_no_crash_LinearOVM(self):
        vehicles = Vehicles()
        vehicles.add(
            veh_id="test",
            acceleration_controller=(LinearOVM,
                                     {"fail_safe": "safe_velocity"}),
            routing_controller=(ContinuousRouter, {}),
            num_vehicles=10,
        )

        self.setUp_failsafe(vehicles=vehicles)

        # run the experiment, see if it fails
        self.exp.run(1, 200)

        self.tearDown_failsafe()


class TestStaticLaneChanger(unittest.TestCase):
    """
    Makes sure that vehicles with a static lane-changing controller do not
    change lanes.
    """
    def setUp(self):
        # add an extra lane to the ring road network
        additional_net_params = {"length": 230, "lanes": 2, "speed_limit": 30,
                                 "resolution": 40}
        net_params = NetParams(additional_params=additional_net_params)

        # create the environment and scenario classes for a ring road
        self.env, scenario = ring_road_exp_setup(net_params=net_params)

    def tearDown(self):
        # terminate the traci instance
        self.env.terminate()

        # free data used by the class
        self.env = None

    def test_static_lane_changer(self):
        self.env.reset()
        ids = self.env.vehicles.get_ids()

        # run the experiment for a few iterations and collect the lane index
        # for vehicles
        lanes = [self.env.vehicles.get_lane(veh_id) for veh_id in ids]
        for i in range(5):
            self.env.step(rl_actions=[])
            lanes += [self.env.vehicles.get_lane(veh_id) for veh_id in ids]

        # set the timer as very high and reset (the timer used to cause bugs at
        # the beginning of a new run for this controller)
        self.env.timer = 10000
        self.env.reset()

        # run the experiment for a few more iterations and collect the lane
        # index for vehicles
        lanes = [self.env.vehicles.get_lane(veh_id) for veh_id in ids]
        for i in range(5):
            self.env.step(rl_actions=[])
            lanes += [self.env.vehicles.get_lane(veh_id) for veh_id in ids]

        # assert that all lane indices are zero
        self.assertEqual(sum(np.array(lanes)), 0)


if __name__ == '__main__':
    unittest.main()
