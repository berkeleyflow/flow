import unittest
import os

from flow.core.experiment import SumoExperiment
from tests.setup_scripts import ring_road_exp_setup
import numpy as np

os.environ["TEST_FLAG"] = "True"
class TestDepartTimeController(unittest.TestCase):
    """
    Tests that departure times being in the past do not cause an error
    """
    def setUp(self):
        # create the environment and scenario classes for a ring road
        self.env, self.scenario = ring_road_exp_setup()
        self.exp = SumoExperiment(self.env, self.scenario)


    def runTest(self):
        vehicles = self.env.vehicles
        ids = vehicles.get_ids()
        traci_con = self.env.traci_connection
        traci_con.vehicle.addFull('test', 'top', depart=-100)
        self.exp.run(num_runs=1, num_steps=10)
        self.assertEqual(self.exp.env.time_counter, 10)


    def tearDown(self):
        # free up used memory
        self.exp = None
        
