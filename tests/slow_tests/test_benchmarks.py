import unittest
import os

from flow.benchmarks.baselines.bottleneck0 import bottleneck0_baseline
from flow.benchmarks.baselines.bottleneck1 import bottleneck1_baseline
from flow.benchmarks.baselines.bottleneck2 import bottleneck2_baseline
from flow.benchmarks.baselines.figureeight012 import figure_eight_baseline
from flow.benchmarks.baselines.grid0 import grid0_baseline
from flow.benchmarks.baselines.grid1 import grid1_baseline
from flow.benchmarks.baselines.merge012 import merge_baseline

os.environ["TEST_FLAG"] = "True"


class TestBaselines(unittest.TestCase):

    """
    Tests that the baselines in the benchmarks folder are running and
    returning expected values (i.e. values that match those in the CoRL paper
    reported on the website, or other).
    """

    def test_bottleneck0(self):
        """
        Tests flow/benchmark/baselines/bottleneck0.py
        """
        # run the bottleneck to make sure it runs
        res = bottleneck0_baseline(num_runs=1, sumo_binary="sumo")

        # check that the resulting performance measure is within some range
        self.assertGreaterEqual(res, 400)
        self.assertLessEqual(res, 1400)

    def test_bottleneck1(self):
        """
        Tests flow/benchmark/baselines/bottleneck1.py
        """
        # run the bottleneck to make sure it runs
        res = bottleneck1_baseline(num_runs=1, sumo_binary="sumo")

        # check that the resulting performance measure is within some range
        self.assertGreaterEqual(res, 400)
        self.assertLessEqual(res, 1400)

    def test_bottleneck2(self):
        """
        Tests flow/benchmark/baselines/bottleneck2.py
        """
        # run the bottleneck to make sure it runs
        res = bottleneck2_baseline(num_runs=1, sumo_binary="sumo")

        # check that the resulting performance measure is within some range
        self.assertGreaterEqual(res, 1300)
        self.assertLessEqual(res, 1700)

    def test_figure_eight(self):
        """
        Tests flow/benchmark/baselines/figureeight{0,1,2}.py
        """
        # run the bottleneck to make sure it runs
        res = figure_eight_baseline(num_runs=1, sumo_binary="sumo")

        # check that the resulting performance measure is within some range
        self.assertGreaterEqual(res, 4.0)
        self.assertLessEqual(res, 4.4)

    def test_grid0(self):
        """
        Tests flow/benchmark/baselines/grid0.py
        """
        # run the bottleneck to make sure it runs
        res = grid0_baseline(num_runs=1, sumo_binary="sumo")

        # check that the resulting performance measure is within some range
        self.assertGreaterEqual(res, 34000)
        self.assertLessEqual(res, 36000)

    def test_grid1(self):
        """
        Tests flow/benchmark/baselines/grid1.py
        """
        # run the bottleneck to make sure it runs
        res = grid1_baseline(num_runs=1, sumo_binary="sumo")

        # check that the resulting performance measure is within some range
        self.assertGreaterEqual(res, 71000)
        self.assertLessEqual(res, 73000)

    def test_merge(self):
        """
        Tests flow/benchmark/baselines/merge{0,1,2}.py
        """
        # run the bottleneck to make sure it runs
        res = merge_baseline(num_runs=1, sumo_binary="sumo")

        # check that the resulting performance measure is within some range
        self.assertGreaterEqual(res, 7.5)
        self.assertLessEqual(res, 9.5)


if __name__ == '__main__':
    unittest.main()
