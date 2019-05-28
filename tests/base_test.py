import unittest
from pycrown import PyCrown

class TestExampleNoLAS(unittest.TestCase):

    def setUp(self):
        ''' initialize test scenario '''
        F_CHM = 'tests/data/CHM.tif'
        F_DTM = 'tests/data/DTM.tif'
        F_DSM = 'tests/data/DSM.tif'
        self.PC = PyCrown(F_CHM, F_DTM, F_DSM, outpath="./")

    def test_treedetection_without_smoothing(self):
        self.PC.tree_detection(self.PC.chm0, ws=5, hmin=16.)
        self.assertGreater(self.PC.trees.size, 0)

    def test_treedetection_with_smoothing(self):
        self.PC.filter_chm(5)
        self.PC.tree_detection(self.PC.chm, ws=5, hmin=16.)
        self.assertGreater(self.PC.trees.size, 0)

    def test_crowndelineation(self):
        self.PC.filter_chm(5)
        self.PC.tree_detection(self.PC.chm, ws=5, hmin=16.)
        self.PC.crown_delineation(
            algorithm='dalponteCIRC_numba', th_tree=15.,
            th_seed=0.7, th_crown=0.55, max_crown=10.
        )
        self.PC.correct_tree_tops()
        self.PC.get_tree_height_elevation(loc='top')
        self.PC.get_tree_height_elevation(loc='top_cor')
        self.PC.screen_small_trees(hmin=20., loc='top')
        self.PC.crowns_to_polys_raster()
        self.PC.quality_control()


if __name__ == '__main__':
    unittest.main()
