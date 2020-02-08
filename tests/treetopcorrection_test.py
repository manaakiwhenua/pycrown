import unittest
from pycrown import PyCrown


class TestTreeTopCorrection(unittest.TestCase):

    def setUp(self):
        ''' initialize test scenario '''
        F_CHM = 'example/data/CHM.tif'
        F_DTM = 'example/data/DTM.tif'
        F_DSM = 'example/data/DSM.tif'
        F_LAS = 'example/data/POINTS.las'
        self.PC = PyCrown(F_CHM, F_DTM, F_DSM, F_LAS)
        self.PC.filter_chm(5)
        self.PC.tree_detection(self.PC.chm, ws=5, hmin=16.)
        self.PC.clip_trees_to_bbox(bbox=(1802160, 1802400, 5467315, 5467470))
        self.PC.crown_delineation(algorithm='dalponteCIRC_numba', th_tree=15.,
                                  th_seed=0.7, th_crown=0.55, max_crown=10.)
        self.PC.get_tree_height_elevation()

    def test_number_corrected_trees(self):
        ''' test the number of corrected trees per method:
        A: DSM top positon, B: centre of mass of tree crown '''
        corr_dsm, corr_com = self.PC.correct_tree_tops()
        self.assertEqual(corr_dsm, 5)
        self.assertEqual(corr_com, 4)

    def test_tree_height_corrected_trees(self):
        ''' test that the corrected tree heights are on average lower than
        the original tree heights (based on assumption that trees are situated
        on steep terrain and or crowns overwang cliff edges) '''
        _, _ = self.PC.correct_tree_tops()
        self.PC.get_tree_height_elevation(loc='top')
        self.PC.get_tree_height_elevation(loc='top_cor')

        trees = self.PC.trees[self.PC.trees.tt_corrected > 0.]

        dsm_corrected = trees.tt_corrected == 1.
        differences_dsm_correction = (trees.top_cor_height[dsm_corrected] -
                                      trees.top_height[dsm_corrected])
        self.assertTrue(differences_dsm_correction.mean() < 0.)

        com_corrected = trees.tt_corrected == 2.
        differences_com_correction = (trees.top_cor_height[com_corrected] -
                                      trees.top_height[com_corrected])
        self.assertTrue(differences_com_correction.mean() < 0.)


if __name__ == '__main__':
    unittest.main()
