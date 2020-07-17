"""
PyCrown - Fast raster-based individual tree segmentation for LiDAR data
-----------------------------------------------------------------------
Copyright: 2018, Jan Zörner
Licence: GNU GPLv3
"""

from numba import jit, float32, int32, float_
import numpy as np


@jit(int32[:, :](float32[:, :], int32[:, :], float_, float_, float_, float_),
     nopython=True, nogil=True, parallel=False)
def _crown_dalponte(Chm, Trees, th_seed, th_crown, th_tree, max_crown):
    '''
    Crown delineation based on Dalponte and Coomes (2016) and
    lidR R-package (https://github.com/Jean-Romain/lidR/)

    Parameters
    ----------
    Chm :       ndarray
                Canopy height model as n x m raster
    Trees :     ndarray
                Tree top pixel coordinates as nx2 ndarray
    th_tree :   float
                Threshold below which a pixel cannot be a tree. Default 2
    th_seed :   float
                Growing threshold 1. A pixel is added to a region if its height
                is greater than the tree height multiplied by this value. It
                should be between 0 and 1. Default 0.45
    th_crown :  float
                Growing threshold 2. A pixel is added to a region if its height
                is greater than the current mean height of the region
                multiplied by this value. It should be between 0 and 1.
                Default 0.55.
    max_crown : float
                Maximum value of the crown diameter of a detected tree (in
                pixels). Default 10

    Returns
    -------
    Cronws :    ndarray
                Raster of tree crowns
    '''

    grown = True
    nrow = Chm.shape[0]
    ncol = Chm.shape[1]
    ntops = Trees.shape[1]
    npixel = np.ones(ntops, dtype=np.float32)
    neighbours = np.zeros((4, 2)).astype(np.int32)
    tidx_x = np.floor(Trees[0]).astype(np.intp)
    tidx_y = np.floor(Trees[1]).astype(np.intp)
    Crowns = np.zeros((nrow, ncol), dtype=np.int32)
    sum_height = np.zeros(ntops)
    for i in range(ntops):
        Crowns[tidx_y[i], tidx_x[i]] = i + 1
        sum_height[i] = Chm[tidx_y[i], tidx_x[i]]
    CrownsTemp = Crowns.copy()

    while grown:
        grown = False
        for row in range(1, nrow - 1):
            for col in range(1, ncol - 1):

                # enter if pixel belongs to a tree top or tree crown
                if Crowns[row, col]:

                    # id of the tree crown for the current pixel
                    tidx = Crowns[row, col] - 1

                    # Pixel coordinates of current seed
                    seed_y = tidx_y[tidx]
                    seed_x = tidx_x[tidx]

                    # Seed height
                    seed_h = Chm[seed_y, seed_x]

                    # Mean height of the crown
                    mh_crown = sum_height[tidx] / npixel[tidx]

                    # Positions of the 4 neighbours
                    neighbours[0, 0] = row - 1
                    neighbours[0, 1] = col
                    neighbours[1, 0] = row
                    neighbours[1, 1] = col - 1
                    neighbours[2, 0] = row
                    neighbours[2, 1] = col + 1
                    neighbours[3, 0] = row + 1
                    neighbours[3, 1] = col

                    # Go through neighbourhood
                    for j in range(4):
                        # Pixel coordinates of current neighbour
                        nb_y = neighbours[j, 0]
                        nb_x = neighbours[j, 1]
                        # Neighbour height
                        nb_h = Chm[nb_y, nb_x]

                        # Perform different checks:
                        # 1. Neighbour height is above minimum threshold
                        # 2. Neighbour does not belong to other crown
                        # 3. Neighbour height is above threshold 1
                        # 4. Neighbour height is above threshold 2
                        # 5. Neighbour height below treetop+5%
                        # 7. Neighbour is not too far from the tree top (x-dir)
                        # 8. Neighbour is not too far from the tree top (y-dir)
                        if nb_h > th_tree and \
                           not CrownsTemp[nb_y, nb_x] and \
                           nb_h > (seed_h * th_seed) and \
                           nb_h > (mh_crown * th_crown) and \
                           nb_h <= (seed_h * 1.05) and \
                           abs(seed_x-nb_x) < max_crown and \
                           abs(seed_y-nb_y) < max_crown:
                            # If all conditions are met, add neighbour to crown
                            CrownsTemp[nb_y, nb_x] = Crowns[row, col]
                            npixel[tidx] += 1
                            sum_height[tidx] += nb_h
                            grown = True

        Crowns = CrownsTemp.copy()

    return Crowns
