#!/usr/bin/python
import sys

class FastMarchingTreesPlanner:
    """
    Applies the Fast-Marching Trees algorithm on a given grid world

    Implementation taken from Janson et al.'s paper "Fast Marching Tree: a Fast Marching Sampling-Based Method for Optimal Motion Planning in Many Dimensions.
    """

    def __init__(self, world):
        # (rows, cols, channels) array with values in {0,..., 255}
        self.world = world

        # (rows, cols) binary array. Cell is 1 iff it is occupied
        self.occ_grid = self.world[:,:,0]
        self.occ_grid = (self.occ_grid == 0).astype('uint8')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: fast_marching_trees_planner.py occupancy_grid.pkl")
        sys.exit(1)
    
    print("Note: Finish this class before using it lol")