#!/usr/bin/python
import sys

class InformedRRTStarPlanner:
    """
    Applies the Informed RRT* algorithm on a given grid world

    Implementation taken from Gammell, Srinivasa, and Barfoot's paper at http://dx.doi.org/10.1109/IROS.2014.6942976
    """

    def __init__(self, world):
        # (rows, cols, channels) array with values in {0,..., 255}
        self.world = world

        # (rows, cols) binary array. Cell is 1 iff it is occupied
        self.occ_grid = self.world[:,:,0]
        self.occ_grid = (self.occ_grid == 0).astype('uint8')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: informed_rrt_star_planner.py occupancy_grid.pkl")
        sys.exit(1)
    
    print("Note: Finish this class before using it lol")