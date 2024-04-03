#!/usr/bin/python
import sys

class BITStarPlanner:
    """
    Applies the BIT* algorithm on a given grid world
    """

    def __init__(self, world):
        # (rows, cols, channels) array with values in {0,..., 255}
        self.world = world

        # (rows, cols) binary array. Cell is 1 iff it is occupied
        self.occ_grid = self.world[:,:,0]
        self.occ_grid = (self.occ_grid == 0).astype('uint8')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: bit_star_planner.py occupancy_grid.pkl")
        sys.exit(1)
    
    print("Note: Finish this class before using it lol")