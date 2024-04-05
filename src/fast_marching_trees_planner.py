#!/usr/bin/python
import sys
import random
import numpy as np

from state import State
from math import sqrt, log


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

        self.start = None
        self.dest = None

        # configurable values
        self.sample_size = 1000
        self.delta = 0.5

        # sets of sample states
        self.S = set()
        self.S_unvisited = set()
        self.S_open = set()
        self.s_closed = set()

    def plan(self, start_state, dest_state, radius_size):
        """
        Fast Marching Tree Algorithm planning

        Parameters:
            start_state (State): Initial location.
            dest_state (state): Destination.
            radius_size (int): Connecting radius size

        Returns:
            None
        """
        self.start, self.dest = start_state, dest_state

        curr_s = self.start
        cost = {s: np.inf for s in self.S}
        cost[curr_s] = 0.0
        visited = []
        radius = radius_size * sqrt(log(self.sample_size) / self.sample_size)

        while curr_s is not self.dest:
            S_open_new = set()
            X_near = self.nearby(self.S_unvisited, curr_s, radius)
            visited.append(curr_s)

            # iterate through neighbors of start_state to find a locally-optimal one-step connection
            for x in X_near:
                Y_near = self.nearby(self.S_open, x, radius)
                cost_options = {y: cost[y] +  self.cost(y, x) for y in Y_near}
                y_min = min(cost_options, key=cost_options.get)

                if self.collision_free(y_min, x):
                    x.parent = y_min
                    S_open_new.add(x)
                    self.S_unvisited.remove(x)
                    cost[x] = cost[y_min] + self.cost(y_min, x)

            self.S_open.update(S_open_new)
            self.S_open.remove(curr_s)
            self.s_closed.add(curr_s)

            if not self.S_open:
                print("Error: open state set is empty.")
                break

            # update lowest-cost state from open state set
            cost_open = {y: cost[y] for y in self.S_open}
            curr_s = min(cost_open, key=cost_open.get)

        # retrieve path after reaching destination state
        path_x, path_y = self.ExtractPath()


    def state_is_free(self, state):
        """
        Does collision detection. Returns true iff the state and its nearby
        surroundings are free.
        """
        return (self.occ_grid[state.y-5:state.y+5, state.x-5:state.x+5] == 0).all()

    def init_helper(self):
        sample = self.sample_states()

        # update state sets
        self.S.add(self.start)
        self.S.update(sample)
        self.S_open.add(self.start)
        self.S_unvisited.update(sample)
        self.S_unvisited.update(self.dest)

    def sample_states(self) -> set:
        """
        Returns randomly sampled states that do not lay on obstacles

        Returns:
            set: Containing State objects.
        """
        num_samples = self.sample_size
        final_sample = set()

        i = 0
        while i < num_samples:
            # x = random.randint(0, self.world.shape[1] - 1)
            # y = random.randint(0, self.world.shape[0] - 1)
            state = State(random.randint(0, self.world.shape[1] - 1),
                           random.randint(0, self.world.shape[0] - 1),
                           None)
            if not self.state_is_free(state):
                # append to list of neighbors and increment index
                continue
            else:
                final_sample.add(state)
                i += 1
        return final_sample

    @staticmethod
    def nearby(states, curr_state, radius) -> dict:
        """
        Returns a dictionary of neighboring states within the connection radius

        Parameters:
            states (set): Set of unvisited states.
            curr_state (State): Current state.
            radius (int): Destination state.

        Returns:
            dict: Containing neighboring states within radius.
        """
        return {s for s in states if 0 < curr_state.euclidean_distance(s) <= radius ** 2}

    def cost(self, curr_state, dest_state) -> int:
        """
        Returns the distance between two states if the path does not collide with any obstacles

        Parameters:
            curr_state (State): Current state.
            dest_state (State): Destination

        Returns:
            int: Distance between two states if there are no collisions, infinity otherwise
        """
        if not self.collision_free(curr_state, dest_state): # check if there is a collision:
            return np.inf
        else:
            return curr_state.euclidean_distance(dest_state)

    def ExtractPath(self) -> tuple:
        """
        Returns the paths of start state to the destination state

        Returns:
            tuple: Containg the path of x's and y's from the start to destination state.
        """
        path_x, path_y = [], []
        state = self.dest

        while state.parent:
            path_x.append(state.x)
            path_y.append(state.y)
            state = state.parent

        path_x.append(self.start.x)
        path_y.append(self.start.y)
        return path_x, path_y

    def collision_free(self, s_from, s_to) -> bool:
        """
        Returns true iff the line path from s_from to s_to is free of collisions

        Parameters:
            s_from (State): Current state.
            s_to (state): Destination state.

        Returns:
            bool: True if free of collisions, False otherwise.
        """
        assert (self.state_is_free(s_from))

        if not (self.state_is_free(s_to)):
            return False

        max_checks = 10
        d = s_from.euclidean_distance(s_to)
        for i in range(max_checks):
            # check if the inteprolated state that is float(i)/max_checks * dist(s_from, s_new)
            # away on the line from s_from to s_new is free or not. If not free return False
            t = float(i) / max_checks # t = d_t/d = (float(i)/max_checks) * d / d
            interpolated_x = int((1 - t) * s_from.x + t * s_to.x)
            interpolated_y = int((1 - t) * s_from.y + t * s_to.y)
            interpolated_state = State(interpolated_x, interpolated_y, None)

            if not (self.state_is_free(interpolated_state)):
                return False

        # Otherwise the line is free, so return true
        return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: fast_marching_trees_planner.py occupancy_grid.pkl")
        sys.exit(1)
