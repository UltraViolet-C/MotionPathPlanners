#!/usr/bin/python
import sys
import random
import cv2
import numpy as np
import pickle
import math

from plotting_utils import draw_plan
from state import State

class InformedRRTStarPlanner:
    """
    Applies the Informed RRT* algorithm on a given grid world

    Implementation taken from Gammell, Srinivasa, and Barfoot's paper at http://dx.doi.org/10.1109/IROS.2014.6942976

    Helper functions also use implementations from the following github repo: https://github.com/zhm-real/PathPlanning/
    """

    def __init__(self, world):
        # (rows, cols, channels) array with values in {0,..., 255}
        self.world = world

        # (rows, cols) binary array. Cell is 1 iff it is occupied
        self.occ_grid = self.world[:,:,0]
        self.occ_grid = (self.occ_grid == 0).astype('uint8')

    def state_is_free(self, state):
        """
        Does collision detection. Returns true iff the state and its nearby
        surroundings are free.
        """
        return (self.occ_grid[state.y-5:state.y+5, state.x-5:state.x+5] == 0).all()
    
    def sample_state(self, c_max, c_min, s_center, C):
        """
        Sample a new state uniformly randomly within an ellipsoid drawn over the initial and final state.
        """
        if c_max < np.inf:
            r = [c_max / 2, math.sqrt(c_max**2 - c_min**2) / 2, math.sqrt(c_max**2 - c_min**2) / 2]
            L = np.diag(r)
            
            # loop until a valid random state is found
            while True:
                s_ball = self.sample_unit_ball()
                s_rand = np.dot(C, np.dot(L, s_ball)) + s_center
                # check whether s_rand is within the world's limits
                if 0 <= s_rand[0] < self.world.shape[1] - 1 and 0 <= s_rand[1] < self.world.shape[0] - 1:
                    break
            s_rand = State(int(s_rand[(0, 0)]), int(s_rand[(1, 0)]), None)
        else:
            s_rand = self.sample_free_space()

        return s_rand

    @staticmethod
    def rotation_to_world_frame(s_start, s_goal, L):
        # implementation from zhm-real's repo
        a1 = np.array([[(s_goal.x - s_start.x) / L],
                       [(s_goal.y - s_start.y) / L], [0.0]])
        e1 = np.array([[1.0], [0.0], [0.0]])
        M = a1 @ e1.T
        U, _, V_T = np.linalg.svd(M, True, True)
        C = U @ np.diag([1.0, 1.0, np.linalg.det(U) * np.linalg.det(V_T.T)]) @ V_T

        return C

    @staticmethod
    def sample_unit_ball():
        """
        Sample a state from a uniform radius of unit radius centered at the origin.
        """
        # implementation from zhm-real's repo
        while True:
            x, y = random.randint(-1, 1), random.randint(-1, 1)
            if x ** 2 + y ** 2 <= 1:
                # returned in array form because it's multiplied by a matrix in sample_state() later
                return np.array([[x], [y], [0.0]])
        

    def sample_free_space(self):
        """
        Sample a random state from the state space.
        """
        x = random.randint(0, self.world.shape[1] - 1)
        y = random.randint(0, self.world.shape[0] - 1)
        return State(x, y, None)

    def _follow_parent_pointers(self, state):
        """
        Returns the path [start_state, ..., destination_state] by following the
        parent pointers.
        """

        curr_ptr = state
        path = [state]

        while curr_ptr is not None:
            path.append(curr_ptr)
            curr_ptr = curr_ptr.parent

        # return a reverse copy of the path (so that first state is starting state)
        return path[::-1]
    
    def find_closest_state(self, tree_nodes, state):
        min_dist = float("Inf")
        closest_state = None
        for node in tree_nodes:
            dist = node.euclidean_distance(state)
            if dist < min_dist:
                closest_state = node
                min_dist = dist

        return closest_state
    
    def find_near_states(self, tree_nodes, state):
        near_states = []
        for node in tree_nodes:
            dist = node.euclidean_distance(state)
            # this constant was chosen arbitrarily - Yuvraaj
            if dist <= 60:
                near_states.append(node)
        return near_states
    
    def steer_towards(self, s_nearest, s_rand, max_radius):
        """
        Returns a new state s_new whose coordinates x and y
        are decided as follows:

        If s_rand is within a circle of max_radius from s_nearest
        then s_new.x = s_rand.x and s_new.y = s_rand.y

        Otherwise, s_rand is farther than max_radius from s_nearest.
        In this case we place s_new on the line from s_nearest to
        s_rand, at a distance of max_radius away from s_nearest.

        """
        #Note: x and y are integers and they should be in {0, ..., cols -1}
        # and {0, ..., rows -1} respectively
        x = 0
        y = 0

        if s_nearest.euclidean_distance(s_rand) <= max_radius:
            x = s_rand.x
            y = s_rand.y
        else:
            # place s_new on the line between s_nearest (x_0, y_0) to s_rand (x_1, y_1) at a distance of max_radius
            # given d_t = max_radius, d = d(s_nearest, s_rand), and t = d_t/d then,
            # our point is (x_t, y_t) = (((1−t)*x_0 + t*x_1), ((1−t)*y_0 + t*y_1))
            # this math is taken from https://math.stackexchange.com/a/1630886
            d = s_nearest.euclidean_distance(s_rand)
            t = max_radius / d
            x = int((1-t)*s_nearest.x + t*s_rand.x)
            y = int((1-t)*s_nearest.y + t*s_rand.y)

        s_new = State(x, y, s_nearest)
        return s_new
    
    def path_is_obstacle_free(self, s_from, s_to):
        """
        Returns true iff the line path from s_from to s_to
        is free
        """
        assert (self.state_is_free(s_from))

        if not (self.state_is_free(s_to)):
            return False

        max_checks = 10
        d = s_from.euclidean_distance(s_to)
        for i in range(max_checks):
            # check if the inteprolated state that is float(i)/max_checks * dist(s_from, s_new)
            # away on the line from s_from to s_new is free or not. If not free return False
            t = float(i)/max_checks # t = d_t/d = (float(i)/max_checks) * d / d
            interpolated_x = int((1-t)*s_from.x + t*s_to.x)
            interpolated_y = int((1-t)*s_from.y + t*s_to.y)
            interpolated_state = State(interpolated_x, interpolated_y, None)

            if not (self.state_is_free(interpolated_state)):
                return False

        # Otherwise the line is free, so return true
        return True
    
    def heuristic_cost(self, s_start, s, s_goal):
        """
        Estimates the cost of a path from s_start to s_goal that passes through state s
        """
        # TODO: implement this
        return -1
    
    def cost(self, state, start_state):
        if state.x == start_state.x and state.y == start_state.y:
            return 0
        
        if state.parent is None:
            return np.inf
        
        cost = 0
        while state.parent is not None:
            cost += state.euclidean_distance(state.parent)
            state = state.parent
        
        return cost
    
    def plan(self, start_state, dest_state, max_num_steps, max_steering_radius, dest_reached_radius):
        """
        Returns a path as a sequence of states [start_state, ..., dest_state]
        if dest_state is reachable from start_state. Otherwise returns [start_state].
        Assume both source and destination are in free space.
        """
        # TODO: implement this

        assert (self.state_is_free(start_state))
        assert (self.state_is_free(dest_state))

        # The set containing the nodes of the tree
        tree_nodes = set()
        tree_nodes.add(start_state)

        # image to be used to display the tree
        img = np.copy(self.world)

        plan = [start_state]
        s_soln = set() # set of all solution states

        # need to find c_min, x_center, and C once
        c_min = start_state.euclidean_distance(dest_state)
        s_center = np.array([[(start_state.x + dest_state.x) / 2.0],
                             [(start_state.y + dest_state.y) / 2.0], [0.0]])
        C = self.rotation_to_world_frame(start_state, dest_state, c_min)

        for step in range(max_num_steps):
            # find the solution state with the lowest cost to see if a better one can be found this iteration
            c_best = np.inf
            if s_soln:
                cost = {state: self.cost(state, start_state) for state in s_soln}
                s_best = min(cost, key=cost.get)
                c_best = cost[s_best]
            s_rand = self.sample_state(c_best, c_min, s_center, C)
            s_nearest = self.find_closest_state(tree_nodes, s_rand)
            s_new = self.steer_towards(s_nearest, s_rand, max_steering_radius)

            if self.path_is_obstacle_free(s_nearest, s_new):
                tree_nodes.add(s_new)
                s_min = s_nearest
                nearby_states = self.find_near_states(tree_nodes, s_new)
                for s_near in nearby_states:
                    if self.path_is_obstacle_free(s_near, s_new):
                        c_new = self.cost(s_near, start_state) + s_near.euclidean_distance(s_new)
                        if c_new < c_min:
                            s_min = s_near
                            c_min = c_new
                
                s_min.children.append(s_new)
                for s_near in nearby_states:
                    c_near = self.cost(s_near, start_state)
                    c_new = self.cost(s_new, start_state) + s_near.euclidean_distance(s_new)
                    if s_near != s_min and self.path_is_obstacle_free(s_new, s_near) and c_new < c_near:
                        # a better node has been found to be s_near's parent so refactor the tree
                        s_parent = s_near.parent
                        s_parent.delete_child(s_near)
                        s_new.children.append(s_near)
                        s_near.parent = s_new
                
                # check if new node is close to destination
                if s_new.euclidean_distance(dest_state) < dest_reached_radius:
                    s_soln.add(s_new) # add new state to set of solution states
                
                # plot new node and edge
                cv2.circle(img, (s_new.x, s_new.y), 2, (0,0,0))
                cv2.line(img, (s_nearest.x, s_nearest.y), (s_new.x, s_new.y), (255,0,0))
            
            # keep showing image even if new node is not added
            cv2.imshow('image', img)
            cv2.waitKey(10)
        
        # find plan by looking for c_best in s_soln and drawing that plan
        if s_soln:
            cost = {state: self.cost(state, start_state) for state in s_soln}
            s_best = min(cost, key=cost.get)
            dest_state.parent = s_best
            plan = self._follow_parent_pointers(dest_state)
        draw_plan(img, plan, bgr=(0,0,255), thickness=2)
        cv2.waitKey(0)
        return plan

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: informed_rrt_star_planner.py occupancy_grid.pkl")
        sys.exit(1)
    
    pkl_file = open(sys.argv[1], 'rb')
    # world is a numpy array with dimensions (rows, cols, 3 color channels)
    world = pickle.load(pkl_file)
    pkl_file.close()

    informed_rrt_star = InformedRRTStarPlanner(world)

    start_state = State(10, 10, None) # default is (10, 10)
    dest_state = State(500, 500, None) # default is (500, 500)
    max_num_steps = 1000     # max number of nodes to be added to the tree
    max_steering_radius = 30 # pixels
    dest_reached_radius = 50 # pixels
    plan = informed_rrt_star.plan(start_state,
                                  dest_state,
                                  max_num_steps,
                                  max_steering_radius,
                                  dest_reached_radius)
