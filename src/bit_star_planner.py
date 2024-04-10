#!/usr/bin/python
import sys
import random
import cv2
import numpy as np
import pickle
import math

from plotting_utils import draw_plan
from state import State

class BITStarPlanner:
    """
    Applies the BIT* algorithm on a given grid world

    Implementation taken from Gammell, Srinivasa, and Barfoot's paper at http://dx.doi.org/10.1109/ICRA.2015.7139620
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
    
    def find_near_states(self, tree_nodes, state, max_dist):
        near_states = []
        for node in tree_nodes:
            dist = node.euclidean_distance(state)
            # this value can be adjusted freely, I just chose a constant we were already using - Yuvraaj
            if dist <= max_dist:
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
    

    def plan(self, start_state, dest_state, max_num_steps, max_steering_radius, dest_reached_radius, plot_graphic=True):
        """
        Returns a path as a sequence of states [start_state, ..., dest_state]
        if dest_state is reachable from start_state. Otherwise returns [start_state].
        Assume both source and destination are in free space.
        """
        assert (self.state_is_free(start_state))
        assert (self.state_is_free(dest_state))
        m = 5000

        img = np.copy(self.world)

        best_plan = [start_state]

        tree_nodes = set(best_plan)
        old_nodes = set()

        q_v = set()

        q_e = set()

        samples = set([dest_state])

        #lowest recorded cost to come for each node
        costs = {start_state: 0}

        #shortest path yet found
        max_cost = np.inf
        min_cost = start_state.euclidean_distance(dest_state)

        center =  np.array([[(start_state.x + dest_state.x) / 2.0],
                             [(start_state.y + dest_state.y) / 2.0], [0.0]])
        C = self.rotation_to_world_frame(start_state, dest_state, min_cost)


        for step in range(max_num_steps):
            
            if not q_v and not q_e:
                #pruning edges that can't improve the current best solution 
                for v in list(samples):
                     if v.euclidean_distance(start_state) + v.euclidean_distance(dest_state) > max_cost:
                         samples.remove(v)

                for v in list(tree_nodes):
                    if self.cost(v, start_state) + v.euclidean_distance(dest_state) > max_cost:   
                        if v.parent is not None: v.parent.delete_child(v)
                        v.parent = None
                        orphans = v.children
                        v.children = []
                        tree_nodes.remove(v)
                        if v.euclidean_distance(start_state) + v.euclidean_distance(dest_state) < max_cost:
                            samples.add(v)

                        subtree  = set(c for c in orphans)
                        prev_size = -1
                        size = len(subtree)
                        while prev_size != size:
                            prev_size = size
                            temp = set()
                            for c in subtree:
                                for cc in c.children:
                                    temp.add(cc)
                            
                            for c in temp:
                                subtree.add(c)

                            size = len(subtree)
                        
                        for c in subtree:
                            tree_nodes.remove(c)
                            c.parent = None
                            c.children = []
                            if c.euclidean_distance(start_state) + c.euclidean_distance(dest_state) < max_cost:
                                samples.add(c)

                            
                            
                
                # repopulating samples
                for i in range(m):
                    samples.add(self.sample_state(max_cost, min_cost, center, C))

                for n in tree_nodes:
                    old_nodes.add(n)
                    q_v.add(n)
            
            # edge expansion
            qv_costs = {v: self.cost(v, start_state) + v.euclidean_distance(dest_state) for v in q_v}
            min_qv = np.inf
            if qv_costs:  
                min_qv = qv_costs[min(qv_costs, key=qv_costs.get)]


            qe_costs = {(v, w): self.cost(v, start_state) + v.euclidean_distance(w) + w.euclidean_distance(dest_state) for (v, w) in q_e}
            min_qe = np.inf
            if qe_costs != {}:
                min_qe = qe_costs[min(qe_costs, key=qe_costs.get)]

            while qv_costs and min_qv <= min_qe :
                
                v = min(qv_costs, key=qv_costs.get)
                q_v.remove(v)

                
                for x in self.find_near_states(samples, v, max_steering_radius):
                    if self.cost(v, start_state) + v.euclidean_distance(x) + x.euclidean_distance(dest_state) < max_cost:
                        q_e.add((v, x))

                if v not in old_nodes:
                    for w in self.find_near_states(tree_nodes, v, max_steering_radius):
                        if self.cost(v, start_state) + v.euclidean_distance(w) + w.euclidean_distance(dest_state) and v.parent != w and w not in v.children:
                            q_e.add((v, w))

                qv_costs = {v: self.cost(v, start_state) for v in q_v}
                qe_costs = {(v, w): self.cost(v, start_state) + v.euclidean_distance(w) for (v, w) in q_e}
                
            e = min(qe_costs, key=qe_costs.get)
            q_e.remove(e)
            v = e[0]
            x = e[1]

            #edge processing
            #TODO prioritize reaching get any solution before opimizing existing paths
            if self.cost(v, start_state) + v.euclidean_distance(x) + x.euclidean_distance(dest_state) < max_cost:
                if self.path_is_obstacle_free(v, x) and start_state.euclidean_distance(v) + v.euclidean_distance(x) + x.euclidean_distance(dest_state) < max_cost:
                    if x in tree_nodes:
                        if self.cost(v, start_state) + v.euclidean_distance(x) < self.cost(x, start_state):
                            if x.parent is not None: x.parent.delete_child(x)
                            v.children.append(x)
                            x.parent = v 
                    else:   
                            
                            samples.remove(x)
                            x.parent = v
                            v.children.append(x)
                            tree_nodes.add(x)
                            q_v.add(x)
                    
                    if plot_graphic:
                        cv2.circle(img, (x.x, x.y), 2, (0,0,0))
                        cv2.line(img, (v.x, v.y), (x.x, x.y), (255,0,0))

            else:
                q_e = set()
                q_v = set()

            #update best_plan and max cost
            if dest_state.parent is not None:
                best_plan = self._follow_parent_pointers(dest_state)
                max_cost = self.cost(dest_state, start_state)
            
            if plot_graphic:
                cv2.imshow('image', img)
                cv2.waitKey(10)

        if plot_graphic:
            draw_plan(img, best_plan, bgr=(0,0,255), thickness=2)
            for x in best_plan:
                print(x.x, x.y)
            cv2.waitKey(0)
        
        return best_plan
      

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: bit_star_planner.py occupancy_grid.pkl")
        sys.exit(1)
        
    pkl_file = open(sys.argv[1], 'rb')
    # world is a numpy array with dimensions (rows, cols, 3 color channels)
    world = pickle.load(pkl_file)
    pkl_file.close()

    bit_star = BITStarPlanner(world)

    start_state = State(10, 10, None)
    dest_state = State(500, 150, None)

    max_num_steps = 1000     # max number of nodes to be added to the tree
    max_steering_radius = 30 # pixels
    dest_reached_radius = 50 # pixels
    plan = bit_star.plan(start_state,
                    dest_state,
                    max_num_steps,
                    max_steering_radius,
                    dest_reached_radius,
                    True)
