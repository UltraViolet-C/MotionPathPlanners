#!/usr/bin/python
import sys
import pickle
import numpy as np
import random
import cv2

from plotting_utils import draw_plan
from state import State

class RRTPlanner:
    """
    Applies the RRT algorithm on a given grid world
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


    def sample_state(self):
        """
        Sample a new state uniformly randomly on the image.
        """
        #TODO: make sure you're not exceeding the row and columns bounds
        # x must be in {0, cols-1} and y must be in {0, rows -1}
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

        #TODO: populate x and y properly according to the description above.
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
            # TODO: check if the inteprolated state that is float(i)/max_checks * dist(s_from, s_new)
            # away on the line from s_from to s_new is free or not. If not free return False
            t = float(i)/max_checks # t = d_t/d = (float(i)/max_checks) * d / d
            interpolated_x = int((1-t)*s_from.x + t*s_to.x)
            interpolated_y = int((1-t)*s_from.y + t*s_to.y)
            interpolated_state = State(interpolated_x, interpolated_y, None)

            if not (self.state_is_free(interpolated_state)):
                return False

        # Otherwise the line is free, so return true
        return True


    def plan(self, start_state, dest_state, max_num_steps, max_steering_radius, dest_reached_radius):
        """
        Returns a path as a sequence of states [start_state, ..., dest_state]
        if dest_state is reachable from start_state. Otherwise returns [start_state].
        Assume both source and destination are in free space.
        """
        assert (self.state_is_free(start_state))
        assert (self.state_is_free(dest_state))

        # The set containing the nodes of the tree
        tree_nodes = set()
        tree_nodes.add(start_state)

        # image to be used to display the tree
        img = np.copy(self.world)

        plan = [start_state]
        total_nodes_visited = 0
        path_distance = 0

        for step in range(max_num_steps):

            # TODO: Use the methods of this class as in the slides to
            # compute s_new
            s_rand = self.sample_state()
            s_nearest = self.find_closest_state(tree_nodes, s_rand)
            s_new = self.steer_towards(s_nearest, s_rand, max_steering_radius)

            if self.path_is_obstacle_free(s_nearest, s_new):
                tree_nodes.add(s_new)
                s_nearest.children.append(s_new)
                total_nodes_visited += 1

                # If we approach the destination within a few pixels
                # we're done. Return the path.
                if s_new.euclidean_distance(dest_state) < dest_reached_radius:
                    dest_state.parent = s_new
                    plan = self._follow_parent_pointers(dest_state)
                    break

                # plot the new node and edge
                cv2.circle(img, (s_new.x, s_new.y), 2, (0,0,0))
                cv2.line(img, (s_nearest.x, s_nearest.y), (s_new.x, s_new.y), (255,0,0))

            # Keep showing the image for a bit even
            # if we don't add a new node and edge
            cv2.imshow('image', img)
            cv2.waitKey(10)

        draw_plan(img, plan, bgr=(0,0,255), thickness=2)
        cv2.waitKey(0)
        # NOTE: uncomment for plotting capabilities
        # path_distance = 0
        # for i in range(len(plan) - 1):
        #     path_distance += plan[i].euclidean_distance(plan[i + 1])
        # if path_distance == 0:
        #     return (total_nodes_visited, np.inf)
        # return (total_nodes_visited, path_distance)
        return [start_state]



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: rrt_planner.py occupancy_grid.pkl")
        sys.exit(1)

    pkl_file = open(sys.argv[1], 'rb')
    # world is a numpy array with dimensions (rows, cols, 3 color channels)
    world = pickle.load(pkl_file)
    pkl_file.close()

    rrt = RRTPlanner(world)

    start_state = State(10, 10, None)
    dest_state = State(500, 150, None)

    max_num_steps = 1000     # max number of nodes to be added to the tree
    max_steering_radius = 30 # pixels
    dest_reached_radius = 50 # pixels
    plan = rrt.plan(start_state,
                    dest_state,
                    max_num_steps,
                    max_steering_radius,
                    dest_reached_radius)
    
    # NOTE: Uncomment the below code to plot the opt_path vs number of nodes
    # data = [] # array of tuples
    # opt_path_length = np.inf

    # for steps in range(max_num_steps):
    #     path = rrt.plan(start_state, dest_state, steps, max_steering_radius, dest_reached_radius)
    #     if path[1] < opt_path_length:
    #         opt_path_length = path[1]
    #         data.append((steps, path[1]))
    #     else:
    #         data.append((steps, opt_path_length))

    # x_data = [x[0] for x in data]
    # y_data = [x[1] for x in data]

    # # plot our data array
    # plt.title("length of optimal path vs number of nodes visited")
    # plt.xlabel("number of nodes visited")
    # plt.ylabel("length of optimal path")
    # plt.plot(x_data, y_data)
    # plt.show()