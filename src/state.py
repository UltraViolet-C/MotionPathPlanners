#!/usr/bin/python
from math import sqrt

class State:
    """
    2D state.
    """

    def __init__(self, x, y, parent):
        """
        x represents the columns on the image and y represents the rows,
        Both are presumed to be integers
        """
        self.x = x
        self.y = y
        self.parent = parent
        self.children = []


    def __eq__(self, state):
        """
        When are two states equal?
        """
        return state and self.x == state.x and self.y == state.y

    def __hash__(self):
        """
        The hash function for this object. This is necessary to have when we
        want to use State objects as keys in dictionaries
        """
        return hash((self.x, self.y))

    def euclidean_distance(self, state):
        assert (state)
        return sqrt((state.x - self.x)**2 + (state.y - self.y)**2)
