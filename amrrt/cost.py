#Copyright (c) 2020 Ocado. All Rights Reserved.

import numpy as np


class Cost:
    def __init__(self, value, blocked=False):
        self.value = value
        self.blocked = blocked

    def to_float(self):
        if self.blocked : return np.inf
        return self.value

    def __add__(self, other):
        if isinstance(other, Cost):
            raise ValueError
        return Cost(self.value + other, self.blocked)

    def __sub__(self, other):
        if isinstance(other, Cost):
            raise ValueError
        return Cost(self.value - other, self.blocked)

    def __lt__(self, other):
        if isinstance(other, Cost):
            if self.blocked != other.blocked : return not self.blocked
            return self.value < other.value
        if self.blocked : return np.inf < other
        return self.value < other

    def __le__(self, other):
        if isinstance(other, Cost):
            if self.blocked != other.blocked : return not self.blocked
            return self.value <= other.value
        if self.blocked : return np.inf <= other
        return self.value <= other

    def __eq__(self, other):
        if isinstance(other, Cost):
            return self.blocked == other.blocked and self.value == other.value
        if self.blocked : return np.inf == other
        return self.value == other

    def __ne__(self, other):
        if isinstance(other, Cost):
            return self.blocked != other.blocked or self.value != other.value
        if self.blocked : return np.inf != other
        return self.value != other

    def __gt__(self, other):
        if isinstance(other, Cost):
            if self.blocked != other.blocked : return self.blocked
            return self.value > other.value
        if self.blocked : return np.inf > other
        return self.value > other

    def __ge__(self, other):
        if isinstance(other, Cost):
            if self.blocked != other.blocked : return self.blocked
            return self.value >= other.value
        if self.blocked : return np.inf >= other
        return self.value >= other
