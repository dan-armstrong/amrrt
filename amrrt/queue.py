#Copyright (c) 2020 Ocado. All Rights Reserved.

class UpdatableQueue:
    """
    Priority queue in which members can have their priority updated
    """
    def __init__(self):
        self.heap = []
        self.node_map = {}

    def push(self, item, priority):
        """
        Add new item to the queue with given priority
        """
        node = HeapNode(item, priority, len(self.heap))
        self.heap.append(node)
        self.node_map[item] = node
        return self._bubble_up(node.index)

    def pop(self, i=0):
        """
        Remove and return i'th item in queue
        """
        data = self.heap[i].data
        self.heap[i] = self.heap[len(self.heap)-1]
        self.heap[i].index = i
        del self.heap[len(self.heap)-1]
        del self.node_map[data]
        self._bubble_down(i)
        return data

    def update(self, item, priority):
        """
        Update item to given priority
        """
        i = self.node_map[item].index
        self.heap[i].priority = priority
        self._bubble_up(i)
        self._bubble_down(i)

    def contains(self, item):
        """
        Checks if given item is in queue
        """
        return item in self.node_map

    def empty(self):
        """
        Checks if queue is empty
        """
        return len(self.heap) == 0

    def _bubble_up(self, i):
        """
        Bubble i'th item upwards in heap to correct position
        """
        while i > 0 and self.heap[i] < self.heap[(i+1)//2-1]:
            self.heap[i], self.heap[(i+1)//2-1] = self.heap[(i+1)//2-1], self.heap[i]
            self.heap[i].index = i
            self.heap[(i+1)//2-1].index = (i+1)//2-1
            i = (i+1)//2-1

    def _bubble_down(self, i):
        """
        Bubble i'th item downwards in heap to correct position
        """
        ci = self._get_smallest_child(i)
        while ci != None and self.heap[i] > self.heap[ci]:
            self.heap[i], self.heap[ci] = self.heap[ci], self.heap[i]
            self.heap[i].index = i
            self.heap[ci].index = ci
            i = ci
            ci = self._get_smallest_child(i)

    def _get_smallest_child(self, i):
        """
        Return the child of i with least priority
        """
        l = 2*i + 1
        r = 2*i + 2
        if l > len(self.heap)-1:
            return None
        if l == len(self.heap)-1 or self.heap[l] <= self.heap[r]:
            return l
        return r


class HeapNode:
    """
    Node in heap that keeps track of priority and current heap index
    """
    def __init__(self, data, priority, index):
        self.data = data
        self.priority = priority
        self.index = index

    def __lt__(self, other):
        return self.priority < other.priority

    def __le__(self, other):
        return self.priority <= other.priority

    def __eq__(self, other):
        return self.priority == other.priority

    def __ne__(self, other):
        return self.priority != other.priority

    def __gt__(self, other):
        return self.priority > other.priority

    def __ge__(self, other):
        return self.priority >= other.priority
