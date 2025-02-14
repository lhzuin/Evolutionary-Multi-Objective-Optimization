from typing import List, Tuple
from individual import Individual
class BinaryHeap:
    def __init__(self):
        """
        Initialize an empty min-heap.
        
        The heap is represented as a list of tuples (priority, uid, item), where 'uid' is a unique 
        identifier (an integer) assigned to each inserted item. A separate dictionary 'position' maps each uid
        to its index in the heap list.
        """
        self.heap = []  # list of (priority, uid, item)
        self.position = {}  # maps uid to index in self.heap
        self._counter = 0   # unique counter to generate uids

    def __len__(self):
        """Return the number of elements in the heap."""
        return len(self.heap)

    def is_empty(self):
        """Check if the heap is empty."""
        return len(self.heap) == 0

    def _swap(self, i, j):
        """Swap elements at indices i and j and update their positions."""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]
        uid_i = self.heap[i][1]
        uid_j = self.heap[j][1]
        self.position[uid_i] = i
        self.position[uid_j] = j

    def _sift_up(self, index):
        """Bubble up the element at index 'index' until the heap property is restored."""
        while index > 0:
            parent = (index - 1) // 2
            if self.heap[parent][0] > self.heap[index][0]:
                self._swap(parent, index)
                index = parent
            else:
                break

    def _sift_down(self, index):
        """Push down the element at index 'index' until the heap property is restored."""
        n = len(self.heap)
        while True:
            left = 2 * index + 1
            right = 2 * index + 2
            smallest = index

            if left < n and self.heap[left][0] < self.heap[smallest][0]:
                smallest = left
            if right < n and self.heap[right][0] < self.heap[smallest][0]:
                smallest = right

            if smallest != index:
                self._swap(index, smallest)
                index = smallest
            else:
                break

    def insert(self, item: Individual, priority: float) -> int:
        """
        Insert a new item with the given priority.
        
        Parameters:
            item: The item to be inserted (e.g. an Individual).
            priority: The associated priority (a number).
        
        Returns:
            The unique identifier (uid) assigned to this insertion.
        
        Time Complexity: O(log n)
        """
        uid = self._counter
        self._counter += 1
        entry = (priority, uid, item)
        index = len(self.heap)
        self.heap.append(entry)
        self.position[uid] = index
        self._sift_up(index)
        return uid

    def find_min(self) -> Tuple[int, any, float]:
        """
        Return the minimum element in the heap without removing it.
        
        Returns:
            A tuple (uid, item, priority) corresponding to the element with the smallest priority.
        
        Time Complexity: Θ(1)
        """
        if not self.heap:
            raise IndexError("find_min from empty heap")
        priority, uid, item = self.heap[0]
        return uid, item, priority

    def extract_min(self) -> Tuple[int, any, float]:
        """
        Remove and return the element with the smallest priority.
        
        Returns:
            A tuple (uid, item, priority) corresponding to the element with the smallest priority.
        
        Time Complexity: O(log n)
        """
        if not self.heap:
            raise IndexError("extract_min from empty heap")
        min_priority, min_uid, min_item = self.heap[0]
        last_elem = self.heap.pop()
        del self.position[min_uid]
        if self.heap:
            self.heap[0] = last_elem
            self.position[last_elem[1]] = 0
            self._sift_down(0)
        return min_uid, min_item, min_priority

    def update_key(self, uid: int, new_priority: float):
        """
        Update the priority (key) of an element identified by its uid to new_priority.
        This method adjusts the element's position in the heap whether the new priority is lower or higher.
        
        Parameters:
            uid (int): The unique identifier of the element whose key is to be updated.
            new_priority (float): The new priority value.
        
        Time Complexity: O(log n)
        
        Raises:
            KeyError: If the uid is not found.
        """
        if uid not in self.position:
            raise KeyError("Item with given uid not found in the heap")
        index = self.position[uid]
        current_priority, _, item = self.heap[index]
        self.heap[index] = (new_priority, uid, item)
        if new_priority < current_priority:
            self._sift_up(index)
        else:
            self._sift_down(index)

    def decrease_key(self, uid: int, new_priority: float):
        """
        Decrease the priority of the element identified by uid to new_priority.
        If new_priority is not lower than the current priority, a ValueError is raised.
        
        Parameters:
            uid (int): The unique identifier of the element.
            new_priority (float): The new, lower priority value.
        
        Time Complexity: O(log n)
        
        Raises:
            KeyError: If the uid is not found.
            ValueError: If new_priority is not lower than the current priority.
        """
        if uid not in self.position:
            raise KeyError("Item with given uid not found in the heap")
        index = self.position[uid]
        current_priority, _, _ = self.heap[index]
        if new_priority > current_priority:
            raise ValueError("New priority is greater than current priority")
        self.heap[index] = (new_priority, uid, self.heap[index][2])
        self._sift_up(index)

    def meld(self, other_heap: 'BinaryHeap'):
        """
        Meld (or merge) another heap into this heap.
        Time Complexity: O(n)
        """
        # Simple solution: just extend our heap and rebuild.
        for entry in other_heap.heap:
            _, uid, item = entry
            self.heap.append(entry)
            self.position[uid] = len(self.heap) - 1
        # Rebuild the heap in O(n) time.
        n = len(self.heap)
        for i in range((n // 2) - 1, -1, -1):
            self._sift_down(i)

    @classmethod
    def make_heap(cls, items: List[Tuple[float, any]]) -> 'BinaryHeap':
        """
        Build a heap from a list of tuples (priority, item).
        Time Complexity: O(n)
        Duplicate items are allowed.
        
        Returns:
            A BinaryHeap instance.
        """
        heap_instance = cls()
        for (priority, item) in items:
            heap_instance.insert(item, priority)
        return heap_instance