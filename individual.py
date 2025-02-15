from typing import List
class Individual:
    def __init__(self, vec: List[bool], n: int):
        self.x = vec
        assert len(vec) == n
        self.n = n

    def __iter__(self):
        """Make the class iterable over its bits."""
        for i in range(self.n):
            yield self.x[i]

    def __len__(self):
        """Make len(obj) work."""
        return self.n
    
    def __getitem__(self, key):
        """
        Enable indexing and slicing.
        - If key is int, return a single bool.
        - If key is slice, return a new Individual for that slice.
        """
        if isinstance(key, slice):
            # Extract the slice as a sub-list of booleans
            sub_vec = self.x[key]
            # Create a new Individual with the sliced vector
            return Individual(sub_vec, len(sub_vec))
        elif isinstance(key, int):
            return self.x[key]
        else:
            raise TypeError("Invalid argument type for indexing: must be int or slice.")
        
    
    def __str__(self):
        return f"Individual(vec={self.x}, n={self.n})"
    
    def __repr__(self):
        return self.__str__()
    
    def __eq__(self, other):
        if not isinstance(other, Individual):
            return NotImplemented
        # Two individuals are equal if they have the same length and same bits.
        return self.n == other.n and self.x == other.x

    def __hash__(self):
        """
        Convert the boolean list to an integer.
        For example, [True, False, True] becomes the integer 5 (binary 101).
        """
        value = 0
        for bit in self.x:
            # Shift the current value left by one and add the bit (converted to int: True -> 1, False -> 0)
            value = (value << 1) | int(bit)
        return hash(value)