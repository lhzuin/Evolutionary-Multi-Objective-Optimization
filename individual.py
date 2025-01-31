from typing import List
class Individual:
    def __init__(self, vec: List[bool], n: int):
        # A list of booleans and its length
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
            # Return a single element
            return self.x[key]
        else:
            raise TypeError("Invalid argument type for indexing: must be int or slice.")
        
    
    def __str__(self):
        # Return a readable string describing this instance
        # You can customize it however you want
        return f"Individual(vec={self.x}, n={self.n})"