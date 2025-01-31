from typing import List
class Individual:
    
    def __init__(self, vec: List[bool], n: int):
        self.x = vec
        assert(len(vec) == n)
        self.n = n
    
    def __iter__(self):
        # Simply yield each element from self.items
        for i in range(self.n):
            item = self.x[i]
            yield item

    def __len__(self):
        return self.n