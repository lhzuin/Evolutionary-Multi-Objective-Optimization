from individual import Individual

class LOTZ:
    def __init__(self, x:Individual):
        self.x = x
    
    def calculate_leading_ones(self):
        counter = 0
        for xk in self.x:
            if xk != 1:
                return counter
            counter += 1
        return counter

    def calculate_trailing_zeros(self):
        counter = 0
        for i in range(len(self.x)-1, -1, -1):
            xk = self.x[i]
            if xk != 0:
                return counter
            counter += 1
        return counter
    
    def __getitem__(self, key): 
        if key == 1:
            return self.calculate_leading_ones()
        elif key == 2:
            return self.calculate_trailing_zeros()
        raise ValueError("Unexpected k value for LOTZ")
    
    def __iter__(self):
        """Make the class iterable over its bits."""
        for i in range(2):
            yield self.__getitem__(i)

    def __len__(self):
        """Make len(obj) work."""
        return 2