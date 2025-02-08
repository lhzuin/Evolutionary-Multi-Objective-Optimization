from individual import Individual
from objective_value import ObjectiveValue, ObjectiveValueConstructor
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
    

class mLOTZ:
    def __init__(self, m: int, n: int, x:Individual):
        self.n = n
        assert(n == len(x))
        self.m = m
        self.x = x
    
    def __getitem__(self, key): 
        """
        Divides x into m/2 chunks of length n'=2n/m and accesses the int(key+1/2)th chunk to return his LOTZ
        """
        if key >= self.m or key<0:
            ValueError("Unexpected k value for mLOTZ")
        n_prime = int(2*self.n/self.m)
        if key%2 == 1:
            start = int(n_prime*(key - 1)/2 + 1) - 1
            stop = int(1 + n_prime*(key + 1)/2) - 1
        else:
            start = int(n_prime*(key/2-1)+1) - 1
            stop = int( 1+n_prime*(key)/2) - 1
        lotz = LOTZ(self.x[start:stop])
        return lotz[2-(key%2)]
    
    def __iter__(self):
        """Make the class iterable over its bits."""
        for i in range(self.m):
            yield self.__getitem__(i)

    def __len__(self):
        """Make len(obj) work."""
        return self.m
    


class mLOTZConstructor(ObjectiveValueConstructor):
    def __init__(self, m: int, n: int):
        assert(m%2 == 0)
        self.n = n
        assert(self.n%(m/2)==0)
        self.m = m

    def create_objective_value(self, x:Individual)->ObjectiveValue:
        mlotz = mLOTZ(self.m, self.n, x)
        return ObjectiveValue(self.m, x, list(mlotz))


    

