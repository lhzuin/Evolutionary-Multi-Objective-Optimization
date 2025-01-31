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
    

class mLOTZ:
    def __init__(self, m: int, x:Individual):
        assert(m%2 == 0)
        self.n = len(x)
        assert(self.n%(m/2)==0)
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
            lotz = LOTZ(self.x[start:stop])
            return lotz[1]
        
        start = int(n_prime*(key/2-1)+1) - 1
        stop = int( 1+n_prime*(key)/2) - 1
        lotz = LOTZ(self.x[start:stop])
        return lotz[2]


    


    
    

