from individual import Individual
from objective_value import ObjectiveValue, ObjectiveValueConstructor
from lotz import LOTZ

class mLOTZ:
    def __init__(self, m: int, n: int, x: Individual):
        # m must be even; x must be of length n.
        self.n = n
        assert n == len(x)
        self.m = m
        # For convenience, store the underlying bit list.
        self.x = x.x

    def __getitem__(self, key):
        if key < 0 or key >= self.m:
            raise ValueError("Unexpected k value for mLOTZ")
        # Each chunk has length n_prime = 2n/m.
        n_prime = int(2 * self.n / self.m)
        # Determine the chunk index: since there are m/2 chunks, keys 0 and 1 use chunk 0, keys 2 and 3 use chunk 1, etc.
        chunk_index = key // 2
        start = chunk_index * n_prime
        stop = (chunk_index + 1) * n_prime
        # Create an Individual for the chunk.
        chunk_ind = Individual(self.x[start:stop], stop - start)
        lotz = LOTZ(chunk_ind)
        # According to the assignment, for 1-indexed k:
        #   if k is odd (i.e. key 0, 2, 4, … in 0-indexed), return LOTZ1 (leading ones),
        #   if k is even (i.e. key 1, 3, 5, …), return LOTZ2 (trailing zeros).
        if key % 2 == 0:
            return lotz.calculate_leading_ones()
        else:
            return lotz.calculate_trailing_zeros()

    def __iter__(self):
        for i in range(self.m):
            yield self.__getitem__(i)

    def __len__(self):
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