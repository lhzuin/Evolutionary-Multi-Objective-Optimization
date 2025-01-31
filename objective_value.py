from individual import Individual
from typing import List


class ObjectiveValue:
    def __init__(self, f, m: int, x:Individual):
        """
        f should be callable function of type Indiviual -> List[Float]
        """
        self.f = f
        self.m = m
        self.x = x
        self._value = self.f(x)
        assert(len(self._value) == m)
    
    def value(self):
        return self._value

    def fk(self, k) -> float:
        if k > self.m-1:
            raise ValueError("Fizemos merda em fk") 
        return self._value[k]
    
    def __iter__(self):
        # Simply yield each element from self.items
        for i in range(self.m):
            item = self.fk(i)
            yield item

    def __len__(self):
        return self.m

    @staticmethod
    def strongly_dominates(u: "ObjectiveValue", v: "ObjectiveValue"):
        if u.m != v.m:
            raise ValueError("Fizemos merda em strongly")
        
        for i in range(u.m):
            if not u.fk(i) > v.fk(i):
                return False
        return True
        
    @staticmethod
    def weakly_dominates(u: "ObjectiveValue", v: "ObjectiveValue"):
        if u.m != v.m:
            raise ValueError("Fizemos weakly merda")
        for i in range(u.m):
            if not u.fk(i) >= v.fk(i):
                return False
        return True

    
class ObjectiveValueConstructor:
    def __init__(self, f, m):
        self.f = f
        self.m = m
    def create_objective_value(self, x:Individual)->ObjectiveValue:
        return ObjectiveValue(self.f, self.m)

