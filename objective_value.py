from individual import Individual
from typing import Iterable
from abc import ABC, abstractmethod


class ObjectiveValue:
    _value: Iterable
    def __init__(self, m: int, x:Individual, val: Iterable):
        """
        Associates an f(x) to each x
        f should be callable function of type Indiviual -> List[Float]
        """
        self.m = m
        self.x = x
        self._value = val
        assert(len(self._value) == m)
    
    def value(self):
        return self._value

    def fk(self, k) -> float:
        if k > self.m-1:
            raise ValueError("K shouldn't be greater or equal to m") 
        return self._value[k]
    
    def __iter__(self):
        # Simply yield each element from self.items
        for i in range(self.m):
            item = self.fk(i)
            yield item

    def __len__(self):
        return self.m
    
    def __str__(self):
        # Create a nicely formatted string including the list of floats
        return f"ObjectiveValue(value={self.value()})"
    
    def __repr__(self):
        # This ensures that printing a list of ObjectiveValue objects uses the __str__ representation.
        return self.__str__()
    def __getitem__(self, key):
        # Allow indexing by simply deferring to fk
        return self.fk(key)

    @staticmethod
    def strictly_dominates(u: "ObjectiveValue", v: "ObjectiveValue"):
        """
        Returns True if u strictly dominates v
        """
        if u.m != v.m:
            raise ValueError("U and V should have the same m so as to calculate the strictly dominance")
        
        for i in range(u.m):
            if not u.fk(i) > v.fk(i):
                return False
        return True
        
    @staticmethod
    def weakly_dominates(u: "ObjectiveValue", v: "ObjectiveValue"):
        if u.m != v.m:
            raise ValueError("U and V should have the same m so as to calculate the weakly dominance")
        for i in range(u.m):
            if not u.fk(i) >= v.fk(i):
                return False
        return True


class ObjectiveValueConstructor(ABC):
    """
    Acts as a function x -> f(x), constructing an instance of Objective Value
    """
    def __init__(self, m):
        self.m = m
    
    @abstractmethod
    def create_objective_value(self, x:Individual)->ObjectiveValue:
        pass
    
    def __call__(self, x: Individual) -> ObjectiveValue:
        return self.create_objective_value(x)


class ObjectiveValueConstructorFromFunction(ObjectiveValueConstructor):
    def __init__(self, m, f):
        self.f = f
        super().__init__(m)
    
    def create_objective_value(self, x:Individual)->ObjectiveValue:
        it = self.f(x)
        return ObjectiveValue(self.m, x, it)
    

    
