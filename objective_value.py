from individual import Individual


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
    
    def __str__(self):
        # Create a nicely formatted string including the list of floats
        return f"ObjectiveValue(value={self.value()})"
    
    def __repr__(self):
        # This ensures that printing a list of ObjectiveValue objects uses the __str__ representation.
        return self.__str__()

    @staticmethod
    def strictly_dominates(u: "ObjectiveValue", v: "ObjectiveValue"):
        """
        Returns True if u strictly dominates v
        """
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
        return ObjectiveValue(self.f, self.m, x)
    def __call__(self, x: Individual) -> ObjectiveValue:
        return self.create_objective_value(x)

