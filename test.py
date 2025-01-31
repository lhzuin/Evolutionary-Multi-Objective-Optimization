from individual import Individual
from mlotz import LOTZ, mLOTZ

x = Individual([1, 1, 0,  0, 1, 0, 0, 0], 8)
lotz = LOTZ(x)
print(lotz[1])
print(lotz[2])

mlotz = mLOTZ(4, x)
print(mlotz[1])
print(mlotz[2])
print(mlotz[3])
print(mlotz[4])


"""
l = [1, 1, 0,  0, 1, 0, 0, 0]
print(l[1:3])


k = 2
print(int((k+1)/2))
k = 3
print(int((k+1)/2))

"""