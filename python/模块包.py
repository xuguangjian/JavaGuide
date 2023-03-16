import itertools

x=[1,2,3]
y=[4,5,6]
z="fishy"
zipped=itertools.zip_longest(x,y,z)
list(zipped)
list(map(max,[1,2,3],[3,4,4],[4,3,343]))