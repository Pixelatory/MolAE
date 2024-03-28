
def test(x, y, z):
    return x, y, z

t = lambda fn, x: fn(x, x, x)

print(t(test, 5))
