import os


def test_load():
    from floky import L2
    lsh = L2(2, 3, 2)
    lsh.describe()
    os.remove("lsh.db3")


