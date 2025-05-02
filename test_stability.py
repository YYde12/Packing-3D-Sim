import pytest
import numpy as np
from utils import StabilityChecker, Transform, Position, Attitude
from container import Container
from object import Item, Geometry

# Dummy container that bypasses full geometry but supports heightmap access
class DummyContainer(Container):
    def __init__(self, heightmap):
        self.heightmap = np.array(heightmap)
        z = np.max(self.heightmap) + 1  # Ensure enough vertical space
        x, y = self.heightmap.shape
        self.boxSize = (z, x, y)
        self.geometry = Geometry(np.zeros((z, x, y)))
        self.number = 0  # Required by StabilityChecker

# Minimal item class with custom size and position
class DummyItem(Item):
    def __init__(self, size, position):
        dx, dy, dz = size
        cube = np.ones((dz, dx, dy))  # (z, x, y)
        super().__init__(cube)
        self.position = Position(*position)
        self.calc_heightmap()

# Transform stub that supplies position only
class DummyTransform(Transform):
    def __init__(self, x, y, z):
        super().__init__(position=Position(x, y, z), attitude=Attitude())

# Parametrized test cases
@pytest.mark.parametrize("heightmap,item_pos,item_size,expect", [
    ([[0,0],[0,0]], (0,0,0), (2,2,1), True),
    ([[1,0],[0,0]], (0,0,1), (1,1,1), True),
    ([[1,0],[0,0]], (0,1,1), (1,1,1), False),
    ([[1,1,0],[0,0,0]], (0,0,1), (2,1,1), True),
    ([[0,0],[0,0]], (0,0,1), (1,1,1), True),
    ([[1,1,1],[1,1,1],[0,0,0],[0,0,0],[0,0,0]], (1,0,1), (1,4,1), False)
])
def test_static_stability(heightmap, item_pos, item_size, expect):
    cont = DummyContainer(heightmap)
    checker = StabilityChecker(cont)
    it = DummyItem(item_size, item_pos)
    tf = DummyTransform(*item_pos)
    assert checker.is_statically_stable(it, tf) == expect
