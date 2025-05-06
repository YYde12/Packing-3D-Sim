from operator import pos
import queue
from copy import deepcopy
import math
from shapely.geometry import MultiPoint, Point

# 使用排序数组实现的优先队列
# class PriorityQueue(object):

#     def __init__(self, node):
#         self._queue = sortedcontainers.SortedList([node])

#     def push(self, node):
#         self._queue.add(node)
#         # # 控制优先队列的长度
#         # if len(self._queue) > self.max_len:
#         #     self._queue.pop(index=len(self._queue)-1)

#     def pop(self):
#         return self._queue.pop(index=0)

#     def empty(self):
#         return len(self._queue) == 0

def dist(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1-x2) * (x1-x2) + (y1-y2) * (y1-y2) + (z1-z2) * (z1-z2))

class Position:

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def set(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, rhs):
        return Position(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)

    def __sub__(self, rhs):
        return Position(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    
    def __truediv__(self, rhs):
        return Position(self.x / rhs, self.y / rhs, self.z / rhs)

    def __floordiv__(self, rhs):
        return Position(self.x // rhs, self.y // rhs, self.z // rhs)
    
    def __repr__(self):
        return "<Position: x={}  y={}  z={}>" \
                .format(self.x, self.y, self.z)


class Attitude:

    def __init__(self, roll=0, pitch=0, yaw=0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
    
    def set(self, roll, pitch, yaw):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
    
    def __repr__(self):
        return "<Attitude: roll={}  pitch={}  yaw={}>" \
                .format(self.roll, self.pitch, self.yaw)


class Transform:

    def __init__(self, position = Position(), attitude = Attitude()):
        self.position = position
        self.attitude = attitude

    def __repr__(self):
        return "<Transform:\n  {}\n  {}\n>".format(self.position, self.attitude)


class AttitudeStability:

    def __init__(self, attitude = Attitude(), stability = 0):
        self.attitude = attitude
        self.stability = stability
    
    def __lt__(self, other):
        return self.stability > other.stability

    def __eq__(self, other):
        return self.stability == other.stability
    
    def __repr__(self):
        return "<AttitudeStability:\n  {}\n  stability={}\n>"\
                .format(self.attitude, self.stability)

class TransformScore:
    
    def __init__(self, transform = Transform(), score = 0):
        self.transform = transform
        self.score = score
    
    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score
    
    def __repr__(self):
        return "<TransformScore:\n  <Transform:\n    {}\n    {}\n  >\n  score={}\n>" \
                .format(self.transform.position, self.transform.attitude, self.score)

class StabilityChecker:
    """
    The center-of-mass projection method is used to determine the static stability of an object when placed in a container.
    """

    def __init__(self, container, threshold=0.2):
        self.container = container
        self.threshold = threshold

    def is_statically_stable(self, item, transform) -> bool:
        """
        Determine whether the item is statically stable under the given transform.
    
        :param item: The Item instance to check
        :param transform: A Transform object containing position and orientation
        :return: True if statically stable, False otherwise
        """
        # Apply the transform to the item
        item.transform(transform)

         # If the item is in contact with the ground, consider it stable
        if item.position.z == 0:
            return True

        # Compute the bounding coordinates of the bottom surface grid points
        x0, y0 = item.position.x, item.position.y
        dx, dy = item.curr_geometry.x_size, item.curr_geometry.y_size

        # Collect the center points of all supporting grid cells
        support_pts = []
        for i in range(dx):
            for j in range(dy):
                # Boundary check to avoid IndexError
                if not (0 <= x0 + i < self.container.heightmap.shape[0] and
                        0 <= y0 + j < self.container.heightmap.shape[1]):
                    continue
                # Support condition: 0 <= item.position.z - self.container.heightmap[x0 + i][y0 + j] <= 1
                if 0 <= item.position.z - self.container.heightmap[x0 + i][y0 + j] <= 1:
                    support_pts.append((x0 + i + 0.5, y0 + j + 0.5))

         # If there are no supporting points, the item is not stable
        if not support_pts:
            return False
        
        # Compute the projection of the center of mass 
        com_x = x0 + dx / 2
        com_y = y0 + dy / 2

        # If the support points form a single point
        if len(support_pts) == 1:
            return Point(com_x, com_y).distance(Point(support_pts[0])) < self.threshold
        # Compute the convex hull of support points
        hull = MultiPoint(support_pts).convex_hull
        # If the hull is a line segment, check if the center of mass projection lies on or near the line
        if hull.geom_type == 'LineString':
            return Point(com_x, com_y).distance(hull) < self.threshold

        return hull.contains(Point(com_x, com_y))
