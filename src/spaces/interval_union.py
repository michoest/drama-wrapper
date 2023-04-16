import numpy as np

from src.spaces.continuous import ContinuousActionSpace


class Node(object):
    def __init__(self, x: float = None, y: float = None):
      self.x = x
      self.y = y
      self.l = None
      self.r = None
      self.h = 1

    def __str__(self):
      return f'<Node ({self.x},{self.y}), left: {self.l}, right: {self.r}>'

    def __repr__(self):
      return self.__str__()

class IntervalUnionTree(ContinuousActionSpace):
  def contains(self, root, x):
    if not root:
      return False
    elif root.x <= x and root.y >= x:
      return True
    elif root.x > x:
      return self.contains(root.l, x)
    else:
      return self.contains(root.r, x)

  def nearest_elements(self, root, x):
    if x > root.y:
      return self._nearest_elements(root.r, x, x - root.y, root.y)
    elif x < root.x:
      return self._nearest_elements(root.l, x, root.x - x, root.x)
    else:
      return x

  def _nearest_elements(self, root, x, min_diff, min_value):
    if not root:
      return [min_value]
    elif x > root.y:
      distance = x - root.y
      return [min_value, root.y] if distance == min_diff else [min_value] if distance > min_diff else self._nearest_elements(root.r, x, distance, root.y)
    elif x < root.x:
      distance = root.x - x
      return [min_value, root.x] if distance == min_diff else [min_value] if distance > min_diff else self._nearest_elements(root.l, x, distance, root.x)
    else:
       return x

  def nearest_element(self, root, x):
    return self.nearest_elements(root, x)[-1]

  def last_interval_before_or_within(self, root, x):
    if x >= root.x and x <= root.y:
      return (root.x, root.y), True
    elif x < root.x:
      return self.last_interval_before_or_within(root.l, x) if root.l is not None else ((root.x, root.y), False)
    else:
      return self.last_interval_before_or_within(root.r, x) if root.r is not None else ((root.x, root.y), False) if x < root.y else ((None, None), False)

  def first_interval_after_or_within(self, root, x):
    if x >= root.x and x <= root.y:
      return (root.x, root.y), True
    elif x > root.y:
      return self.first_interval_after_or_within(root.r, x) if root.r is not None else ((root.x, root.y), False)
    else:
      return self.first_interval_after_or_within(root.l, x) if root.l is not None else ((root.x, root.y), False) if x > root.x else ((None, None), False)

  def smallest_interval(self, root):
    if root is None or root.l is None:
        return root
    else:
        return self.smallest_interval(root.l)

  def insert(self, root, x, y):
    if not root:
      return Node(x, y)
    elif y < root.x:
      root.l = self.insert(root.l, x, y)
    elif x > root.y:
      root.r = self.insert(root.r, x, y)
    else:
      root.x = min(root.x, x)
      root.y = max(root.y, y)

    root.h = 1 + max(self.getHeight(root.l),
                     self.getHeight(root.r))

    b = self.getBal(root)

    if b > 1 and y < root.l.x:
      return self.rRotate(root)

    if b < -1 and x > root.r.y:
      return self.lRotate(root)

    if b > 1 and x > root.l.y:
      root.l = self.lRotate(root.l)
      return self.rRotate(root)

    if b < -1 and y < root.r.x:
      root.r = self.rRotate(root.r)
      return self.lRotate(root)

    return root

  def remove(self, root, x, y):
    if not root:
      return None
    elif x > root.x and y < root.y:
      old_maximum = root.y
      root.y = x
      self.insert(root, y, old_maximum)
    elif x < root.x and y < root.y and y > root.x:
      root.x = y
      root.l = self.remove(root.l, x, y)
    elif x > root.x and x < root.y and y > root.y:
      root.y = x
      root.r = self.remove(root.r, x, y)
    elif y < root.x:
      root.l = self.remove(root.l, x, y)
    elif x > root.y:
      root.r = self.remove(root.r, x, y)
    else:
      if root.l is None:
            return self.remove(root.r, x, y)
      elif root.r is None:
            return self.remove(root.l, x, y)
      rgt = self.smallest_interval(root.r)
      root.x = rgt.x
      root.y = rgt.y
      root.r = self.remove(root.r, rgt.x, rgt.y)
      root = self.remove(root, x, y)
    if not root:
      return None

    root.h = 1 + max(self.getHeight(root.l), 
                     self.getHeight(root.r))
    
    b = self.getBal(root)

    if b > 1 and self.getBal(root.l) >= 0:
        return self.rRotate(Node)

    if b < -1 and self.getBal(root.r) <= 0:
        return self.lRotate(root)

    if b > 1 and self.getBal(root.l) < 0:
        root.l = self.rotateL(root.l)
        return self.rRotate(root)

    if b < -1 and self.getBal(root.r) > 0:
        root.r = self.rotateR(root.r)
        return self.lRotate(root)

    return root
    

  def lRotate(self, z):

    y = z.r
    T2 = y.l

    y.l = z
    z.r = T2

    z.h = 1 + max(self.getHeight(z.l),
                  self.getHeight(z.r))
    y.h = 1 + max(self.getHeight(y.l),
                  self.getHeight(y.r))

    return y

  def rRotate(self, z):

    y = z.l
    T3 = y.r

    y.r = z
    z.l = T3

    z.h = 1 + max(self.getHeight(z.l),
                  self.getHeight(z.r))
    y.h = 1 + max(self.getHeight(y.l),
                  self.getHeight(y.r))

    return y

  def getHeight(self, root):
    if not root:
      return 0

    return root.h

  def getBal(self, root):
    if not root:
      return 0

    return self.getHeight(root.l) - self.getHeight(root.r)

  def order(self, root):
    ordered = []
    if root.l is not None:
      ordered = ordered + self.order(root.l)
    ordered.append((root.x,root.y))
    if root.r is not None:
      ordered = ordered + self.order(root.r)
    return ordered

  def __str__(self):
      return f'<IntervalUnionTree>'

  def __repr__(self):
      return self.__str__()