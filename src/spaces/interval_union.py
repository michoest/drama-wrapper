from typing import SupportsFloat, Union, Optional, Any

import numpy as np

from src.spaces.continuous import ContinuousActionSpace

from decimal import *
from random import uniform


class Node(object):
    """ Node in the AVL tree which represents a valid interval """

    def __init__(self, x: float = None, y: float = None, left: object = None, right: object = None, height: int = 1):
        """
        Args:
            x (float): Lower bound of the interval
            y (float): Upper bound of the interval
            left (Node): Left, smaller interval
            right (Node): Right, larger interval
        """
        self.x: Decimal = Decimal(f'{x}') if x is not None else None
        self.y: Decimal = Decimal(f'{y}') if y is not None else None
        self.l = left
        self.r = right
        self.h = height

    def __str__(self):
        return f'<Node ({self.x},{self.y}), height: {self.h}, left: {self.l}, right: {self.r}>'

    def __repr__(self):
        return self.__str__()


class IntervalUnionTree(ContinuousActionSpace):
    """ Interval Action Space as AVL tree """

    root_tree = None
    size: Decimal = 0
    draw = None

    def __init__(self, low: SupportsFloat, high: SupportsFloat,
                 dtype: Union[type[np.floating[Any]], type[np.integer[Any]]] = np.float32,
                 seed: Optional[Union[int, np.random.Generator]] = None) -> None:
        super().__init__(low, high, dtype, seed)
        getcontext().prec = 28

        self.root_tree = Node(self.low[0], self.high[0])
        self.size = Decimal(self.high[0]) - Decimal(self.low[0])

    def __contains__(self, item):
        return self.contains(item)

    def contains(self, x, root: object = 'root'):
        """ Determines if a number is part of the action space

        Args:
            x: Number
            root: Node to start the search from or 'root' for searching the whole tree, default is 'root'

        Returns:
            Boolean indicating if it is part of the action space
        """
        if root == 'root':
            root = self.root_tree

        x = Decimal(f'{x}')

        if not root:
            return False
        elif root.x <= x <= root.y:
            return True
        elif root.x > x:
            return self.contains(x, root.l)
        else:
            return self.contains(x, root.r)

    def nearest_elements(self, x, root: Node = 'root'):
        """ Finds nearest actions for a number in the action space

        Args:
            x: Number
            root: Node to start the search from or 'root' for searching the whole tree, default is 'root'

        Returns:
            Nearest elements in the action space. It is the number itself if it is valid.
        """
        if root == 'root':
            root = self.root_tree

        x = Decimal(f'{x}')

        if x > root.y:
            return self._nearest_elements(x, x - root.y, root.y, root.r)
        elif x < root.x:
            return self._nearest_elements(x, root.x - x, root.x, root.l)
        else:
            return x

    def _nearest_elements(self, x, min_diff, min_value, root: Node = 'root'):
        if root == 'root':
            root = self.root_tree

        x = Decimal(f'{x}')
        min_diff = Decimal(f'{min_diff}')
        min_value = Decimal(f'{min_value}')

        if not root:
            return [min_value]
        elif x > root.y:
            distance = x - root.y
            return [min_value, root.y] if distance == min_diff else [
                min_value] if distance > min_diff else self._nearest_elements(x, distance, root.y, root.r)
        elif x < root.x:
            distance = root.x - x
            return [min_value, root.x] if distance == min_diff else [
                min_value] if distance > min_diff else self._nearest_elements(x, distance, root.x, root.l)
        else:
            return x

    def nearest_element(self, x, root: Node = 'root'):
        """ Finds nearest action for a number in the action space. Larger actions preferred.

        Args:
            x: Number
            root: Node to start the search from or 'root' for searching the whole tree, default is 'root'

        Returns:
            Nearest element in the action space. It is the number itself if it is valid.
        """
        if root == 'root':
            root = self.root_tree

        x = Decimal(f'{x}')

        return self.nearest_elements(x, root)[-1]

    def last_interval_before_or_within(self, x, root: Node = 'root'):
        """ Returns the last interval before or within a number

        Args:
            x: Number
            root: Node to start the search from or 'root' for searching the whole tree, default is 'root'

        Returns:
            Tuple containing the lower and upper boundaries of the interval and a variable indicating
            if the number lies in the interval. For example:

            (root.x, root.y), True
        """
        if root == 'root':
            root = self.root_tree

        x = Decimal(f'{x}')

        if root.x <= x <= root.y:
            return (root.x, root.y), True
        elif x < root.x:
            return self.last_interval_before_or_within(x, root.l) if root.l is not None else ((root.x, root.y), False)
        else:
            return self.last_interval_before_or_within(x, root.r) if root.r is not None else (
                (root.x, root.y), False) if x < root.y else ((None, None), False)

    def first_interval_after_or_within(self, x, root: Node = 'root'):
        """ Returns the first interval after or within a number

        Args:
            x: Number
            root: Node to start the search from or 'root' for searching the whole tree, default is 'root'

        Returns:
            Tuple containing the lower and upper boundaries of the interval and a variable indicating
            if the number lies in the interval. For example:

            (root.x, root.y), True
        """
        if root == 'root':
            root = self.root_tree

        x = Decimal(f'{x}')

        if root.x <= x <= root.y:
            return (root.x, root.y), True
        elif x > root.y:
            return self.first_interval_after_or_within(x, root.r) if root.r is not None else ((root.x, root.y), False)
        else:
            return self.first_interval_after_or_within(x, root.l) if root.l is not None else (
                (root.x, root.y), False) if x > root.x else ((None, None), False)

    def smallest_interval(self, root: Node = 'root'):
        """ Returns the Node of the smallest interval

        Args:
            root: Node to start the search from or 'root' for searching the whole tree, default is 'root'

        Returns:
            Node of the smallest interval
        """
        if root == 'root':
            root = self.root_tree

        if root is None or root.l is None:
            return root
        else:
            return self.smallest_interval(root.l)

    def add(self, x, y, root: Node = 'root'):
        """ Adds an interval to the action space

        Args:
            x: Lower bound of the interval
            y: Upper bound of the interval
            root: Node to start the insertion from or 'root' for inserting over the whole tree, default is 'root'

        Returns:
            Updated root node of the action space
        """
        assert y > x, 'Upper must be larger than lower bound'

        if root == 'root':
            root = self.root_tree
            if root is None:
                self.root_tree = Node(x, y)
                self.size += y - x
                return self.root_tree

        x = Decimal(f'{x}')
        y = Decimal(f'{y}')
        if not root:
            self.size += y - x
            return Node(x, y)
        elif y < root.x:
            root.l = self.add(x, y, root.l)
        elif x > root.y:
            root.r = self.add(x, y, root.r)
        else:
            old_size = root.y - root.x
            root.x = min(root.x, x)
            root.y = max(root.y, y)
            self.size += root.y - root.x - old_size

            updated = False
            if root.r is not None and root.y >= root.r.x:
                self.size -= root.y - root.r.y
                root.y = root.r.y
                updated = True

            if root.l is not None and root.x <= root.l.y:
                self.size -= root.l.x - root.x
                root.x = root.l.x
                updated = True

            root.r = self.remove(root.x, root.y, root.r)
            root.l = self.remove(root.x, root.y, root.l)
            if updated:
                root = self.add(x, y, root)

        root.h = 1 + max(self.getHeight(root.l),
                         self.getHeight(root.r))

        b = self.getBal(root)

        if b > 1 and y < root.l.x and self.getBal(root.l) > 0:
            self.root_tree = self.rRotate(root)
            return self.root_tree

        if b < -1 and x > root.r.y and self.getBal(root.r) < 0:
            self.root_tree = self.lRotate(root)
            return self.root_tree

        if b > 1 and x > root.l.y and self.getBal(root.l) < 0:
            root.l = self.lRotate(root.l)
            self.root_tree = self.rRotate(root)
            return self.root_tree

        if b < -1 and y < root.r.x and self.getBal(root.r) > 0:
            root.r = self.rRotate(root.r)
            self.root_tree = self.lRotate(root)
            return self.root_tree

        self.root_tree = root
        return root

    def sample(self, root: Node = 'root') -> float:
        """ Sample a random action from a uniform distribution over the action space

        Args:
            root: Root node of the action space, default is 'root'

        Returns:
            Sampled action as a float
        """
        if root == 'root':
            root = self.root_tree

        if root is None:
            raise Exception('Empty Action Space')

        if self.draw is None:
            self.draw = Decimal(f'{uniform(0.0, float(self.size))}')

        self.draw -= root.y - root.x
        if self.draw > 0:
            result = None
            if root.l is not None:
                result = self.sample(root.l)
            if not result and root.r is not None:
                result = self.sample(root.r)
            return result
        else:
            result = float(root.y + self.draw)
            self.draw = None
            return result

    def remove(self, x, y, root: Node = 'root', adjust_size: bool = True):
        """ Removes an interval from the action space

        Args:
            x: Lower bound of the interval
            y: Upper bound of the interval
            root: Node to start the removal from or 'root' for removing over the whole tree, default is 'root'

        Returns:
            Updated root node of the action space
        """
        assert y > x, 'Upper must be larger than lower bound'

        if root == 'root':
            root = self.root_tree
            if root is None:
                return root

        x = Decimal(f'{x}')
        y = Decimal(f'{y}')

        if not root:
            return None
        elif x > root.x and y < root.y:
            self.size -= root.y - x
            old_maximum = root.y
            root.y = x
            root = self.add(y, old_maximum, root)
        elif x == root.x and y < root.y:
            self.size -= y - x
            root.x = y
        elif x > root.x and y == root.y:
            self.size -= y - x
            root.y = x
        elif x < root.x < y < root.y:
            self.size -= y - root.x
            root.x = y
            root.l = self.remove(x, y, root.l, adjust_size)
        elif root.x < x < root.y < y:
            self.size -= root.y - x
            root.y = x
            root.r = self.remove(x, y, root.r, adjust_size)
        elif y <= root.x:
            root.l = self.remove(x, y, root.l, adjust_size)
        elif x >= root.y:
            root.r = self.remove(x, y, root.r, adjust_size)
        else:
            if adjust_size:
                self.size -= root.y - root.x
            if root.l is None:
                self.root_tree = self.remove(x, y, root.r, adjust_size)
                return self.root_tree
            elif root.r is None:
                self.root_tree = self.remove(x, y, root.l, adjust_size)
                return self.root_tree
            rgt = self.smallest_interval(root.r)
            root.x = rgt.x
            root.y = rgt.y
            root.r = self.remove(rgt.x, rgt.y, root.r, adjust_size=False)
            root = self.remove(x, y, root, adjust_size)
        if not root:
            return None

        root.h = 1 + max(self.getHeight(root.l),
                         self.getHeight(root.r))

        b = self.getBal(root)

        if b > 1 and self.getBal(root.l) > 0:
            self.root_tree = self.rRotate(root)
            return self.root_tree

        if b < -1 and self.getBal(root.r) < 0:
            self.root_tree = self.lRotate(root)
            return self.root_tree

        if b > 1 and self.getBal(root.l) < 0:
            root.l = self.lRotate(root.l)
            self.root_tree = self.rRotate(root)
            return self.root_tree

        if b < -1 and self.getBal(root.r) > 0:
            root.r = self.rRotate(root.r)
            self.root_tree = self.lRotate(root)
            return self.root_tree

        self.root_tree = root
        return root

    def lRotate(self, z: Node):
        """ Performs a left rotation. Switches roles of parent and child nodes.

        Args:
            z (Node): Parent node for the rotation

        Returns:
            Updated parent Node
        """
        y = z.r
        T2 = y.l

        y.l = z
        z.r = T2

        z.h = 1 + max(self.getHeight(z.l),
                      self.getHeight(z.r))
        y.h = 1 + max(self.getHeight(y.l),
                      self.getHeight(y.r))

        return y

    def rRotate(self, z: Node):
        """ Performs a right rotation. Switches roles of parent and child nodes.

        Args:
            z (Node): Parent node for the rotation

        Returns:
            Updated parent Node
        """
        y = z.l
        T3 = y.r

        y.r = z
        z.l = T3

        z.h = 1 + max(self.getHeight(z.l),
                      self.getHeight(z.r))
        y.h = 1 + max(self.getHeight(y.l),
                      self.getHeight(y.r))

        return y

    def getHeight(self, root: Node = 'root'):
        """ Returns the height of a Node

        Args:
            root: Node to return the height from or 'root' for the height of the whole tree, default is 'root'

        Returns:
            Integer indicating the height
        """
        if root == 'root':
            root = self.root_tree

        if not root:
            return 0

        return root.h

    def getBal(self, root: Node = 'root'):
        """ Calculates balance factor

        Args:
            root: Node to calculate the balance factor for or 'root' for the balance factor of the whole tree,
            default is 'root'

        Returns:
            Integer indicating the balance factor
        """
        if root == 'root':
            root = self.root_tree

        if not root:
            return 0

        return self.getHeight(root.l) - self.getHeight(root.r)

    def intervals(self, root: Node = 'root'):
        """ Returns all intervals of the action space ordered

        Args:
            root: Node to start the search from or 'root' for searching the whole tree, default is 'root'

        Returns:
            List of tuples containing the ordered intervals. For example:

            [(0.1,0.5), (0.7,0.9)]
        """
        if root == 'root':
            root = self.root_tree

        if root is None:
            return []

        ordered = []
        if root.l is not None:
            ordered = ordered + self.intervals(root.l)
        ordered.append((float(root.x), float(root.y)))
        if root.r is not None:
            ordered = ordered + self.intervals(root.r)
        return ordered

    def reset(self):
        """ Resets the action space to the unrestricted state
        """
        self.root_tree = Node(self.low[0], self.high[0])
        self.size = Decimal(self.high[0]) - Decimal(self.low[0])

    def __str__(self):
        return f'<IntervalUnion>'

    def __repr__(self):
        return self.__str__()
