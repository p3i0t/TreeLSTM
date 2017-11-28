class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        """

        :param child: a Tree object represent the child
        :return:
        """
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if hasattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        """
        Calculate the depth of the tree recursively.
        :return:
        """
        if hasattr(self, '_depth'):
            return self._depth

        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                if self.children[i].depth() > count:
                    count = self.children[i].depth()
            count += 1
        self._depth = count
        return self._depth

    def get_levels(self):
        if hasattr(self, '_level'):
            return self._level
        self._level = 1

        queue = []
        queue.append(self)
        while len(queue) > 0:
            p = queue.pop(0)
            for i in range(p.num_children):
                p.children[i]._level = p._level + 1
                queue.append(p.children[i])

    def display(self):
        return 'Number of children: {}, depth: {}\n children: {}'.format(
            self.size(), self.depth(), self.children)