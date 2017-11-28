import os
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.utils.data as data
from tree import Tree
from vocab import Vocab
import Constants


# Dataset class for SICK dataset
class SICKDataset(data.Dataset):
    def __init__(self, path, vocab, depend_rels, num_classes):
        super(SICKDataset, self).__init__()
        self.vocab = vocab
        self.rels = depend_rels
        self.num_classes = num_classes

        self.lsentences = self.read_sentences(os.path.join(path, 'a.toks'))
        self.rsentences = self.read_sentences(os.path.join(path, 'b.toks'))

        self.lrels = self.read_rels(os.path.join(path, 'a.rels'))
        self.rrels = self.read_rels(os.path.join(path, 'b.rels'))

        self.ltrees = self.read_trees(os.path.join(path, 'a.parents'))
        self.rtrees = self.read_trees(os.path.join(path, 'b.parents'))

        self.labels = self.read_labels(os.path.join(path, 'sim.txt'))

        self.size = self.labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        ltree = deepcopy(self.ltrees[index])
        rtree = deepcopy(self.rtrees[index])
        lrel = deepcopy(self.lrels[index])
        rrel = deepcopy(self.rrels[index])
        lsent = deepcopy(self.lsentences[index])
        rsent = deepcopy(self.rsentences[index])
        label = deepcopy(self.labels[index])
        return ltree, lsent, lrel, rtree, rsent, rrel, label

    def read_sentences(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.LongTensor(indices)

    def read_rels(self, filename):
        with open(filename, 'r') as f:
            rels = [self.read_rel(line) for line in tqdm(f.readlines())]
        return rels

    def read_rel(self, line):
        indices = self.rels.convertToIdx(line.split())
        return torch.LongTensor(indices)

    def read_trees(self, filename):
        with open(filename, 'r') as f:
            trees = [self.read_tree(line) for line in tqdm(f.readlines())]
        return trees

    def read_tree(self, line):
        parents = list(map(int,line.split()))
        trees = dict()
        root = None
        for i in range(1,len(parents)+1):
            if i-1 not in trees.keys() and parents[i-1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx-1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx-1] = tree
                    tree.idx = idx-1
                    #if trees[parent-1] is not None:
                    if parent-1 in trees.keys():
                        trees[parent-1].add_child(tree)
                        break
                    elif parent==0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        #root = self.compact_tree(root, level=3)
        #print('root depth after compacting: ', root.depth())
        return root

    # def collect_indices(self, root, indices=[]):
    #     if root is not None:
    #         indices.append(root.idx)
    #         for i in range(root.num_children):
    #             self.collect_indices(root.children[i], indices)
    #     return indices
    #
    # def compact_tree(self, root, level=3):
    #     if root is not None:
    #         if root.get_levels() == level:
    #             indices = self.collect_indices(root, [])
    #             root.num_children = 0
    #             root.idx = sorted(indices)
    #             return
    #         for i in range(root.num_children):
    #             self.compact_tree(root.children[i], level)
    #     return root

    def read_labels(self, filename):
        with open(filename, 'r') as f:
            labels = list(map(float, f.readlines()))
            labels = torch.Tensor(labels)
        return labels
