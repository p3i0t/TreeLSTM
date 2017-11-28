# vocab object from harvardnlp/opennmt-py
class DependRels(object):
    def __init__(self, filename=None, data=None, lower=False):
        self.idxToRel = {}
        self.relToIdx = {}
        self.lower = lower

        if filename  is not None:
            self.loadFile(filename)

    def size(self):
        return len(self.idxToRel)

    # Load entries from a file.
    def loadFile(self, filename):
        idx = 0
        for line in open(filename):
            rel = line.rstrip()
            self.add(rel)
            idx += 1

    def getIndex(self, key, default=None):
        try:
            return self.relToIdx[key]
        except KeyError:
            return default

    def getRel(self, idx, default=None):
        try:
            return self.idxToRel[idx]
        except KeyError:
            return default

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, rel):
        if rel in self.relToIdx:
            idx = self.relToIdx[rel]
        else:
            idx = len(self.idxToRel)
            self.idxToRel[idx] = rel
            self.relToIdx[rel] = idx
        return idx

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, rels):
        vec = []
        vec += [self.getIndex(rel) for rel in rels]

        return vec

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convertToLabels(self, idx, stop):
        rels = []

        for i in idx:
            rels += [self.getRel(i)]
            if i == stop:
                break

        return rels
