import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Var
from torch.nn import Parameter


class ChildSumTreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        ChildSumTreeLSTM for dependency parsing tree.
        :param vocab_size: size of the vocab.
        :param in_dim: input dimension.
        :param mem_dim: internal memory dimension of LSTM.
        """
        super(ChildSumTreeLSTM, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_gate = False

        """
        Define all the parameters for computing the 4 gates in TreeLSTM cell.
        see the original paper: https://arxiv.org/abs/1503.00075 
        """
        # linear parameters for transformation from input to hidden state. same for all 4 gates
        stdv = 1. / math.sqrt(input_size+hidden_size)

        self.i_weight = Parameter(torch.FloatTensor(hidden_size, input_size+hidden_size).uniform_(-stdv, stdv))
        self.i_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

        self.u_weight = Parameter(torch.FloatTensor(hidden_size, input_size+hidden_size).uniform_(-stdv, stdv))
        self.u_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

        if self._output_gate:
            self.o_weight = Parameter(torch.FloatTensor(hidden_size, input_size + hidden_size).uniform_(-stdv, stdv))
            self.o_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

        stdv = 1. / math.sqrt(input_size)
        self.fi_weight = Parameter(torch.FloatTensor(hidden_size, input_size).uniform_(-stdv, stdv))
        self.fi_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

        stdv = 1. / math.sqrt(hidden_size)
        self.fh_weight = Parameter(torch.FloatTensor(hidden_size, hidden_size).uniform_(-stdv, stdv))
        self.fh_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

    def forward(self, inputs, tree):
        """
        :param inputs:
        :param tree:
        :return: output, (h_n, c_n)
        """
        children_outputs = [self(inputs, child) for child in tree.children] # to be parallelized
        if children_outputs:
            _, children_states = zip(*children_outputs)
        else:
            children_states = None

        return self.node_forward(inputs[tree.idx].unsqueeze(0), children_states)

    def node_forward(self, inputs, children_states):
        # N for batch size
        # C for hidden size
        # K for number of children

        if children_states:
            h_sum = torch.sum(torch.cat([state[0].unsqueeze(0) for state in children_states]), dim=0)  # (1, C)
            children_h_cat = torch.cat([state[0].unsqueeze(1) for state in children_states], dim=1)  # (1, K, C)
            # concatenation of children cell states
            children_cs = torch.cat([state[1].unsqueeze(1) for state in children_states], dim=1)  # (1, K, C)
            f_act = F.linear(inputs, self.fi_weight, self.fi_bias) +\
                    F.linear(children_h_cat, self.fh_weight, self.fh_bias) #(1, K, C)
            f = F.sigmoid(f_act)
        else:
            h_sum = Parameter(torch.zeros(1, self._hidden_size))

        out = torch.cat([inputs, h_sum], dim=1)
        i = F.sigmoid(F.linear(out, self.i_weight, self.i_bias))
        u = F.tanh(F.linear(out, self.u_weight, self.u_bias))

        next_c = i * u
        if children_states:
            next_c = next_c + torch.sum(children_cs*f, dim=1)
        if self._output_gate:
            o = F.sigmoid(F.linear(out, self.o_weight, self.o_bias))
            next_h = o * F.tanh(next_c)
        else:
            next_h = F.tanh(next_c)

        return next_h, [next_h, next_c]


class ChildSumTreeLSTMVerbose(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        ChildSumTreeLSTM for dependency parsing tree.
        :param vocab_size: size of the vocab.
        :param in_dim: input dimension.
        :param mem_dim: internal memory dimension of LSTM.
        """
        super(ChildSumTreeLSTMVerbose, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._output_gate = False

        """
        Define all the parameters for computing the 4 gates in TreeLSTM cell.
        see the original paper: https://arxiv.org/abs/1503.00075 


        """
        # linear parameters for transformation from input to hidden state. same for all 4 gates
        stdv = 1. / math.sqrt(input_size+hidden_size)

        self.ix_weight = Parameter(torch.FloatTensor(hidden_size, input_size).uniform_(-stdv, stdv))
        self.ix_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

        self.ih_weight = Parameter(torch.FloatTensor(hidden_size, hidden_size).uniform_(-stdv, stdv))
        self.ih_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

        self.ux_weight = Parameter(torch.FloatTensor(hidden_size, input_size).uniform_(-stdv, stdv))
        self.ux_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

        self.uh_weight = Parameter(torch.FloatTensor(hidden_size, hidden_size).uniform_(-stdv, stdv))
        self.uh_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

        if self._output_gate:
            self.ox_weight = Parameter(torch.FloatTensor(hidden_size, input_size).uniform_(-stdv, stdv))
            self.ox_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

            self.oh_weight = Parameter(torch.FloatTensor(hidden_size, hidden_size).uniform_(-stdv, stdv))
            self.oh_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

        stdv = 1. / math.sqrt(input_size)
        self.fi_weight = Parameter(torch.FloatTensor(hidden_size, input_size).uniform_(-stdv, stdv))
        self.fi_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

        stdv = 1. / math.sqrt(hidden_size)
        self.fh_weight = Parameter(torch.FloatTensor(hidden_size, hidden_size).uniform_(-stdv, stdv))
        self.fh_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

    def forward(self, inputs, tree):
        """
        :param inputs:
        :param tree:
        :return: output, (h_n, c_n)
        """
        children_outputs = [self(inputs, child) for child in tree.children] # to be parallelized
        if children_outputs:
            _, children_states = zip(*children_outputs)
        else:
            children_states = None

        return self.node_forward(inputs[tree.idx].unsqueeze(0), children_states)

    def node_forward(self, inputs, children_states):
        # N for batch size
        # C for hidden size
        # K for number of children

        if children_states:
            h_sum = torch.sum(torch.cat([state[0].unsqueeze(0) for state in children_states]), dim=0)  # (1, C)
            children_h_cat = torch.cat([state[0].unsqueeze(1) for state in children_states], dim=1)  # (1, K, C)
            # concatenation of children cell states
            children_cs = torch.cat([state[1].unsqueeze(1) for state in children_states], dim=1)  # (1, K, C)
            f_act = F.linear(inputs, self.fi_weight, self.fi_bias) +\
                    F.linear(children_h_cat, self.fh_weight, self.fh_bias) #(1, K, C)
            f = F.sigmoid(f_act)
        else:
            h_sum = Parameter(torch.zeros(1, self._hidden_size))

        # out = torch.cat([inputs, h_sum], dim=1)
        # i = F.sigmoid(F.linear(out, self.i_weight, self.i_bias))
        # u = F.tanh(F.linear(out, self.u_weight, self.u_bias))
        i = F.sigmoid(F.linear(inputs, self.ix_weight, self.ix_bias) + F.linear(h_sum, self.ih_weight, self.ih_bias))
        u = F.tanh(F.linear(inputs, self.ux_weight, self.ux_bias) + F.linear(h_sum, self.uh_weight, self.uh_bias))

        next_c = i * u
        if children_states:
            next_c = next_c + torch.sum(children_cs*f, dim=1)
        if self._output_gate:
            o = F.sigmoid(F.linear(inputs, self.ox_weight, self.ox_bias)+F.linear(h_sum, self.oh_weight, self.oh_bias))
            next_h = o * F.tanh(next_c)
        else:
            next_h = F.tanh(next_c)

        return next_h, [next_h, next_c]


class ChildSumTreeGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        """
        ChildSumTreeGRU for dependency parsing tree.
        :param vocab_size: size of the vocab.
        :param in_dim: input dimension.
        :param mem_dim: internal memory dimension of LSTM.
        """
        super(ChildSumTreeLSTM, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size

        """
        Define all the parameters for computing the 4 gates in TreeLSTM cell.
        see the original paper: https://arxiv.org/abs/1503.00075 


        """
        # linear parameters for transformation from input to hidden state. same for all 4 gates
        stdv = 1. / math.sqrt(input_size+hidden_size)

        # update gate
        self.z_weight = Parameter(torch.FloatTensor(hidden_size, input_size+hidden_size).uniform_(-stdv, stdv))
        self.z_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

        # reset gate
        stdv = 1. / math.sqrt(input_size)
        self.ri_weight = Parameter(torch.FloatTensor(hidden_size, input_size).uniform_(-stdv, stdv))
        self.ri_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

        stdv = 1. / math.sqrt(hidden_size)
        self.rh_weight = Parameter(torch.FloatTensor(hidden_size, hidden_size).uniform_(-stdv, stdv))
        self.rh_bias = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

    def forward(self, inputs, tree):
        """
        :param inputs:
        :param tree:
        :return: output, (h_n, c_n)
        """
        children_outputs = [self(inputs, child) for child in tree.children]  # to be parallelized
        if children_outputs:
            _, children_states = zip(*children_outputs)
        else:
            children_states = None

        return self.node_forward(inputs[tree.idx].unsqueeze(0), children_states)

    def node_forward(self, inputs, children_states):
        # N for batch size
        # C for hidden size
        # K for number of children

        if children_states:
            h_sum = torch.sum(torch.cat([state[0].unsqueeze(0) for state in children_states]), dim=0)  # (1, C)
            children_h_cat = torch.cat([state[0].unsqueeze(1) for state in children_states], dim=1)  # (1, K, C)
            # concatenation of children cell states
            children_cs = torch.cat([state[1].unsqueeze(1) for state in children_states], dim=1)  # (1, K, C)
            f_act = F.linear(inputs, self.fi_weight, self.fi_bias) +\
                    F.linear(children_h_cat, self.fh_weight, self.fh_bias) #(1, K, C)
            f = F.sigmoid(f_act)
        else:
            h_sum = Parameter(torch.zeros(1, self._hidden_size))

        out = torch.cat([inputs, h_sum], dim=1)
        i = F.sigmoid(F.linear(out, self.i_weight, self.i_bias))
        u = F.tanh(F.linear(out, self.u_weight, self.u_bias))

        next_c = i * u
        if children_states:
            next_c = next_c + torch.sum(children_cs*f, dim=1)
        if self._output_gate:
            o = F.sigmoid(F.linear(out, self.o_weight, self.o_bias))
            next_h = o * F.tanh(next_c)
        else:
            next_h = F.tanh(next_c)

        return next_h, [next_h, next_c]


class Similarity(nn.Module):
    def __init__(self, input_size, hidden_size, n_classes):
        """

        :param input_size: input size of the two way inputs
        :param hidden_size:
        :param n_classes:
        """
        super(Similarity, self).__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self._num_classes = n_classes

        stdv = 1 / math.sqrt(2 * input_size)
        self.w1 = Parameter(torch.FloatTensor(hidden_size, 2 * input_size).uniform_(-stdv, stdv))
        self.b1 = Parameter(torch.FloatTensor(hidden_size).uniform_(-stdv, stdv))

        stdv = 1 / math.sqrt(2 * input_size)
        self.w2 = Parameter(torch.FloatTensor(n_classes, hidden_size).uniform_(-stdv, stdv))
        self.b2 = Parameter(torch.FloatTensor(n_classes).uniform_(-stdv, stdv))

    def forward(self, lvec, rvec):
        """

        :param lvec:
        :param rvec:
        :return: log probabilities.
        """
        mult = lvec * rvec
        abs_sub = torch.abs(lvec - rvec)
        v = torch.cat([mult, abs_sub], dim=1)
        out = F.sigmoid(F.linear(v, self.w1, self.b1))
        # out = self.bm(out)
        out = F.log_softmax(F.linear(out, self.w2, self.b2))
        return out


class TreeLSTMSimilarity(nn.Module):
    def __init__(self, vocab_size, input_size, mem_size, hidden_size, n_classes, n_relations, with_rel):
        super(TreeLSTMSimilarity, self).__init__()
        # Embedding matrix in shape of (vocab_size, in_dim)
        self.embedding = nn.Embedding(vocab_size, input_size)
        self.embedding.weight.requires_grad = False

        self._with_rel = with_rel
        if with_rel:
            self.rel_embedding = nn.Embedding(n_relations, n_relations)
            self.rel_embedding.weight.data = torch.eye(n_relations, n_relations)
            self.rel_embedding.weight.requires_grad = False
            input_size += n_relations

        self.treelstm = ChildSumTreeLSTMVerbose(input_size, mem_size)
        self.similarity = Similarity(mem_size, hidden_size, n_classes)

    def forward(self, ltree, lsent, lrel, rtree, rsent, rrel):
        linputs = self.embedding(lsent)
        rinputs = self.embedding(rsent)

        if self._with_rel:
            lrel = self.rel_embedding(lrel)
            rrel = self.rel_embedding(rrel)
            linputs = torch.cat([linputs, lrel], dim=1)
            rinputs = torch.cat([rinputs, rrel], dim=1)

        lvec = self.treelstm(linputs, ltree)[0]
        rvec = self.treelstm(rinputs, rtree)[0]
        out = self.similarity(lvec, rvec)
        return out



