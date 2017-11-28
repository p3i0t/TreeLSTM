import os
import random
import logging
from tqdm import tqdm
import numpy
import torch
import torch.optim as optim
from torch.autograd import Variable as Var
import torch.nn.functional as F

from model import TreeLSTMSimilarity
import Constants
from config import parse_args
from utils import build_vocab, build_rels, map_label_to_target
from vocab import Vocab
from rels import DependRels
from dataset import SICKDataset
from metrics import Metrics


def load_data(filename, file_dir, vocab, depend_rels, n_classes):
    if os.path.isfile(filename):
        dataset = torch.load(filename)
    else:
        dataset = SICKDataset(file_dir, vocab, depend_rels, n_classes)
        torch.save(dataset, filename)
    return dataset


def convert_time(n_seconds):
    m, s = divmod(n_seconds, 60)
    h, m = divmod(m, 60)
    strs = []
    strs.append('{:2d}h'.format(int(h)))
    strs.append('{:2d}m'.format(int(m)))
    strs.append('{:2d}s'.format(int(s)))
    return '|'.join(strs)


def train(args, model, vocab, depend_rels, logger):
    # torch.manual_seed(args.seed+rank)
    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')

    # load SNLI dataset splits
    train_file = os.path.join(args.data, 'snli_train.pth')
    train_dataset = load_data(train_file, train_dir, vocab, depend_rels, args.num_classes)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))

    dev_file = os.path.join(args.data, 'snli_dev.pth')
    dev_dataset = load_data(dev_file, dev_dir, vocab, depend_rels, args.num_classes)
    logger.debug('==> Size of dev data     : %d ' % len(dev_dataset))

    test_file = os.path.join(args.data, 'snli_test.pth')
    test_dataset = load_data(test_file, test_dir, vocab, depend_rels, args.num_classes)
    logger.debug('==> Size of test data    : %d ' % len(test_dataset))

    trainable_parameters = filter(lambda x: x.requires_grad, model.parameters())

    if args.optim == 'adam':
        optimizer = optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(trainable_parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(trainable_parameters, lr=args.lr, weight_decay=args.wd)

    best_pearson = -float('inf')
    metrics = Metrics(args.num_classes)
    for epoch in range(1, args.epochs + 1):
        train_epoch(epoch, args, model, train_dataset, optimizer)

        if epoch == args.epochs:
            train_loss, train_pred = test_epoch(args, model, train_dataset)
        dev_loss, dev_pred = test_epoch(args, model, dev_dataset)
        test_loss, test_pred = test_epoch(args, model, test_dataset)

        if epoch == args.epochs:
            train_pearson = metrics.pearson(train_pred, train_dataset.labels)
            train_mse = metrics.mse(train_pred, train_dataset.labels)
            logger.debug("Epoch {}, Train loss: {:.4f}, Pearson: {:.4f}, MSE: {:.4f}".format(epoch, train_loss, train_pearson, train_mse))

        dev_pearson = metrics.pearson(dev_pred, dev_dataset.labels)
        dev_mse = metrics.mse(dev_pred, dev_dataset.labels)
        logger.debug("Epoch {}, Dev loss: {:.4f}, Pearson: {:.4f}, MSE: {:.4f}".format(epoch, dev_loss, dev_pearson, dev_mse))

        test_pearson = metrics.pearson(test_pred, test_dataset.labels)
        test_mse = metrics.mse(test_pred, test_dataset.labels)
        logger.debug("Epoch {}, Test loss: {:.4f}, Pearson: {:.4f}, MSE: {:.4f}".format(epoch, test_loss, test_pearson, test_mse))

        if test_pearson > best_pearson:
            best_pearson = test_pearson
            checkpoint = {'model': model.state_dict(), 'optim': optimizer,
                          'pearson': best_pearson, 'mse': test_mse,
                          'args': args, 'epoch': epoch}
            logger.debug("====> New optimum found, saving checkpoint ...")
            torch.save(checkpoint, os.path.join(args.save, args.expname + '.pth'))


def train_epoch(epoch, args, model, dataset, optimizer):
    model.train()
    optimizer.zero_grad()
    indices = torch.randperm(len(dataset))
    batch_size = args.batch_size
    loss, k = 0.0, 0
    for idx in tqdm(range(len(dataset)), desc="Training epoch {}".format(epoch)):
        ltree, lsent, lrel, rtree, rsent, rrel, sim = dataset[indices[idx]]
        linput, rinput = Var(lsent), Var(rsent)
        lrel, rrel = Var(lrel), Var(rrel)
        target = Var(map_label_to_target(sim, args.num_classes))

        if args.cuda:
            linput, rinput = linput.cuda(), rinput.cuda()
            lrel, rrel = lrel.cuda(), rrel.cuda()
            target = target.cuda()

        output = model(ltree, linput, lrel, rtree, rinput, rrel)
        err = F.kl_div(output, target)
        loss += err.data[0]
        (err/batch_size).backward()
        k += 1
        if k % batch_size == 0:

            optimizer.step()
            optimizer.zero_grad()
    avg_loss = loss/len(dataset)
    return avg_loss


def test_epoch(args, model, dataset):
    model.eval()
    test_loss = 0
    predictions = torch.zeros(len(dataset))
    indices = torch.arange(1, dataset.num_classes + 1)
    #for idx in range(len(dataset)):
    for idx in tqdm(range(len(dataset)), desc="Testing "):
        ltree, lsent, lrel, rtree, rsent, rrel, sim = dataset[idx]
        linput, rinput = Var(lsent, volatile=True), Var(rsent, volatile=True)
        lrel, rrel = Var(lrel, volatile=True), Var(rrel, volatile=True)
        target = Var(map_label_to_target(sim, args.num_classes), volatile=True)

        if args.cuda:
            linput, rinput = linput.cuda(), rinput.cuda()
            lrel, rrel = lrel.cuda(), rrel.cuda()
            target = target.cuda()

        out = model(ltree, linput, lrel, rtree, rinput, rrel)
        test_loss += F.kl_div(out, target).data[0]
        out = out.data.squeeze().cpu()

        predictions[idx] = torch.dot(indices, torch.exp(out))
    test_loss /= len(dataset)
    return test_loss, predictions


def main():
    global args
    args = parse_args()
    args.input_dim, args.mem_dim = 300, 150
    args.hidden_dim, args.num_classes = 50, 5
    args.cuda = args.cuda and torch.cuda.is_available()

    if args.sparse and args.wd != 0:
        print('Sparsity and weight decay are incompatible, pick one!')
        exit()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    numpy.random.seed(args.seed)

    file_halder = logging.FileHandler(os.path.join(args.save, args.expname+'.log'), mode='a')
    formatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')
    file_halder.setFormatter(formatter)
    logger = logging.getLogger(__name__)
    logger.addHandler(file_halder)
    logger.setLevel(logging.DEBUG)

    print(args)
    logger.debug(args)
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    train_dir = os.path.join(args.data, 'train')
    dev_dir = os.path.join(args.data, 'dev')
    test_dir = os.path.join(args.data, 'test')

    rels_file = os.path.join(args.data, 'rels_set.txt')
    if not os.path.isfile(rels_file):
        files_a = [os.path.join(split, 'a.rels') for split in [train_dir, dev_dir, test_dir]]
        files_b = [os.path.join(split, 'b.rels') for split in [train_dir, dev_dir, test_dir]]
        files = files_a + files_b
        build_rels(files, rels_file)

    depend_rels = DependRels(filename=rels_file)
    logger.debug("==> Depenency Relations size: {}".format(depend_rels.size()))

    # write unique words from all token files
    sick_vocab_file = os.path.join(args.data, 'sick.vocab')
    if not os.path.isfile(sick_vocab_file):
        token_files_a = [os.path.join(split, 'a.toks') for split in [train_dir, dev_dir, test_dir]]
        token_files_b = [os.path.join(split, 'b.toks') for split in [train_dir, dev_dir, test_dir]]
        token_files = token_files_a + token_files_b
        sick_vocab_file = os.path.join(args.data, 'sick.vocab')
        build_vocab(token_files, sick_vocab_file)

    # get vocab object from vocab file previously written
    vocab = Vocab(filename=sick_vocab_file,
                  data=[Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD])
    logger.debug('==> SICK vocabulary size : %d ' % vocab.size())

    model = TreeLSTMSimilarity(
        vocab.size(),
        args.input_dim,
        args.mem_dim,
        args.hidden_dim,
        args.num_classes,
        depend_rels.size(),
        args.with_rels)

    emb_file = os.path.join(args.data, 'sick_embed.pth')
    if os.path.isfile(emb_file):
        emb = torch.load(emb_file)
    else:
        raise KeyError('No glove embeddings available')

    model.embedding.weight.data.copy_(emb)

    train(args, model, vocab, depend_rels, logger)


if __name__ == '__main__':
    main()


