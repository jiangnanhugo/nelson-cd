# Implementation of Graph-based dependency parsing
import os
import sys
import gc
import json

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import time
import argparse
import math
import numpy as np
np.random.seed(10086)
import torch
torch.manual_seed(10086)
from torch.optim.adamw import AdamW
from torch.optim import SGD
from torch.nn.utils import clip_grad_norm_
from neuronlp2.nn.utils import total_grad_norm
from neuronlp2.io import get_logger, conllx_data, conllx_stacked_data, iterate_data
from neuronlp2.models import DeepBiAffine, NeuroMST, ConstraintDeepBiAffine, SampledMST
from neuronlp2.optim import ExponentialScheduler
from neuronlp2 import utils
from neuronlp2.io import CoNLLXWriter
from neuronlp2.nn.utils import freeze_embedding
from eval import evaluate_on_dev, eval


def get_optimizer(parameters, optim, learning_rate, lr_decay, betas, eps, amsgrad, weight_decay, warmup_steps):
    if optim == 'sgd':
        optimizer = SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    else:
        optimizer = AdamW(parameters, lr=learning_rate, betas=betas, eps=eps, amsgrad=amsgrad,
                          weight_decay=weight_decay)
    init_lr = 1e-7
    scheduler = ExponentialScheduler(optimizer, lr_decay, warmup_steps, init_lr)
    return optimizer, scheduler


def train(args):
    logger = get_logger("Parsing")

    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    train_path = args.train
    dev_path = args.dev
    test_path = args.test

    num_epochs = args.num_epochs
    batch_size = args.batch_size
    optim = args.optim
    learning_rate = args.learning_rate
    lr_decay = args.lr_decay
    amsgrad = args.amsgrad
    eps = args.eps
    betas = (args.beta1, args.beta2)
    warmup_steps = args.warmup_steps
    weight_decay = args.weight_decay
    grad_clip = args.grad_clip

    loss_ty_token = args.loss_type == 'token'
    unk_replace = args.unk_replace
    freeze = args.freeze

    model_name = os.path.join(args.model_path, 'model.pt')
    punctuation = args.punctuation

    print(args)

    word_dict, word_dim = utils.load_embedding_dict(args.word_embedding, args.word_path)
    char_dict = None
    if args.char_embedding != 'random':
        char_dict, char_dim = utils.load_embedding_dict(args.char_embedding, args.char_path)
    else:
        char_dict = None
        char_dim = None

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(args.model_path, 'alphabets')
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_data.create_alphabets(alphabet_path, train_path,
                                                                                             data_paths=[dev_path,
                                                                                                         test_path],
                                                                                             embedd_dict=word_dict,
                                                                                             max_vocabulary_size=200000)



    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    result_path = os.path.join(args.model_path, 'tmp')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    def construct_word_embedding_table():
        scale = np.sqrt(3.0 / word_dim)
        table = np.empty([word_alphabet.size(), word_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.zeros([1, word_dim]).astype(np.float32) if freeze else np.random.uniform(
            -scale, scale, [1, word_dim]).astype(np.float32)
        oov = 0
        for word, index in word_alphabet.items():
            if word in word_dict:
                embedding = word_dict[word]
            elif word.lower() in word_dict:
                embedding = word_dict[word.lower()]
            else:
                zero_vect = np.zeros([1, word_dim]).astype(np.float32)
                rand_vect = np.random.uniform(-scale, scale, [1, word_dim]).astype(np.float32)
                embedding = zero_vect if freeze else rand_vect
                oov += 1
            table[index, :] = embedding
        print('word OOV: %d' % oov)
        return torch.from_numpy(table)

    def construct_char_embedding_table():
        if char_dict is None:
            return None

        scale = np.sqrt(3.0 / char_dim)
        table = np.empty([num_chars, char_dim], dtype=np.float32)
        table[conllx_data.UNK_ID, :] = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
        oov = 0
        for char, index, in char_alphabet.items():
            if char in char_dict:
                embedding = char_dict[char]
            else:
                embedding = np.random.uniform(-scale, scale, [1, char_dim]).astype(np.float32)
                oov += 1
            table[index, :] = embedding
        print('character OOV: %d' % oov)
        return torch.from_numpy(table)

    word_table = construct_word_embedding_table()
    char_table = construct_char_embedding_table()

    logger.info("constructing network...")

    hyps = json.load(open(args.config, 'r'))
    json.dump(hyps, open(os.path.join(args.model_path, 'config.json'), 'w'), indent=2)
    model_type = hyps['model']
    assert model_type in ['DeepBiAffine', 'ConstraintDeepBiAffine', 'NeuroMST', 'SampledMST']
    assert word_dim == hyps['word_dim']
    if char_dim is not None:
        assert char_dim == hyps['char_dim']
    else:
        char_dim = hyps['char_dim']
    use_pos = hyps['pos']
    pos_dim = hyps['pos_dim']
    mode = hyps['rnn_mode']
    hidden_size = hyps['hidden_size']
    arc_space = hyps['arc_space']
    type_space = hyps['type_space']
    p_in = hyps['p_in']
    p_out = hyps['p_out']
    p_rnn = hyps['p_rnn']
    activation = hyps['activation']
    prior_order = None

    alg = 'graph'
    if model_type == 'DeepBiAffine':
        num_layers = hyps['num_layers']
        network = DeepBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                               mode, hidden_size, num_layers, num_types, arc_space, type_space,
                               embedd_word=word_table, embedd_char=char_table,
                               p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
    elif model_type == 'ConstraintDeepBiAffine':
        num_layers = hyps['num_layers']
        network = ConstraintDeepBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                                         mode, hidden_size, num_layers, num_types, arc_space, type_space,
                                         embedd_word=word_table, embedd_char=char_table,
                                         p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
    elif model_type == 'NeuroMST':
        num_layers = hyps['num_layers']
        network = NeuroMST(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                           mode, hidden_size, num_layers, num_types, arc_space, type_space,
                           embedd_word=word_table, embedd_char=char_table,
                           p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
    elif model_type == 'SampledMST':
        num_layers = hyps['num_layers']
        network = SampledMST(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                           mode, hidden_size, num_layers, num_types, arc_space, type_space,
                           embedd_word=word_table, embedd_char=char_table,
                           p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
    else:
        raise RuntimeError('Unknown model type: %s' % model_type)

    if freeze:
        freeze_embedding(network.word_embed)

    network = network.to(device)
    model = "{}-{}".format(model_type, mode)
    logger.info("Network: %s, num_layer=%s, hidden=%d, act=%s" % (model, num_layers, hidden_size, activation))
    logger.info("dropout(in, out, rnn): %s(%.2f, %.2f, %s)" % ('variational', p_in, p_out, p_rnn))
    logger.info('# of Parameters: %d' % (sum([param.numel() for param in network.parameters()])))

    logger.info("Reading Data")
    if alg == 'graph':
        constraint_pos_dict, constraint_relation_dict = conllx_data.create_pos_consraint(args.const_file,
                                                                                         args.const_relation_file)
        data_train = conllx_data.read_bucketed_data(train_path, word_alphabet, char_alphabet, pos_alphabet,
                                                    type_alphabet, symbolic_root=True,
                                                    constraint_pos_dict=constraint_pos_dict,
                                                    constraint_relation_dict=constraint_relation_dict)

        data_dev = conllx_data.read_data(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                         symbolic_root=True,
                                         constraint_pos_dict=constraint_pos_dict,
                                         constraint_relation_dict=constraint_relation_dict)

        data_test = conllx_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                          symbolic_root=True,
                                          constraint_pos_dict=constraint_pos_dict,
                                          constraint_relation_dict=constraint_relation_dict)
        # cont=data_train[2].relation_dict
        # json.dump(cont, open("relation.train.json",'w'))
        # cont = data_dev[2].relation_dict
        # json.dump(cont, open("relation.dev.json", 'w'))
        # cont = data_test[2].relation_dict
        # json.dump(cont, open("relation.test.json", 'w'))
        # print("done")
        # exit()
    else:
        data_train = conllx_stacked_data.read_bucketed_data(train_path, word_alphabet, char_alphabet, pos_alphabet,
                                                            type_alphabet, prior_order=prior_order)
        data_dev = conllx_stacked_data.read_data(dev_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                 prior_order=prior_order)
        data_test = conllx_stacked_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                  prior_order=prior_order)
    num_data = sum(data_train[1])
    logger.info("training: #training data: %d, batch: %d, unk replace: %.2f" % (num_data, batch_size, unk_replace))

    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    optimizer, scheduler = get_optimizer(network.parameters(), optim, learning_rate, lr_decay, betas, eps, amsgrad,
                                         weight_decay, warmup_steps)

    best_ucorrect_nopunc = 0.0
    best_lcorrect_nopunc = 0.0

    patient = 0
    beam = args.beam
    reset = args.reset
    num_batches = num_data // batch_size + 1
    if optim == 'adam':
        opt_info = 'adam, betas=(%.1f, %.3f), eps=%.1e, amsgrad=%s' % (betas[0], betas[1], eps, amsgrad)
    else:
        opt_info = 'sgd, momentum=0.9, nesterov=True'
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train_loss = 0.
        train_arc_loss = 0.
        train_type_loss = 0.
        num_insts = 0
        num_words = 0
        num_back = 0
        num_nans = 0
        network.train()
        lr = scheduler.get_lr()[0]
        print('Epoch %d (%s, lr=%.6f, lr decay=%.7f, grad clip=%.1f, l2=%.1e): ' % (epoch, opt_info, lr, lr_decay,
                                                                                    grad_clip, weight_decay))
        if args.cuda:
            torch.cuda.empty_cache()
        gc.collect()
        idx =0
        for step, data in enumerate(
                iterate_data(data_train, batch_size, bucketed=True, unk_replace=unk_replace, shuffle=True)):
            # if idx>20:
            #     break
            # else:
            #     idx+=1
            optimizer.zero_grad()
            words = data['WORD'].to(device)
            chars = data['CHAR'].to(device)
            postags = data['POS'].to(device)
            heads = data['HEAD'].to(device)
            types = data['TYPE'].to(device)
            masks = data['MASK'].to(device)
            nbatch = words.size(0)
            nwords = masks.sum() - nbatch

            if model_type == 'SampledMST':
                loss_arc, loss_type = network.sampled_loss(words, chars, postags, heads, types, mask=masks)
                # loss_arc, loss_type = network.loss(words, chars, postags, heads, types, mask=masks,
                #                                    constraint_pos_msk=constraint_pos_masks)
            if model_type == 'NeuroMST':
                loss_arc, loss_type = network.loss(words, chars, postags, heads, types, mask=masks)
            elif model_type == 'ConstraintDeepBiAffine':
                constraint_pos_masks = data['CONSTRAINT_POS_MASK'].to(device)
                con_relation_masks = data['CON_RELATION_MASK'].to(device)

                loss_arc, loss_type = network.loss(words, chars, postags, heads, types, mask=masks,
                                                   constraint_pos_msk=constraint_pos_masks,
                                                   con_relation_masks=con_relation_masks)
                # print(loss_arc)
            elif model_type == 'DeepBiAffine':

                loss_arc, loss_type = network.loss(words, chars, postags, heads, types, mask=masks)
            loss_arc = loss_arc.sum()
            loss_type = loss_type.sum()
            loss_total = loss_arc + loss_type
            if loss_ty_token:
                loss = loss_total.div(nwords)
            else:
                loss = loss_total.div(nbatch)
            loss.backward()
            if grad_clip > 0:
                grad_norm = clip_grad_norm_(network.parameters(), grad_clip)
            else:
                grad_norm = total_grad_norm(network.parameters())

            if math.isnan(grad_norm):
                num_nans += 1
            else:
                optimizer.step()
                scheduler.step()

                with torch.no_grad():
                    num_insts += nbatch
                    num_words += nwords
                    train_loss += loss_total.item()
                    train_arc_loss += loss_arc.item()
                    train_type_loss += loss_type.item()
            # update log
            if step % 10 == 0:
                torch.cuda.empty_cache()
                sys.stdout.write("\b" * num_back)
                sys.stdout.write(" " * num_back)
                sys.stdout.write("\b" * num_back)
                curr_lr = scheduler.get_lr()[0]
                num_insts = max(num_insts, 1)
                num_words = max(num_words, 1)
                log_info = '[%d/%d (%.0f%%) lr=%.6f (%d)] loss: %.4f (%.4f), arc: %.4f (%.4f), type: %.4f (%.4f)' % (
                    step, num_batches, 100. * step / num_batches, curr_lr, num_nans,
                    train_loss / num_insts, train_loss / num_words,
                    train_arc_loss / num_insts, train_arc_loss / num_words,
                    train_type_loss / num_insts, train_type_loss / num_words)
                sys.stdout.write(log_info)
                sys.stdout.flush()
                num_back = len(log_info)

        sys.stdout.write("\b" * num_back)
        sys.stdout.write(" " * num_back)
        sys.stdout.write("\b" * num_back)
        print('total: %d (%d), loss: %.4f (%.4f), arc: %.4f (%.4f), type: %.4f (%.4f), time: %.2fs nan: %d' % (
            num_insts, num_words, train_loss / num_insts, train_loss / num_words,
            train_arc_loss / num_insts, train_arc_loss / num_words,
            train_type_loss / num_insts, train_type_loss / num_words,
            time.time() - start_time, num_nans))
        print('-' * 120)

        evaluate_on_dev(result_path, epoch, pred_writer, gold_writer, model_type, data_dev, data_test, network,
                        punct_set,
                        word_alphabet, pos_alphabet, device, beam,
                        best_ucorrect_nopunc, best_lcorrect_nopunc, model_name, patient)
        if patient >= reset:
            logger.info('reset optimizer momentums')
            network.load_state_dict(torch.load(model_name, map_location=device))
            scheduler.reset_state()
            patient = 0


def parse(args):
    logger = get_logger("Parsing")
    args.cuda = torch.cuda.is_available()
    device = torch.device('cuda', 0) if args.cuda else torch.device('cpu')
    test_path = args.test

    model_path = args.model_path
    model_name = os.path.join(model_path, 'model.pt')
    punctuation = args.punctuation
    print(args)

    logger.info("Creating Alphabets")
    alphabet_path = os.path.join(model_path, 'alphabets')
    assert os.path.exists(alphabet_path)
    word_alphabet, char_alphabet, pos_alphabet, type_alphabet = conllx_data.create_alphabets(alphabet_path, None)

    num_words = word_alphabet.size()
    num_chars = char_alphabet.size()
    num_pos = pos_alphabet.size()
    num_types = type_alphabet.size()

    logger.info("Word Alphabet Size: %d" % num_words)
    logger.info("Character Alphabet Size: %d" % num_chars)
    logger.info("POS Alphabet Size: %d" % num_pos)
    logger.info("Type Alphabet Size: %d" % num_types)

    result_path = os.path.join(model_path, 'tmp')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    punct_set = None
    if punctuation is not None:
        punct_set = set(punctuation)
        logger.info("punctuations(%d): %s" % (len(punct_set), ' '.join(punct_set)))

    logger.info("loading network...")
    hyps = json.load(open(os.path.join(model_path, 'config.json'), 'r'))
    model_type = hyps['model']
    assert model_type in ['DeepBiAffine', 'NeuroMST', 'StackPtr']
    word_dim = hyps['word_dim']
    char_dim = hyps['char_dim']
    use_pos = hyps['pos']
    pos_dim = hyps['pos_dim']
    mode = hyps['rnn_mode']
    hidden_size = hyps['hidden_size']
    arc_space = hyps['arc_space']
    type_space = hyps['type_space']
    p_in = hyps['p_in']
    p_out = hyps['p_out']
    p_rnn = hyps['p_rnn']
    activation = hyps['activation']
    prior_order = None

    alg = 'transition' if model_type == 'StackPtr' else 'graph'
    if model_type == 'DeepBiAffine':
        num_layers = hyps['num_layers']
        network = DeepBiAffine(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                               mode, hidden_size, num_layers, num_types, arc_space, type_space,
                               p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
    elif model_type == 'NeuroMST':
        num_layers = hyps['num_layers']
        network = NeuroMST(word_dim, num_words, char_dim, num_chars, pos_dim, num_pos,
                           mode, hidden_size, num_layers, num_types, arc_space, type_space,
                           p_in=p_in, p_out=p_out, p_rnn=p_rnn, pos=use_pos, activation=activation)
    else:
        raise RuntimeError('Unknown model type: %s' % model_type)

    network = network.to(device)
    network.load_state_dict(torch.load(model_name, map_location=device))
    model = "{}-{}".format(model_type, mode)
    logger.info("Network: %s, num_layer=%s, hidden=%d, act=%s" % (model, num_layers, hidden_size, activation))

    logger.info("Reading Data")
    if alg == 'graph':
        data_test = conllx_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                          symbolic_root=True)
    else:
        data_test = conllx_stacked_data.read_data(test_path, word_alphabet, char_alphabet, pos_alphabet, type_alphabet,
                                                  prior_order=prior_order)

    beam = args.beam
    pred_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    gold_writer = CoNLLXWriter(word_alphabet, char_alphabet, pos_alphabet, type_alphabet)
    pred_filename = os.path.join(result_path, 'pred.txt')
    pred_writer.start(pred_filename)
    gold_filename = os.path.join(result_path, 'gold.txt')
    gold_writer.start(gold_filename)

    with torch.no_grad():
        print('Parsing...')
        start_time = time.time()
        eval(alg, data_test, network, pred_writer, gold_writer, punct_set, word_alphabet, pos_alphabet, device, beam,
             batch_size=args.batch_size)
        print('Time: %.2fs' % (time.time() - start_time))

    pred_writer.close()
    gold_writer.close()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description='Tuning with graph-based parsing')
    args_parser.add_argument('--mode', choices=['train', 'parse'], required=True, help='processing mode')
    args_parser.add_argument('--config', type=str, help='config file')
    args_parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    args_parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
    args_parser.add_argument('--loss_type', choices=['sentence', 'token'], default='sentence',
                             help='loss type (default: sentence)')
    args_parser.add_argument('--optim', choices=['sgd', 'adam'], help='type of optimizer')
    args_parser.add_argument('--learning_rate', type=float, default=0.1, help='Learning rate')
    args_parser.add_argument('--beta1', type=float, default=0.9, help='beta1 of Adam')
    args_parser.add_argument('--beta2', type=float, default=0.999, help='beta2 of Adam')
    args_parser.add_argument('--eps', type=float, default=1e-8, help='epsilon for adam or adamax')
    args_parser.add_argument('--lr_decay', type=float, default=0.999995, help='Decay rate of learning rate')
    args_parser.add_argument('--amsgrad', action='store_true', help='AMS Grad')
    args_parser.add_argument('--grad_clip', type=float, default=0,
                             help='max norm for gradient clip (default 0: no clip')
    args_parser.add_argument('--warmup_steps', type=int, default=0, metavar='N',
                             help='number of steps to warm up (default: 0)')
    args_parser.add_argument('--reset', type=int, default=10, help='Number of epochs to reset optimizer (default 10)')
    args_parser.add_argument('--weight_decay', type=float, default=0.0, help='weight for l2 norm decay')
    args_parser.add_argument('--unk_replace', type=float, default=0.,
                             help='The rate to replace a singleton word with UNK')
    args_parser.add_argument('--freeze', action='store_true', help='frozen the word embedding (disable fine-tuning).')
    args_parser.add_argument('--punctuation', nargs='+', type=str, help='List of punctuations')
    args_parser.add_argument('--beam', type=int, default=1, help='Beam size for decoding')
    args_parser.add_argument('--word_embedding', choices=['glove', 'senna', 'sskip', 'polyglot'],
                             help='Embedding for words')
    args_parser.add_argument('--word_path', help='path for word embedding dict')
    args_parser.add_argument('--char_embedding', choices=['random', 'polyglot'], help='Embedding for characters')
    args_parser.add_argument('--char_path', help='path for character embedding dict')
    args_parser.add_argument('--train', help='path for training file.')
    args_parser.add_argument('--dev', help='path for dev file.')
    args_parser.add_argument('--test', help='path for test file.', required=True)
    args_parser.add_argument('--const_file', help='path for constraint on dependency grammar file.',
                             default='/home/jiangnanhugo/PycharmProjects/wilson/dataset/UD_English-EWT/upos.heads.json',
                             required=False)
    args_parser.add_argument('--const_relation_file', help='path for constraint on labels dependency grammar file.',
                             default='/home/jiangnanhugo/PycharmProjects/wilson/dataset/UD_English-EWT/upos.relation.json',
                             required=False)
    args_parser.add_argument('--model_path', help='path for saving model file.', required=True)

    args = args_parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        parse(args)
