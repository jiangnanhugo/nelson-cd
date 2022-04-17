import os
import sys

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

import torch
from neuronlp2.io import conllx_data, iterate_data

from neuronlp2.tasks import parser


def evaluate_on_dev(result_path, epoch, pred_writer, gold_writer, model_type, data_dev, data_test, network, punct_set,
                    word_alphabet, pos_alphabet, device, beam,
                    best_ucorrect_nopunc, best_lcorrect_nopunc, model_name, patient):
    best_ucorrect = 0.0
    best_lcorrect = 0.0
    best_ucomlpete = 0.0
    best_lcomplete = 0.0

    best_ucomlpete_nopunc = 0.0
    best_lcomplete_nopunc = 0.0
    best_root_correct = 0.0
    best_total = 0
    best_total_nopunc = 0
    best_total_inst = 0
    best_total_root = 0

    best_epoch = 0

    test_ucorrect = 0.0
    test_lcorrect = 0.0
    test_ucomlpete = 0.0
    test_lcomplete = 0.0

    test_ucorrect_nopunc = 0.0
    test_lcorrect_nopunc = 0.0
    test_ucomlpete_nopunc = 0.0
    test_lcomplete_nopunc = 0.0
    test_root_correct = 0.0
    test_total = 0
    test_total_nopunc = 0
    test_total_inst = 0
    test_total_root = 0

    with torch.no_grad():
        pred_filename = os.path.join(result_path, 'pred_dev%d' % epoch)
        pred_writer.start(pred_filename)
        gold_filename = os.path.join(result_path, 'gold_dev%d' % epoch)
        gold_writer.start(gold_filename)

        print('Evaluating dev:')
        dev_stats, dev_stats_nopunct, dev_stats_root = eval(model_type, data_dev, network, pred_writer, gold_writer, punct_set,
                                                            word_alphabet, pos_alphabet, device, beam=beam)
        # print('Evaluating dev with constraint mask:')
        # dev_stats, dev_stats_nopunct, dev_stats_root = eval(model_type, data_dev, network, pred_writer, gold_writer,
        #                                                     punct_set,
        #                                                     word_alphabet, pos_alphabet, device, beam=beam, use_constraint_mask=True)

        pred_writer.close()
        gold_writer.close()

        dev_ucorr, dev_lcorr, dev_ucomlpete, dev_lcomplete, dev_total = dev_stats
        dev_ucorr_nopunc, dev_lcorr_nopunc, dev_ucomlpete_nopunc, dev_lcomplete_nopunc, dev_total_nopunc = dev_stats_nopunct
        dev_root_corr, dev_total_root, dev_total_inst = dev_stats_root

        if best_ucorrect_nopunc + best_lcorrect_nopunc < dev_ucorr_nopunc + dev_lcorr_nopunc:
            best_ucorrect_nopunc = dev_ucorr_nopunc
            best_lcorrect_nopunc = dev_lcorr_nopunc
            best_ucomlpete_nopunc = dev_ucomlpete_nopunc
            best_lcomplete_nopunc = dev_lcomplete_nopunc

            best_ucorrect = dev_ucorr
            best_lcorrect = dev_lcorr
            best_ucomlpete = dev_ucomlpete
            best_lcomplete = dev_lcomplete

            best_root_correct = dev_root_corr
            best_total = dev_total
            best_total_nopunc = dev_total_nopunc
            best_total_root = dev_total_root
            best_total_inst = dev_total_inst

            best_epoch = epoch
            patient = 0
            torch.save(network.state_dict(), model_name)

            pred_filename = os.path.join(result_path, 'pred_test%d' % epoch)
            pred_writer.start(pred_filename)
            gold_filename = os.path.join(result_path, 'gold_test%d' % epoch)
            gold_writer.start(gold_filename)

            print('Evaluating test:')
            test_stats, test_stats_nopunct, test_stats_root = eval(model_type, data_test, network, pred_writer, gold_writer,
                                                                   punct_set, word_alphabet, pos_alphabet, device,
                                                                   beam=beam)


            test_ucorrect, test_lcorrect, test_ucomlpete, test_lcomplete, test_total = test_stats
            test_ucorrect_nopunc, test_lcorrect_nopunc, test_ucomlpete_nopunc, test_lcomplete_nopunc, test_total_nopunc = test_stats_nopunct
            test_root_correct, test_total_root, test_total_inst = test_stats_root

            pred_writer.close()
            gold_writer.close()
        else:
            patient += 1

        print('-' * 125)
        print(
            'best dev  W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                best_ucorrect, best_lcorrect, best_total, best_ucorrect * 100 / best_total,
                best_lcorrect * 100 / best_total, best_ucomlpete * 100 / dev_total_inst,
                best_lcomplete * 100 / dev_total_inst, best_epoch))
        print(
            'best dev  Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                best_ucorrect_nopunc, best_lcorrect_nopunc, best_total_nopunc,
                best_ucorrect_nopunc * 100 / best_total_nopunc, best_lcorrect_nopunc * 100 / best_total_nopunc,
                best_ucomlpete_nopunc * 100 / best_total_inst, best_lcomplete_nopunc * 100 / best_total_inst,
                best_epoch))
        print('best dev  Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
            best_root_correct, best_total_root, best_root_correct * 100 / best_total_root, best_epoch))
        print('-' * 125)
        print(
            'best test W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                test_ucorrect, test_lcorrect, test_total, test_ucorrect * 100 / test_total,
                test_lcorrect * 100 / test_total,
                test_ucomlpete * 100 / test_total_inst, test_lcomplete * 100 / test_total_inst,
                best_epoch))
        print(
            'best test Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%% (epoch: %d)' % (
                test_ucorrect_nopunc, test_lcorrect_nopunc, test_total_nopunc,
                test_ucorrect_nopunc * 100 / test_total_nopunc, test_lcorrect_nopunc * 100 / test_total_nopunc,
                test_ucomlpete_nopunc * 100 / test_total_inst, test_lcomplete_nopunc * 100 / test_total_inst,
                best_epoch))
        print('best test Root: corr: %d, total: %d, acc: %.2f%% (epoch: %d)' % (
            test_root_correct, test_total_root, test_root_correct * 100 / test_total_root, best_epoch))
        print('=' * 125)


def eval(model_type, data, network, pred_writer, gold_writer, punct_set, word_alphabet, pos_alphabet, device, beam=1,
         batch_size=1):
    network.eval()
    accum_ucorr = 0.0
    accum_lcorr = 0.0
    accum_total = 0.001
    accum_ucomlpete = 0.0
    accum_lcomplete = 0.0
    accum_ucorr_nopunc = 0.0
    accum_lcorr_nopunc = 0.0
    accum_total_nopunc = 0.001
    accum_ucomlpete_nopunc = 0.0
    accum_lcomplete_nopunc = 0.0
    accum_root_corr = 0.0
    accum_total_root = 0.00001
    accum_total_inst = 0.00001
    accum_by_len = {}
    for i in range(0, 80, 20):
        accum_by_len[(i, i+20)] = dict()

    for x in accum_by_len:
        accum_by_len[x]['uas'] = 0
        accum_by_len[x]['las'] = 0
        accum_by_len[x]['len'] = 0
    for data in iterate_data(data, batch_size):
        words = data['WORD'].to(device)
        chars = data['CHAR'].to(device)
        postags = data['POS'].to(device)
        heads = data['HEAD'].numpy()
        types = data['TYPE'].numpy()
        lengths = data['LENGTH'].numpy()

        if model_type == 'ConstraintDeepBiAffine':
            constraint_pos_masks = data['CONSTRAINT_POS_MASK'].to(device)
            masks = data['MASK'].to(device)
            heads_pred, types_pred = network.decode_local(words, chars, postags, mask=masks,
                                                          leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS,
                                                          constraint_pos_msk=constraint_pos_masks)
        elif model_type == 'SampledMST':
            # constraint_pos_masks = data['CONSTRAINT_POS_MASK'].to(device)
            masks = data['MASK'].to(device)
            heads_pred, types_pred = network.decode(words, chars, postags, mask=masks,
                                                          leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
        elif model_type == 'NeuroMST':
            # constraint_pos_masks = data['CONSTRAINT_POS_MASK'].to(device)
            masks = data['MASK'].to(device)
            heads_pred, types_pred = network.decode(words, chars, postags, mask=masks,
                                                          leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)
        elif model_type == 'DeepBiAffine':
            masks = data['MASK'].to(device)
            heads_pred, types_pred = network.decode_local(words, chars, postags, mask=masks,
                                                          leading_symbolic=conllx_data.NUM_SYMBOLIC_TAGS)

        words = words.cpu().numpy()
        postags = postags.cpu().numpy()
        pred_writer.write(words, postags, heads_pred, types_pred, lengths, symbolic_root=True)
        gold_writer.write(words, postags, heads, types, lengths, symbolic_root=True)

        stats, stats_nopunc, stats_root, num_inst = parser.eval(words, postags, heads_pred, types_pred, heads, types,
                                                                word_alphabet, pos_alphabet, lengths,
                                                                punct_set=punct_set, symbolic_root=True)
        ucorr, lcorr, total, ucm, lcm = stats
        ucorr_nopunc, lcorr_nopunc, total_nopunc, ucm_nopunc, lcm_nopunc = stats_nopunc
        corr_root, total_root = stats_root

        accum_ucorr += ucorr
        accum_lcorr += lcorr
        accum_total += total
        accum_ucomlpete += ucm
        accum_lcomplete += lcm

        accum_ucorr_nopunc += ucorr_nopunc
        accum_lcorr_nopunc += lcorr_nopunc
        accum_total_nopunc += total_nopunc
        accum_ucomlpete_nopunc += ucm_nopunc
        accum_lcomplete_nopunc += lcm_nopunc

        accum_root_corr += corr_root
        accum_total_root += total_root

        accum_total_inst += num_inst
        for idx, lens in enumerate(lengths):
            for x in accum_by_len:
                if x[0] < lens <= x[1]:
                    accum_by_len[x]['uas']+=ucorr
                    accum_by_len[x]['las']+=lcorr
                    # accum_by_len[x]['len'] += 1
                    accum_by_len[x]['len'] += total

    for x in accum_by_len:
        print("{}  uas: {} {:.4f} las: {} {:.4f}".format(x,
                                          int(accum_by_len[x]['uas']), (accum_by_len[x]['uas'])/(0.0001+accum_by_len[x]['len']),
                                          int(accum_by_len[x]['las']), (accum_by_len[x]['las'])/(0.0001+accum_by_len[x]['len'])),
              end="    ")
    print()

    print('W. Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        accum_ucorr, accum_lcorr, accum_total, accum_ucorr * 100 / accum_total, accum_lcorr * 100 / accum_total,
        accum_ucomlpete * 100 / accum_total_inst, accum_lcomplete * 100 / accum_total_inst))
    print('Wo Punct: ucorr: %d, lcorr: %d, total: %d, uas: %.2f%%, las: %.2f%%, ucm: %.2f%%, lcm: %.2f%%' % (
        accum_ucorr_nopunc, accum_lcorr_nopunc, accum_total_nopunc, accum_ucorr_nopunc * 100 / accum_total_nopunc,
        accum_lcorr_nopunc * 100 / accum_total_nopunc,
        accum_ucomlpete_nopunc * 100 / accum_total_inst, accum_lcomplete_nopunc * 100 / accum_total_inst))
    print('Root: corr: %d, total: %d, acc: %.2f%%' % (
    accum_root_corr, accum_total_root, accum_root_corr * 100 / accum_total_root))
    return (accum_ucorr, accum_lcorr, accum_ucomlpete, accum_lcomplete, accum_total), \
           (accum_ucorr_nopunc, accum_lcorr_nopunc, accum_ucomlpete_nopunc, accum_lcomplete_nopunc, accum_total_nopunc), \
           (accum_root_corr, accum_total_root, accum_total_inst)
