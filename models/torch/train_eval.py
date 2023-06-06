from __future__ import print_function, division
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import itertools

from utils import *
from train_langmod import *
from operator import truediv
import pandas as pd
import json
import spot

SOS_token = 0
EOS_token = 1
UNK_token = 2

use_cuda = torch.cuda.is_available()
import pathlib

if use_cuda:
    device = torch.device("cuda")


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(input_lang, output_lang, pair):
    input_variable = variableFromSentence(input_lang, ' '.join(list(reversed(pair[0].split()))))
    target_variable = variableFromSentence(output_lang, pair[1])
    return input_variable, target_variable

teacher_forcing_ratio = 0.5



def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(in_lang, out_lang, encoder, decoder, samples, n_iters, max_length, 
                print_every=1000, plot_every=10000, learning_rate=10**-4, fold=None, epochs = 10, val_samples = None,
                starting_iter = 0, starting_epoch = 0, starting_epoch_loss = None): #0.000001
    start = time.time()
    plot_losses = []
    x_losses = []

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = itertools.cycle(iter([variablesFromPair(in_lang, out_lang, s) for s in samples]))
    criterion = nn.NLLLoss()

    epoch_losses = []
    x_epoch_losses = []
    epoch = starting_epoch
    
    p = pathlib.Path("../checkpoints")
    p.mkdir(parents=True, exist_ok=True)

    isStart =  True
    while epoch < epochs:
        # print("starting epoch")
        encoder.train()
        decoder.train()

        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        i = starting_iter + 1
        
        print(n_iters, i)
        if isStart:
            i = starting_iter + 1
            isStart =  False
        else:
            i = 1

        print("epoch", epoch, "i", i)
        
        while i < n_iters:
            # print("i", i)
            # exit()
        # for i in range(1, n_iters + 1):
            # if i == 143000:
                # break
            training_pair = next(training_pairs)
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            loss = train(input_variable, target_variable, encoder,
                        decoder, encoder_optimizer, decoder_optimizer, criterion, max_length)
            print_loss_total += loss
            plot_loss_total += loss

            if i % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iters),
                                            i, i / n_iters * 100, print_loss_avg))
            # print("i",i, "mod", i % print_every, i % plot_every)
            # print(print_every, plot_every)
            # plot_losses.append(plot_loss_avg)
            # x_losses.append(epoch*n_iters+ i)
            if i % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                x_losses.append(epoch*n_iters+ i)
                plot_loss_total = 0
                print("plot losses are, ",plot_losses)
                showPlot([x_losses,plot_losses], "training.png")
                fn = f"encoder_e{epoch}.pt" # save last epoch
                filepath = p / fn
                torch.save(encoder, filepath)

                fn = f"encoder.pt" # save last epoch
                filepath = p / fn
                torch.save(encoder, filepath)

                fn = f"decoder_e{epoch}.pt" # I don't know what is your fn
                filepath = p / fn
                torch.save(decoder, filepath)

                fn = f"decoder.pt" # I don't know what is your fn
                filepath = p / fn
                torch.save(decoder, filepath)

                
                fn = "meta_information.txt" # I don't know what is your fn
                filepath = p / fn
                with filepath.open("a", encoding ="utf-8") as f:
                    f.writelines(f"fold: {fold},epoch: {epoch}, iter: {i}, train loss: {plot_losses[-1]}\n")

                fn = "meta_information.json" # I don't know what is your fn
                filepath = p / fn
                
                dict = {
                    "fold": fold, 
                    "epoch":epoch,
                    "iter": i , 
                    "train_loss": plot_losses[-1]
                }

                # Serializing json
                json_object = json.dumps(dict, indent=4)
                
                with filepath.open("w", encoding ="utf-8") as fp:
                    fp.write(json_object)
            i+=1
        
        # plot_loss_avg = plot_loss_total / plot_every
        # plot_losses.append(plot_loss_avg)
        # x_losses.append(epoch*n_iters+ i)
        # plot_loss_total = 0
        
        fn = f"encoder_e{epoch}.pt" # I don't know what is your fn
        filepath = p / fn
        torch.save(encoder, filepath)

        fn = f"encoder.pt" # I don't know what is your fn
        filepath = p / fn
        torch.save(encoder, filepath)
        
        fn = f"decoder_e{epoch}.pt" # I don't know what is your fn
        filepath = p / fn
        torch.save(decoder, filepath)

        fn = f"decoder.pt" # I don't know what is your fn
        filepath = p / fn
        torch.save(decoder, filepath)


        print("starting validation loss calculation")
        encoder.eval()
        decoder.eval()
        corr, tot, acc, loss = evaluateSamples(in_lang, out_lang, encoder, decoder, val_samples, max_length, criterion = criterion)
        print('Fold #{0}, Epoch # {4} ,Val Accuracy: {1}/{2} = {3}%'.format(fold + 1, corr, tot, 100. * acc, epoch))
        print('Fold #{0}, Epoch # {2},Val Loss: {1}'.format(fold + 1, loss, epoch))
        epoch_losses.append(loss)
        x_epoch_losses.append((epoch+1)*n_iters)

        fn = "meta_information.txt" # I don't know what is your fn
        filepath = p / fn
        with filepath.open("a", encoding ="utf-8") as f:
            f.writelines(f"fold: {fold}, epoch: {epoch}, iter: {i} , epoch loss: {epoch_losses[-1]}\n")
        
        fn = "meta_information.json" # I don't know what is your fn
        filepath = p / fn
        
        if len(plot_losses) == 0:
            dict = {
            "fold": fold, 
            "epoch":epoch,
            "iter": i , 
            "epoch_loss": starting_epoch_loss
            }
        else:
            dict = {
                "fold": fold, 
                "epoch":epoch,
                "iter": i , 
                "epoch_loss": plot_losses[-1]
            }

        # Serializing json
        json_object = json.dumps(dict, indent=4)
        
        with filepath.open("w", encoding ="utf-8") as fp:
            fp.write(json_object)
        # correct += corr
        # total += tot
        plot = []
        plot.append([x_losses, plot_losses])
        plot.append([x_epoch_losses, epoch_losses])
        print("plot_losses", plot_losses)
        print("x_losses", x_losses)
        print("epoch_losses", epoch_losses)
        print("x_epoch_losses", x_epoch_losses)
        showPlot(plot, "epoch.jpg", is_multiple=True)
        print("ended the epoch ")
        epoch+=1

    return criterion


def evaluate(input_lang, output_lang, encoder, decoder, sentence, max_length, criterion=None, target_ltl = None):
    input_variable = variableFromSentence(input_lang, ' '.join(list(reversed(sentence.split()))))
    if target_ltl != None:
        target_variable = variableFromSentence(output_lang, target_ltl)
        # print(target_variable)
        target_length = target_variable.size()[0]
        # print(target_length)

    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = torch.tensor([0.0]).to(device)
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        # print("index is", di)
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        if decoder_attention is not None:
            decoder_attentions[di] = decoder_attention.data

        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni.item()])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        if di < target_length:
            if criterion != None:
                loss += criterion(decoder_output, target_variable[di])

    # print("max_length is", max_length)
    # if loss == None:
        # return(decoded_words, decoder_attentions[:di + 1], None)
    # else:
    return decoded_words, decoder_attentions[:di + 1], loss.item() / target_length


def evaluate2(input_lang, output_lang, encoder, decoder, sentence, max_length):
    input_variable = variableFromSentence(input_lang, ' '.join(list(reversed(sentence.split()))))
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_hidden = encoder_hidden

    candidates = [([SOS_token], decoder_hidden, 0.0)]
    beam_width = 10
    completed = []

    while len(candidates) > 0:
        new_candidates = []
        for i in range(len(candidates)):
            seq, hidden, score = candidates[i]

            if seq[-1] == EOS_token or len(seq) == max_length:
                continue

            decoder_input = Variable(torch.LongTensor([[seq[-1]]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, hidden, encoder_outputs)

            topv, topi = decoder_output.data.topk(decoder_output.size()[-1])
            # print(('Input token: ', output_lang.index2word[seq[-1]]))
            # print(topi[0].numpy())
            for v, i in zip(topv[0], topi[0]):
                new_candidates.append((seq + [i], decoder_hidden, score + v))
        candidates = sorted(new_candidates, key=lambda x: x[2], reverse=True)
        # print([([output_lang.index2word[i] for i in x], z) for x, y, z in candidates])
        # print()

        candidates = candidates[:beam_width]
        next_candidates = []
        for x, y, z in candidates:
            if x[-1] == EOS_token:
                if len(completed) == 0:
                    completed.append((x, z))
                else:
                    if z > completed[0][1]:
                        completed = [(x, z)]
            else:
                next_candidates.append((x, y, z))

        candidates = next_candidates
        # print([([output_lang.index2word[i] for i in x], z) for x, y, z in candidates])
        # print()
        # print([([output_lang.index2word[i] for i in x[0]], x[1]) for x in completed])
        # raw_input()

    completed = sorted(completed, key=lambda x: x[1], reverse=True)
    decoded_words = [output_lang.index2word[i] for i in completed[0][0]]

    return decoded_words[1:], None


def evaluateRandomly(input_lang, output_lang, encoder, decoder, p, max_length, n=10):
    print("length of p", len(p["valid"]))
    valid_iter = p["valid"][0]
    print("length of valid_iter", len(valid_iter))
    # exit()
    # random.seed(10)
    for i in range(n):
        pair = random.choice(valid_iter)
        # pair = p[0]
        print('>', pair[0])
        print('=', pair[1])
        # exit()
        # output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, pair[0], max_length)
        # output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, pair[0], max_length)
        output_words, attentions, current_loss = evaluate(input_lang, output_lang, encoder, decoder, pair[0], max_length, target_ltl = pair[1])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


def evaluateTraining(input_lang, output_lang, encoder, decoder, pairs, max_length):
    corr, tot = 0, 0
    for p in pairs:
        output_words, attentions = evaluate(input_lang, output_lang, encoder, decoder, p[0], max_length)
        output_words = ' '.join(output_words[:-1])
        if output_words == p[1]:
            corr += 1
        # print('Input: {0}\tOutput: {1}\tExpected:{2}'.format(p[0], output_words, p[1]))
        tot += 1

    print('Training Accuracy: {0}/{1} = {2}%'.format(corr, tot, corr / tot))


def evaluateSamples(input_lang, output_lang, encoder, decoder, samples, max_length, criterion=None):
    corr, tot, loss = 0, 0, 0
    for p in samples:
        output_words, attentions, current_loss = evaluate(input_lang, output_lang, encoder, decoder, p[0], max_length, criterion = criterion, target_ltl = p[1])
        output_words = ' '.join(output_words[:-1])
        loss+=current_loss
        # print((output_words, p[1]))
        if output_words == p[1]:
            corr += 1
        # else:
            # print((p[0], output_words, p[1]))
        tot += 1
    loss = loss/len(samples)
    return corr, tot, corr / tot, loss


def resetWeights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def crossValidation(in_lang, out_lang, encoder, decoder, samples, max_length, n_folds=5, lang2ltl = False, is_load = False, subset = False):
    correct, total = 0, 0
    if not lang2ltl:
        for _ in range(10):
            random.shuffle(samples)
        fold_range = list(range(0, len(samples), int(len(samples) / n_folds)))
        fold_range.append(len(samples))


    if is_load:
        # print("loading!")
        p = pathlib.Path("../checkpoints")
        fn = "meta_information.json" # I don't know what is your fn
        filepath = p / fn
 
        # Opening JSON file
        with filepath.open("r", encoding ="utf-8") as fp:
        # with open('sample.json', 'r') as openfile:
        
            # Reading from json file
            json_object = json.load(fp)
 
        print(json_object)
        # print(type(json_object))
        starting_fold = json_object["fold"]
        starting_iter = json_object["iter"]
        starting_epoch = json_object["epoch"]
        starting_epoch_loss = json_object["epoch_loss"]
        print("Starting epoch is", starting_epoch)
        # exit()
    else:
        starting_fold = 0
        starting_iter = 0
        starting_epoch = 0
        starting_epoch_loss = None

    print('Starting {0}-fold cross validation'.format(n_folds))
    per_fold_accuracy = []
    columns = [f"Fold num #{i}" for i in range(1)]
    columns.append("Mean")
    columns.append("Std Dev")
    df = pd.DataFrame(columns = columns)
    
    f = starting_fold
    while f < n_folds:
    # for f in range(n_folds):
        if f+1 > 1:
            print("here")
            break
        a = list(range(n_folds))
        print('Running cross validation fold {0}/{1}...'.format(f + 1, n_folds))


        encoder.train()
        decoder.train()

        if lang2ltl:
            if subset:
                train_samples = []
                subset_samples = random.choices(samples["train"][0], k=857)
                # print(len(subset_samples))
                fold_range = list(range(0, len(subset_samples), int(len(subset_samples) / n_folds)))
                fold_range.append(len(subset_samples))
                # print(type(samples))
                # print("f is:", f)
                # print("fold is:", fold_range)
                train_samples = subset_samples[:fold_range[f]] + subset_samples[fold_range[f + 1]:] ## train on 4 
                val_samples = subset_samples[fold_range[f]:fold_range[f + 1]]

            else:
                train_samples = []
                a.remove(f)
                for fold in a:
                    train_samples.extend(samples["train"][fold]) #samples[f][0]

                val_samples = samples["valid"][f] #samples[f][1]
        else:
            train_samples = samples[:fold_range[f]] + samples[fold_range[f + 1]:] ## train on 4 
            val_samples = samples[fold_range[f]:fold_range[f + 1]]

        # print(len(train_samples))
        # print(len(val_samples))
        # exit()
        # learning_rate=0.00001
        # plot_every=20000
        criterion = trainIters(in_lang, out_lang, encoder, decoder, train_samples, 38930*4, max_length, 
                                print_every=1000, plot_every=20000, fold=f, val_samples=val_samples,
                                starting_iter= starting_iter, starting_epoch=starting_epoch, starting_epoch_loss = starting_epoch_loss)

        encoder.eval()
        decoder.eval()

        corr, tot, acc, loss = evaluateSamples(in_lang, out_lang, encoder, decoder, val_samples, max_length, criterion= criterion)
        print('Cross validation fold #{0} Accuracy: {1}/{2} = {3}%'.format(f + 1, corr, tot, 100. * acc))
        correct += corr
        total += tot

        per_fold_accuracy.append(corr/tot * 100.)

        encoder.apply(resetWeights)
        decoder.apply(resetWeights)
        f+=1

    mean_accuracy = 100. * correct / total
    std_accuracy = np.std(per_fold_accuracy)
    print('Average {0}-fold Cross Validation Accuracy: {1}/{2} = {3}%'.format(n_folds, correct, total, mean_accuracy))
    print('{0}-fold Cross Validation Standard Deviation : {1}%'.format(n_folds, std_accuracy))
    
    # print(per_fold_accuracy, type(per_fold_accuracy))
    # print(mean_accuracy, type(mean_accuracy))
    # print(std_accuracy, type(std_accuracy))
    per_fold_accuracy.append(mean_accuracy)
    per_fold_accuracy.append(std_accuracy)
    df.loc[len(df.index)] = per_fold_accuracy
    df.to_csv("../../results.csv")
    return train_samples, val_samples


def write_train_vs_test_hidden_params(in_lang, out_lang, encoder, decoder, train_samples, eval_samples, max_length):


    with open('train1.csv', 'w') as fh:
        decoder.testing_mode(fh)
        evaluateSamples(in_lang, out_lang, encoder, decoder, train_samples, max_length)

    with open('test1.csv', 'w') as fh:
        decoder.testing_mode(fh)
        evaluateSamples(in_lang, out_lang, encoder, decoder, eval_samples, max_length)



def createTrainingData(samples, in_lang, out_lang):
    x = 1





def testClassifier(in_lang, out_lang, encoder, decoder, samples, perc, max_length):
    createTrainingData(samples, in_lang, out_lang)

def evalGeneralization(in_lang, out_lang, encoder, decoder, samples, perc, max_length):
    # for _ in range(10):
    #     random.shuffle(samples)

    encoder.train()
    decoder.train()

    tar_set = list(set([s[1] for s in samples]))
    tar_num = int(np.ceil(perc * len(tar_set)))
    train_forms = random.sample(tar_set, tar_num)
    print('GLTL Training Formulas: {0}'.format(train_forms))
    print('GLTL Evaluation Formulas: {0}'.format([s for s in tar_set if s not in train_forms]))
    train_samples = [s for s in samples if s[1] in train_forms]
    eval_samples = [s for s in samples if s[1] not in train_forms]

    print('Training with {0}/{3} unique GLTL formulas => {1} training samples | {2} testing samples'.format(tar_num, len(train_samples), len(eval_samples), len(tar_set)))
    trainIters(in_lang, out_lang, encoder, decoder, train_samples, 10000, max_length, print_every=10000)

    encoder.eval()
    decoder.eval()

    corr, tot, acc = evaluateSamples(in_lang, out_lang, encoder, decoder, eval_samples, max_length)



    # write_train_vs_test_hidden_params(in_lang, out_lang, encoder, decoder, train_samples, eval_samples, max_length)
    print('Held-out Accuracy: {0}/{1} = {2}%'.format(corr, tot, 100. * acc))
    return acc


def evalGeneralizationPT(in_lang, out_lang, encoder, decoder, langmod, samples, perc, max_length, train_data, batch_size, bptt):
    for _ in range(10):
        random.shuffle(samples)

    encoder.train()
    decoder.train()
    langmod.train()

    tar_set = list(set([s[1] for s in samples]))
    tar_num = int(np.ceil(perc * len(tar_set)))
    train_forms = random.sample(tar_set, tar_num)
    print('GLTL Training Formulas: {0}'.format(train_forms))
    print('GLTL Evaluation Formulas: {0}'.format([s for s in tar_set if s not in train_forms]))
    train_samples = [s for s in samples if s[1] in train_forms]
    eval_samples = [s for s in samples if s[1] not in train_forms]

    print('Training with {0}/{3} unique GLTL formulas => {1} training samples | {2} testing samples'.format(tar_num, len(train_samples), len(eval_samples), len(tar_set)))
    for _ in range(10):
        trainIters(in_lang, out_lang, encoder, decoder, train_samples, 1000, max_length, print_every=1000)
        langmod_train2(train_data, langmod, batch_size, bptt, 0, 2000, 0.01)
    trainIters(in_lang, out_lang, encoder, decoder, train_samples, 1000, max_length, print_every=1000)

    encoder.eval()
    decoder.eval()
    langmod.eval()

    corr, tot, acc = evaluateSamples(in_lang, out_lang, encoder, decoder, eval_samples, max_length)
    print('Held-out Accuracy: {0}/{1} = {2}%'.format(corr, tot, 100. * acc))
    return acc


def evalSampleEff(in_lang, out_lang, encoder, decoder, samples, perc, max_length):
    for _ in range(10):
        random.shuffle(samples)

    encoder.train()
    decoder.train()

    train_samples = samples[:int(perc * len(samples))]
    train_forms = set([s[1] for s in train_samples])
    eval_samples = samples[int(perc * len(samples)):]
    eval_forms = set([s[1] for s in eval_samples])
    print('Training with {0}/{1} random data samples'.format(len(train_samples), len(samples)))
    print('{1} Distinct GLTL formulas in training sample: {0}'.format(train_forms, len(train_forms)))
    print('{1} Distinct GLTL formulas in eval sample: {0}'.format(eval_forms, len(eval_forms)))
    trainIters(in_lang, out_lang, encoder, decoder, train_samples, 10000, max_length, print_every=10000)

    encoder.eval()
    decoder.eval()

    corr, tot, acc = evaluateSamples(in_lang, out_lang, encoder, decoder, eval_samples, max_length)
    print('Held-out Accuracy: {0}/{1} = {2}%'.format(corr, tot, 100. * acc))
    return acc