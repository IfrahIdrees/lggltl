from __future__ import print_function, division
import sys
import copy

from lang import *
from networks import *
from train_eval import *
from train_langmod import *
import argparse
import re

def parseArguments():

    subparsers = parser.add_subparsers(dest='dataset')
    subparsers.required = True  # required since 3.7

    #  subparser for lggltl dataset
    parser_dump = subparsers.add_parser('lggltl')
    parser_dump.add_argument(
        "--src_file_path", type=str, default="../../data/hard_pc_src.txt",
        help="src path")
    parser_dump.add_argument(
        "--tar_file_path", type=str, default="../../data/hard_pc_tar.txt",
        help="src path")

    #  subparser for lang2ltl dataset
    parser_upload = subparsers.add_parser('lang2ltl')
    parser_upload.add_argument(
        "--src_dir_path", type=str, default="../../data/osm/lang2ltl/boston/",
        help="src path")
    parser_upload.add_argument(
        "--is_load", type=bool, default=False,
        help="is_load")
    
    # is_load = False
    
    args = parser.parse_args()
    return parser, args

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")

parser = argparse.ArgumentParser()
parser, args = parseArguments()
if args.dataset == "lggltl":
    src, tar = args.src_file_path, args.tar_file_path
else:
    src = args.src_dir_path


SEED = 0 #int(sys.argv[1])
MODE = -1 #2 #-1 #2 ##-1 for manual check## 2 for cross validation,  #9 for learning curve - fig 3
GLOVE = True
random.seed(SEED)
torch.manual_seed(SEED) if not use_cuda else torch.cuda.manual_seed(SEED)
print('Running with random seed {0}'.format(SEED))

is_lang2ltl = args.dataset == "lang2ltl"
print(is_lang2ltl)
if is_lang2ltl:
    pairs = {
        "train": [],
        "valid": []
    }
    input_lang = Lang(src)
    output_lang = Lang(src)
    MAX_LENGTH, MAX_TAR_LENGTH = -math.inf, -math.inf
    index = 0
    for path in os.listdir(src):
        index+=1
    # check if current path is a file
        filename = os.path.join(src, path)
        if "utt" in filename and os.path.isfile(filename):
            print("filename", filename)
            train_iter, valid_iter, curr_max_src_len, curr_max_tar_len  = readPkl(filename)
            pairs["train"].append(train_iter)
            pairs["valid"].append(valid_iter)
            print(curr_max_src_len, curr_max_tar_len)
            if MAX_LENGTH < curr_max_src_len:
                MAX_LENGTH = curr_max_tar_len

            if MAX_TAR_LENGTH < curr_max_tar_len:
                MAX_TAR_LENGTH = curr_max_tar_len

            input_lang, output_lang = prepareDataPkl(input_lang, output_lang, train_iter, valid_iter,index)
    MAX_LENGTH = int(MAX_LENGTH)
    MAX_TAR_LENGTH = int(MAX_TAR_LENGTH)
            # res.append(path)
    # readPkl(lang1, pairs)
    # exit()
else:
    input_lang, output_lang, pairs, MAX_LENGTH, MAX_TAR_LENGTH = prepareData(src, tar, False)
if not is_lang2ltl:
    random.shuffle(pairs)


# adding this code so the list order is the same for all tests irrespective of further calls to random
# in setting up the networks.
number_of_tests = 10


print('Maximum source sentence length: {0}'.format(MAX_LENGTH))
embed_size = 200 #50
hidden_size = 256
if GLOVE:
    glove_map = vectors_for_input_language(input_lang)
    glove_encoder = TestEncoderRNN(input_lang.n_words, embed_size, hidden_size, glove_map)
encoder1 = EncoderRNN(input_lang.n_words, embed_size, hidden_size)
attn_decoder1 = AttnDecoderRNN(embed_size, hidden_size, output_lang.n_words)
new_attn_decoder1 = NewAttnDecoderRNN(embed_size, hidden_size, output_lang.n_words, MAX_LENGTH)
com_attn_decoder1 = CombinedAttnDecoderRNN(embed_size, hidden_size, output_lang.n_words, MAX_LENGTH)
decoder1 = DecoderRNN(embed_size, hidden_size, output_lang.n_words)

if use_cuda:
    encoder1 = encoder1.cuda()
    if GLOVE:
        glove_encoder = glove_encoder.to(device)
    attn_decoder1 = attn_decoder1.cuda()
    new_attn_decoder1 = new_attn_decoder1.cuda()
    decoder1 = decoder1.cuda()
    com_attn_decoder1 = com_attn_decoder1.cuda()

SAVE = False
CLI = False
SUBSET = False #True


def main():
    # print("dataset length", len(pairs))
    global encoder1
    global attn_decoder1
    # print("length of p", len(pairs["valid"]))
    # valid_iter = p["valid"][0]
    print("length of valid_iter", len(pairs["valid"][0]))
    # exit()
    if MODE == -1:
        if args.is_load == True:
            p = pathlib.Path("../checkpoints_june6_test")
            fn = "encoder.pt" # I don't know what is your fn
            filepath = p / fn
            encoder1 = torch.load(filepath)

            fn = "decoder.pt" # I don't know what is your fn
            filepath = p / fn
            attn_decoder1 = torch.load(filepath)

        # input_sentence = raw_input("Enter a command: ")
        # output_words, attentions = evaluate(input_lang, output_lang, encoder1, attn_decoder1, input_sentence,
                                            # MAX_LENGTH)
        # print('input =', input_sentence)
        # print('output =', ' '.join(output_words))
        # crossValidation(input_lang, output_lang, encoder1, attn_decoder1, pairs, MAX_LENGTH, lang2ltl=is_lang2ltl, is_load = False) #False)
        if SUBSET:
            # train_samples = []
            n_folds = 5
            f = 0
            subset_samples = random.choices(pairs["train"][0], k=857)
            # print(len(subset_samples))
            fold_range = list(range(0, len(subset_samples), int(len(subset_samples) / n_folds)))
            fold_range.append(len(subset_samples))
            # print(type(samples))
            # print("f is:", f)
            # print("fold is:", fold_range)
            train_samples = subset_samples[:fold_range[f]] + subset_samples[fold_range[f + 1]:] ## train on 4 
            val_samples = subset_samples[fold_range[f]:fold_range[f + 1]]

            subset_pairs = {"train": [train_samples],
                            "valid": [val_samples]}
            evaluateRandomly(input_lang, output_lang, encoder1, attn_decoder1, subset_pairs, MAX_LENGTH)
        else:
            evaluateRandomly(input_lang, output_lang, encoder1, attn_decoder1, pairs, MAX_LENGTH)

    elif MODE == 0:
        trainIters(input_lang, output_lang, encoder1, attn_decoder1, pairs, 10000, MAX_LENGTH, print_every=500)
        encoder1.eval()
        attn_decoder1.eval()
        evaluateRandomly(input_lang, output_lang, encoder1, attn_decoder1, pairs, MAX_LENGTH)
    elif MODE == 1:
        trainIters(input_lang, output_lang, glove_encoder, attn_decoder1, pairs, 10000, MAX_LENGTH, print_every=500)
        encoder1.eval()
        attn_decoder1.eval()
        evaluateTraining(input_lang, output_lang, glove_encoder, attn_decoder1, pairs, MAX_LENGTH)
    elif MODE == 2:
        print("is load ", args.is_load,)
        
        if args.is_load == True:
            p = pathlib.Path("../checkpoints")
            fn = "encoder.pt" # I don't know what is your fn
            filepath = p / fn
            encoder1 = torch.load(filepath)

            fn = "decoder.pt" # I don't know what is your fn
            filepath = p / fn
            attn_decoder1 = torch.load(filepath)


        print('Running cross validation on encoder and BA decoder...')
        # crossValidation(input_lang, output_lang, encoder1, attn_decoder1, pairs, MAX_LENGTH, lang2ltl=is_lang2ltl, is_load = args.is_load) #False)
        crossValidation(input_lang, output_lang, glove_encoder, attn_decoder1, pairs, MAX_LENGTH, lang2ltl=is_lang2ltl, is_load = args.is_load, subset=SUBSET) #False)
    elif MODE == 3:
        print('Running cross validation on encoder and vanilla decoder...')
        crossValidation(input_lang, output_lang, encoder1, decoder1, pairs, MAX_LENGTH)
    elif MODE == 4:
        print('Running cross validation on encoder and EAA decoder...')
        crossValidation(input_lang, output_lang, encoder1, new_attn_decoder1, pairs, MAX_LENGTH)
    elif MODE == 5:
        print('Running generalization experiment with encoder and BA decoder...')
        results = []
        for i in reversed(range(1, 10)):
            # randomized_pairs = [pairs[o] for o in list_of_orders[i]]
            acc = evalGeneralization(input_lang, output_lang, encoder1, attn_decoder1, pairs, 0.1 * i, MAX_LENGTH)
            results.append(acc)
            encoder1.apply(resetWeights)
            attn_decoder1.apply(resetWeights)
        print(', '.join(map(str, reversed(results))))
    elif MODE == 6:
        print('Running generalization experiment with encoder and EAA decoder...')
        results = []
        for i in reversed(range(1, 10)):
            # randomized_pairs = [pairs[o] for o in list_of_orders[i]]
            acc = evalGeneralization(input_lang, output_lang, encoder1, new_attn_decoder1, pairs, 0.1 * i, MAX_LENGTH)
            results.append(acc)
            encoder1.apply(resetWeights)
            new_attn_decoder1.apply(resetWeights)
        print(', '.join(map(str, reversed(results))))
    elif MODE == 7:
        print('Running generalization experiment with encoder and vanilla decoder...')
        results = []
        for i in reversed(range(1, 10)):
            # randomized_pairs = [pairs[o] for o in list_of_orders[i]]
            acc = evalGeneralization(input_lang, output_lang, encoder1, decoder1, pairs, 0.1 * i,
                                     MAX_LENGTH)
            results.append(acc)
            encoder1.apply(resetWeights)
            decoder1.apply(resetWeights)
        print(', '.join(map(str, reversed(results))))
    elif MODE == 8:
        print('Running generalization experiment with encoder and CA decoder...')
        results = []
        for i in reversed(range(1, 10)):
            # randomized_pairs = [pairs[o] for o in list_of_orders[i]]
            acc = evalGeneralization(input_lang, output_lang, encoder1, com_attn_decoder1, pairs, 0.1 * i,
                                     MAX_LENGTH)
            results.append(acc)
            encoder1.apply(resetWeights)
            com_attn_decoder1.apply(resetWeights)
        print(', '.join(map(str, reversed(results))))

    elif MODE == 9:
        print('Running generalization experiment with glove encoder and EAA decoder...')
        results = []
        for i in reversed(range(1, 10)): # 1, 10
            # randomized_pairs = [pairs[o] for o in list_of_orders[i]]
            acc = evalGeneralization(input_lang, output_lang, glove_encoder, new_attn_decoder1, pairs, 0.1 * i,
                                     MAX_LENGTH)
            results.append(acc)
            glove_encoder.resetWeights()
            new_attn_decoder1.apply(resetWeights)
        print(', '.join(map(str, reversed(results))))

    # elif MODE == 10:
    #     print('Running classification test')
    #     acc = testClassifier(input_lang, output_lang, encoder1, new_attn_decoder1, randomized_pairs, 0.1 * i,
    #                              MAX_LENGTH)
    elif MODE == 200:
        langmod_path = './langmod_pre_train.pt'
        data = '../../data/gltl_langmod2.txt'
        corpus = Corpus(data)
        batch_size = 64
        train_data = batchify(corpus.train, batch_size)
        bptt = 1
        num_epochs = 3
        langmod = Langmod(50, 256, output_lang.n_words)

        if use_cuda:
            langmod = langmod.cuda()

        if not os.path.exists(langmod_path):
            print('Pre-training RNN language model...')

            for epoch in range(num_epochs):
                langmod_train(data, langmod, batch_size, bptt, epoch, log_interval=200, lr=1.0)

            torch.save(langmod.state_dict(), langmod_path)
        else:
            langmod.load_state_dict(torch.load(langmod_path))
        orig_e = copy.deepcopy(langmod.embed.weight)
        attn_decoder1.inherit(langmod)

        print('Running generalization + pre-training experiment with encoder and BA decoder...')
        results = []
        for i in reversed(range(1, 10)):
            # acc = evalGeneralization(input_lang, output_lang, encoder1, attn_decoder1, pairs, 0.1 * i, MAX_LENGTH)
            acc = evalGeneralizationPT(input_lang, output_lang, encoder1, attn_decoder1, langmod, pairs, 0.1 * i,
                                       MAX_LENGTH,
                                       train_data, batch_size, bptt)
            results.append(acc)
            print(results)
            encoder1.apply(resetWeights)
            attn_decoder1.apply(resetWeights)
            langmod = Langmod(50, 256, output_lang.n_words)
            if use_cuda:
                langmod = langmod.cuda()

            langmod.load_state_dict(torch.load(langmod_path))
            attn_decoder1.inherit(langmod)

            assert torch.sum(attn_decoder1.embedding.weight - orig_e).data[0] == 0.0
        print(', '.join(map(str, reversed(results))))



    # elif MODE == 7:
    #     results = []
    #     for i in range(1, 10):
    #         acc = evalSampleEff(input_lang, output_lang, encoder1, attn_decoder1, pairs, 0.1 * i, MAX_LENGTH)
    #         results.append(acc)
    #         encoder1.apply(resetWeights)
    #         attn_decoder1.apply(resetWeights)
    #     print(', '.join(map(str, results)))
    elif MODE == 100:
        from flask import Flask, request, jsonify

        app = Flask(__name__)

        encoder1.load_state_dict(torch.load('./pytorch_encoder'))
        attn_decoder1.load_state_dict(torch.load('./pytorch_decoder'))

        @app.route('/model')
        def model():
            nl_command = request.args.get('command')
            output_words, _ = evaluate(input_lang, output_lang, encoder1, attn_decoder1, nl_command, MAX_LENGTH)
            return ' '.join(output_words[:-1])

        app.run()
    else:
        print('Unknown MODE specified...exiting...')
        sys.exit(0)

    if SAVE:
        print('Serializing trained model...')
        torch.save(encoder1.state_dict(), './pytorch_encoder')
        torch.save(attn_decoder1.state_dict(), './pytorch_decoder')
        print('Serialized trained model to disk...')

    if CLI:
        while True:
            try:
                input_sentence = raw_input("Enter a command: ")
                output_words, attentions = evaluate(input_lang, output_lang, encoder1, attn_decoder1, input_sentence,
                                                    MAX_LENGTH)
                print('input =', input_sentence)
                print('output =', ' '.join(output_words))
            except EOFError:
                break


main()
