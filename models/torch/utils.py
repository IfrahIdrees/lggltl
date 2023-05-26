import math
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


#
# def trainLangmod(input_variable, target_variable, langmod, langmod_optimizer, criterion):
#     langmod_hidden = langmod.initHidden()
#     langmod_optimizer.zero_grad()
#
#     loss = 0.0
#
#     langmod_output, langmod_hidden = langmod(input_variable, langmod_hidden)
#     topv, topi = langmod_output.data.topk(1)
#     loss += criterion(langmod_output, target_variable[0])
#
#     loss.backward()
#
#     langmod_optimizer.step()
#
#     return loss.data[0]
#
#
# def trainLangmodIters(langmod, lang, samples, n_iters, print_every=100, learning_rate=0.001):
#     start = time.time()
#     print_loss_total = 0
#
#     langmod_optimizer = optim.Adam(langmod.parameters(), lr=learning_rate)
#     train_data = itertools.cycle(iter([[SOS_token] + indexesFromSentence(lang, s) + [EOS_token] for s in samples]))
#     criterion = nn.NLLLoss()
#     for i in range(1, n_iters + 1):
#         indexes = train_data.next()
#         for j in range(1, len(indexes)):
#             input_variable = Variable(torch.LongTensor([indexes[j - 1]]).view(-1, 1))
#             if use_cuda:
#                 input_variable = input_variable.cuda()
#             target_variable = Variable(torch.LongTensor([indexes[j]]).view(-1, 1))
#             if use_cuda:
#                 target_variable = target_variable.cuda()
#
#             loss = trainLangmod(input_variable, target_variable, langmod, langmod_optimizer, criterion)
#             print_loss_total += loss
#
#         if i % print_every == 0:
#             print_loss_avg = print_loss_total / print_every
#             print_loss_total = 0
#             print('%s (%d %d%%) %.4f' % (timeSince(start, i / n_iters), i, i / n_iters * 100, print_loss_avg))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points, filepath=None, is_multiple =False):
    plt.figure()
    fig, ax = plt.subplots()
    if is_multiple:
        plt.plot(points[0][0], points[0][1], label = "train loss")
        plt.plot(points[1][0], points[1][1], label = "valid loss")
    else:
        plt.plot(points[0], points[1], label = "train loss")
    ax.legend()
    plt.savefig(filepath)
    plt.close()
