import math
import operator
import sys
import numpy as np

test_file = "data/commondefs/test.txt"
rvd_file = "data/commondefs/models/" + sys.argv[1] + "/rvd.out.tsv"
pos_redirect = {
    's':'adj',
    'a':'adj',
    'r':'misc',
    'n':'noun',
    'v':'verb',
    'v. t.':'verb',
    'v. i.':'verb',
    'n.':'noun',
    'a.':'adj',
    'adv.':'misc',
    'imp.':'misc',
    'interj.':'misc',
    'p. p.':'misc',
    'p. pr.':'misc',
    'n. pl.':'noun',
    'prop. n.':'noun',
    'n. sing.':'noun',
    'v. n.':'verb',
    'p. a.':'adj'}
pos_map = {}
ll_by_pos = {}
word_defs = {}
with open(test_file) as ifp:
    for line in ifp:
        parts = line.strip().split('\t')
        key = parts[0] + ' <def> ' + parts[3]
        pos = pos_redirect[parts[1]]
        pos_map[key] = pos
        if parts[0] not in word_defs:
            word_defs[parts[0]] = set()
        word_defs[parts[0]].add(parts[3])

with open(rvd_file) as ifp:
    for line in ifp:
        parts = line.strip().split('\t')
        key = parts[0] + ' <def> ' + parts[1]
        pos = pos_map[key]
        samples = len(parts[1].split(' ')) + 1
        # if samples < 3 or samples > 20: continue
        position = (int(parts[2]) - 1) * 2 + 3
        ll = -1 * math.log(float(parts[position + 1])) * samples
        if pos not in ll_by_pos:
            ll_by_pos[pos] = [0, 0]
        ll_by_pos[pos][0] = ll_by_pos[pos][0] + ll
        ll_by_pos[pos][1] = ll_by_pos[pos][1] + samples

print('Perplexity by POS:')
for key in ll_by_pos:
    e = ll_by_pos[key]
    ppl = math.exp(- e[0] / e[1])
    print('- {}: PPL = {}, Samples = {}'.format(key, ppl, e[1]))
print('\n')
pos_map = None
ll_by_pos = None

prob_by_word = {}
prob_by_def = {}
rvd_rank = []
with open(rvd_file) as ifp:
    for line in ifp:
        parts = line.strip().split('\t')
        d = parts[1]
        rvd_rank.append(int(parts[2]))
        samples = len(d.split(' ')) + 1
        for i in range(3, len(parts), 2):
            w = parts[i]
            ll = -1 * math.log(float(parts[i + 1])) * samples
            prob = math.exp(ll)
            if w not in prob_by_word:
                prob_by_word[w] = {}
            prob_by_word[w][d] = prob
            if d not in prob_by_def:
                prob_by_def[d] = 0
            prob_by_def[d]  = prob_by_def[d] + prob

top1 = 0
top10 = 0
top100 = 0
avg_rank = 0
count = 0
rprec = 0.0
all_rank = []
for w in prob_by_word:
    for d in prob_by_word[w]:
        prob_by_word[w][d] = prob_by_word[w][d] / prob_by_def[d]
    out = sorted(prob_by_word[w].items(), key=operator.itemgetter(1),
                 reverse=True)
    def_set = word_defs[w]
    prec = 0.0
    r = 1
    for o in out:
        if o[0] in def_set:
            prec += 1
        r += 1
        if r >= len(def_set):
            break
    prec = prec / len(def_set)
    rprec += prec
    for d in def_set:
        c = 1
        for o in out:
            if o[0] == d:
                if c <= 1: top1 += 1
                if c <= 10: top10 += 1
                if c <= 100: top100 += 1
                avg_rank += c
                all_rank.append(c)
                count +=1
                break
            if o[0] not in def_set:
                c += 1
print('Forward Dictionary:')
print('- Accuracy = {}'.format(top1 / float(count)))
print('- Top 10 = {}'.format(top10 / float(count)))
print('- Top 100 = {}'.format(top100 / float(count)))
print('- Average rank = {}'.format(avg_rank / float(count)))
print('- R-Precision = {}'.format(rprec / len(prob_by_word)))
ar = np.array(all_rank)
print('- STD = {}'.format(ar.std()))
rvd_ar = np.array(rvd_rank)
print('Reverse Dictionary:')
print('- STD = {}'.format(rvd_ar.std()))
