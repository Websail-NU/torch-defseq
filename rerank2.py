import os
import sys
import operator
from collections import Counter
from nltk.util import ngrams

def_file = sys.argv[1]
score_file = sys.argv[2]
function_word_file = sys.argv[3]
output_file = sys.argv[4]

function_words = set()
with open(function_word_file) as ifp:
    for line in ifp:
        function_words.add(line.strip())

def read_definition_file(ifp, sfp):
    ndefs = 0
    defs = {}
    for line, score in zip(ifp, sfp):
        ppl = float(score)
        parts = line.strip().split('\t')
        word = parts[0]
        definition = parts[-1]
        if word not in defs:
            defs[word] = []
        prev_def = None
        while prev_def != definition:
            prev_def = definition
        defs[word].append([definition, ppl])
        ndefs += 1
    return defs, ndefs

def score(word, definition, ppl, function_words=None):

    tokens = definition.split(' ')
    if function_words is not None:
        n_tokens = []
        for t in tokens:
            if t not in function_words:
                n_tokens.append(t)
    if len(n_tokens) > 0:
        tokens = n_tokens
    unigrams = Counter(ngrams(tokens, 1))
    unigram_penalty = sum(unigrams.values()) / float(len(unigrams.keys()))
    self_ref_penalty = 1
    if word in tokens:
        self_ref_penalty = 5
    # trigram_penalty = 1
    # bigram_penalty = 1
    # if len(tokens) > 2:
    #     bigram = Counter(ngrams(tokens, 2))
    #     bigram_penalty =  float(sum(bigram.values())) / len(bigram.keys())
    # if len(tokens) > 3:
    #     trigram = Counter(ngrams(tokens, 3))
    #     trigram_penalty =  float(sum(trigram.values())) / len(trigram.keys())
    return ppl * unigram_penalty * self_ref_penalty

print("Reading the definitions...")
with open(def_file) as ifp, open(score_file) as sfp:
    defs, ndefs = read_definition_file(ifp, sfp)
print(" - {} words being defined".format(len(defs)))
print(" - {} definitions".format(ndefs))

print("Reranking...")
# ofp_all = open(output_file, 'w')
ofp_top = open(output_file, 'w')
for w in defs:
    score_defs = []
    for d in defs[w]:
        d[1] = score(w, d[0], d[1], function_words)
        score_defs.append(d)
    score_defs.sort(key=lambda tup: tup[1])
    ofp_top.write(w + '\t' + score_defs[0][0] + '\n')
ofp_top.close()
