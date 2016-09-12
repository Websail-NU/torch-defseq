import os
import sys
import kenlm
from collections import Counter
from nltk.util import ngrams
import operator
import re

pattern = re.compile(r'(?P<words>.+) (or|,|, or|of|and) (?P=words)\b')
def_file = sys.argv[1]
lm_file = sys.argv[2]
function_word_file = sys.argv[3]
output_file = sys.argv[4]

lm = kenlm.Model(lm_file)
function_words = set()
with open(function_word_file) as ifp:
    for line in ifp:
        function_words.add(line.strip())

def clean_repeated(text, max_len=6, min_len=1):
    global pattern
    s = pattern.search(text)
    new_text = text
    if s is not None:
        new_text = text[0:s.start()] + s.groups()[0]
        if s.end() < len(text):
            new_text = new_text + text[s.end():]
    text = new_text
    tokens = text.split()
    for n in range(max_len, min_len - 1, -1):
        if n > len(tokens):
            continue
        done = False
        while not done:
            ngram = Counter(ngrams(tokens, n))
            p = sorted(ngram.items(), key=operator.itemgetter(1), reverse=True)
            for k in p:
                if k[1] == 1:
                    done = True
                    break
                r = list(k[0])
                pos = [(i, i+len(r)) for i in range(len(tokens)) if tokens[i:i+len(r)] == r]
                prev_end = -1
                r_start = -1
                r_end = -1
                for start, end in pos:
                    if start <= prev_end:
                        if r_start == -1:
                            r_start = prev_end
                        r_end = end
                    prev_end = end
                if r_end != -1:
                    tokens = tokens[:r_start] + tokens[r_end:]
                    done = False
                    break
                else:
                    done = True
    return ' '.join(tokens)

def read_definition_file(ifp):
    ndefs = 0
    defs = {}
    for line in ifp:
        parts = line.strip().split('\t')
        word = parts[0]
        definition = parts[-1]
        if word not in defs:
            defs[word] = []
        prev_def = None
        while prev_def != definition:
            prev_def = definition
            definition = clean_repeated(clean_repeated(definition))
        defs[word].append(definition)
        ndefs += 1
    return defs, ndefs

def score(definition, function_words=None):
    if function_words is None:
        function_words = set()
    definition = definition.replace('<unk>', 'UNK')
    tokens = definition.split(' ')
    trigram_penalty = 1
    bigram_penalty = 1
    # if len(tokens) > 2:
    #     bigram = Counter(ngrams(tokens, 2))
    #     bigram_penalty =  float(sum(bigram.values())) / len(bigram.keys())
    # if len(tokens) > 3:
    #     trigram = Counter(ngrams(tokens, 3))
    #     trigram_penalty =  float(sum(trigram.values())) / len(trigram.keys())
    return -1 * lm.score(definition) / (len(tokens) + 1) * trigram_penalty * bigram_penalty

print("Reading the definitions...")
with open(def_file) as ifp:
    defs, ndefs = read_definition_file(ifp)
print(" - {} words being defined".format(len(defs)))
print(" - {} definitions".format(ndefs))

print("Reranking...")
ofp_all = open(output_file, 'w')
ofp_top = open(output_file + '.top', 'w')
for w in defs:
    score_defs = []
    for d in defs[w]:
        score_defs.append((d, score(d)))
    score_defs.sort(key=lambda tup: tup[1])
    ofp_top.write(w + '\t' + score_defs[0][0] + '\n')
    for d in score_defs:
        ofp_all.write(w + '\t' + d[0] + '\n')
ofp_all.close()
ofp_top.close()
