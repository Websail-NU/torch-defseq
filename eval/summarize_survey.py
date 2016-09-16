import os
import sys
import random
import csv
from numpy import corrcoef

data_dir = sys.argv[1]
survey_file = "20160913122653-SurveyExport.csv"
ref_file = "test.txt"
ne_file = "test_nearest.txt.all"
sys_a_file = "gen_output_a.txt"
sys_b_file = "gen_output_b.txt"

all_defs = {}

def read_definition_file(ifp):
    defs = {}
    for line in ifp:
        parts = line.strip().split('\t')
        word = parts[0]
        definition = parts[-1]
        if word not in defs:
            defs[word] = []
        defs[word].append(definition)
    return defs

print('Reading definitions...')
with open(os.path.join(data_dir, ref_file)) as ifp:
    all_defs['ref'] = read_definition_file(ifp)
with open(os.path.join(data_dir, ne_file)) as ifp:
    all_defs['ne'] = read_definition_file(ifp)
with open(os.path.join(data_dir, sys_a_file)) as ifp:
    all_defs['a'] = read_definition_file(ifp)
with open(os.path.join(data_dir, sys_b_file)) as ifp:
    all_defs['b'] = read_definition_file(ifp)

print('Reading survey data...')
header = True
column_map = {}
data = {}
num_annotators = 0
with open(os.path.join(data_dir, survey_file)) as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        for i, h in enumerate(row):
            if header:
                column_map[i] = h
                data[h] = []
            else:
                data[column_map[i]].append(h)
        header = False

def find_source(word, definition):
    m = set()
    for k in all_defs:
        if word not in all_defs[k]:
            continue
        defs = all_defs[k][word]
        for d in defs:
            if d == definition:
                m.add(k)
    return m

print('Processing data...')
ranking = {}
annotator_ranks = {}
for k in data:
    if ":" not in k:
        continue
    parts = k.split(':')
    word = parts[1]
    definition = parts[0]
    sources = find_source(word, definition)
    ranks = [int(x) for x in data[k]]
    num_annotators = len(ranks)
    if word not in ranking:
        ranking[word] = []
        annotator_ranks[word] = [[] for _ in ranks]
    for i, r in enumerate(ranks):
        annotator_ranks[word][i].append(r)
    for s in sources:
        ranking[word].append((definition, s, len(sources), ranks))

raw_data = data

print('Resolving conflict...')
for w in ranking:
    data = ranking[w]
    d_count = {}
    s_count = {}
    for p in data:
        if p[0] not in d_count:
            d_count[p[0]] = 0
        d_count[p[0]] += 1
        if p[1] not in s_count:
            s_count[p[1]] = 0
        s_count[p[1]] += 1
    new_data = []
    removed_data = []
    for p in data:
        if d_count[p[0]] > 1 and p[2] > 1 and s_count[p[1]] > 1:
            removed_data.append(p)
        else:
            new_data.append(p)
    for k in s_count:
        s_count[k] = 0
    for p in new_data:
        s_count[p[1]] = 1
    for p in removed_data:
        if s_count[p[1]] == 0:
            new_data.append(p)
    ranking[w] = new_data

print('Summarizing...')
sum_rank = {}
split_rank = {}
sum_each_rank = [0,0,0,0]
count_rank = {}
system_rank={}
for w in ranking:
    system_rank[w] = {'a':[], 'b':[], 'ref':[], 'ne':[]}
    for p in ranking[w]:
        if p[1] not in sum_rank:
            sum_rank[p[1]] = 0
            count_rank[p[1]] = 0
            split_rank[p[1]] = [0, 0, 0, 0]
        sum_rank[p[1]] += sum(p[3]) / float(len(p[3]))
        system_rank[w][p[1]] += p[3]
        for r in p[3]:
            split_rank[p[1]][r - 1] += 1
            sum_each_rank[r - 1] += 1
        count_rank[p[1]] += 1

for k in sum_rank:
    display = []
    print(k + ": ")
    print("- Average rank: {}".format(sum_rank[k] / count_rank[k]))
    for r in range(4):
        print("- {} ranked {}".format(
            split_rank[k][r] / float(sum_each_rank[r]), r + 1))
        display.append(100 * split_rank[k][r] / float(sum_each_rank[r]))
    display.append(sum_rank[k] / count_rank[k])
    display = ['{:.2f}'.format(x) for x in display]
    print(' & '.join(display) + ' \\\\')

print('Computing correlation...')

correlation = [[1 for _ in range(num_annotators)]
               for __ in range(num_annotators)]
for i in range(num_annotators):
    for j in range(num_annotators):
        cs = []
        for w in annotator_ranks:
            ranks = annotator_ranks[w]
            cs.append(corrcoef(ranks[i], ranks[j])[0, 1])
        correlation[i][j] = sum(cs) / len(cs)
for r in correlation:
    print('\t'.join([str(x) for x in r]))

print('Average system rank:')
systems = ['ref', 'ne', 'a', 'b']
for w in system_rank:
    data = [w]
    for s in systems:
        r = system_rank[w][s]
        data.append(str(sum(r) / float(len(r))))
    print('\t'.join(data))
