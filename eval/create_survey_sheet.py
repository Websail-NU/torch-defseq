import os
import sys
import random

data_dir = sys.argv[1]
ref_file = "test.txt"
ne_file = "test_nearest.txt.all"
sys_a_file = "gen_output_a.txt"
sys_b_file = "gen_output_b.txt"
out_file_path =sys.argv[2]
target_site = sys.argv[3]

page_header = "Please rank the definitions following word"
new_page = "::NewPage:: " + page_header
n_pages = 10
words_per_page = 5

ref_defs, ne_defs, a_defs, b_defs = None, None, None, None
dictionaries = set()

def read_definition_file(ifp):
    defs = {}
    for line in ifp:
        if "<unk>" in line: continue
        parts = line.strip().split('\t')
        word = parts[0]
        dictionary = 'any'
        if len(parts) > 3:
            dictionary = parts[2]
            dictionaries.add(dictionary)
        definition = parts[-1]
        if word not in defs:
            defs[word] = {}
        if dictionary not in defs[word]:
            defs[word][dictionary] = []
        defs[word][dictionary].append(definition)
    return defs

print('Reading definitions...')
with open(os.path.join(data_dir, ref_file)) as ifp:
    ref_defs = read_definition_file(ifp)
with open(os.path.join(data_dir, ne_file)) as ifp:
    ne_defs = read_definition_file(ifp)
with open(os.path.join(data_dir, sys_a_file)) as ifp:
    a_defs = read_definition_file(ifp)
with open(os.path.join(data_dir, sys_b_file)) as ifp:
    b_defs = read_definition_file(ifp)

okay_ref_words = set()
okay_ne_words = set()
okay_a_words = set(a_defs.keys())
okay_b_words = set(b_defs.keys())
for word in ref_defs.keys():
    if len(ref_defs[word]) == len(dictionaries):
        okay_ref_words.add(word)
for word in ne_defs.keys():
    if len(ne_defs[word]) == len(dictionaries):
        okay_ne_words.add(word)
okay_words = okay_ref_words.intersection(okay_ne_words)
okay_words = okay_words.intersection(okay_a_words).intersection(okay_b_words)
okay_words = [x for x in okay_words]
dictionaries = [x for x in dictionaries]
used_words = set()

print('Generating survey data...')
ofp = open(os.path.join(data_dir, out_file_path), 'w')
if target_site == "google_sheet":
    for ipage in range(n_pages):
        ofp.write('PAGE_BREAK\t')
        ofp.write(page_header)
        ofp.write(' (' + str(ipage+1) + '/' + str(n_pages) + ')\n')
        for iw in range(words_per_page):
            word = random.choice(okay_words)
            while word in used_words:
                word = random.choice(okay_words)
            used_words.add(word)
            dictionary = random.choice(dictionaries)
            # dictionary = 'wordnet'
            ofp.write('MULTIPLE_CHOICE\t' + word + '\t\t\tYES\t')
            ofp.write(ref_defs[word][dictionary][0] + "\t")
            ofp.write(ne_defs[word][dictionary][0] + "\t")
            ofp.write(a_defs[word]['any'][0] + "\t")
            ofp.write(b_defs[word]['any'][0] + "\n")
elif target_site == 'survey_gizmo':
    for ipage in range(n_pages):
        ofp.write(new_page)
        ofp.write(' (' + str(ipage+1) + '/' + str(n_pages) + ')\n\n')
        for iw in range(words_per_page):
            word = random.choice(okay_words)
            while word in used_words:
                word = random.choice(okay_words)
            used_words.add(word)
            dictionary = random.choice(dictionaries)
            #dictionary = 'wordnet'
            ofp.write(word + '\n()')
            choices = [
                random.choice(ref_defs[word][dictionary]),
                random.choice(ne_defs[word][dictionary]),
                a_defs[word]['any'][0],
                b_defs[word]['any'][0]
            ]
            random.shuffle(choices)
            for c in choices:
                ofp.write(c + "\n")
            ofp.write("\n")
ofp.close()
print('Done')
if target_site=="google_sheet":
    print('Do not forget to enable randomized order option in the survey.')
