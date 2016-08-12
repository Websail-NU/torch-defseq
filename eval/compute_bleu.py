import nltk
import sys

true_defs_file = sys.argv[1]
gen_defs_file = sys.argv[2]

true_defs = []
with open(true_defs_file, 'r') as f:
    for line in f:
        true_defs.append(line.strip().split('\t'))

gen_defs = []
with open(gen_defs_file, 'r') as f:
    for line in f:
        gen_defs.append(line.strip().split('\t'))

# create references dictionary
ref_dict = {}
hyps = []
refs_list = []
for e in true_defs:
    word = e[0]
    sent = e[-1].split(' ')
    if word in ref_dict:
        ref_dict[word].append(sent)
    else:
        ref_dict[word] = [sent]

for e in gen_defs:
    word = e[0]
    gen_sent = e[-1].split(' ')
    hyps.append(gen_sent)
    refs_list.append(ref_dict[word])

bleu_score = nltk.translate.bleu_score.corpus_bleu(
    refs_list, hyps, weights=(0.25, 0.25, 0.25, 0.25),
    smoothing_function=None)

bleu_score1 = nltk.translate.bleu_score.corpus_bleu(
    refs_list, hyps, weights=(1, 0, 0, 0),
    smoothing_function=None)

bleu_score2 = nltk.translate.bleu_score.corpus_bleu(
    refs_list, hyps, weights=(0, 1, 0, 0),
    smoothing_function=None)

bleu_score3 = nltk.translate.bleu_score.corpus_bleu(
    refs_list, hyps, weights=(0, 0, 1, 0),
    smoothing_function=None)

bleu_score4 = nltk.translate.bleu_score.corpus_bleu(
    refs_list, hyps, weights=(0, 0, 0, 1),
    smoothing_function=None)

print 'corpus bleu score is', bleu_score
print 'bleu1 score is', bleu_score1
print 'bleu2 score is', bleu_score2
print 'bleu3 score is', bleu_score3
print 'bleu4 score is', bleu_score4

print("{}\t{}\t{}\t{}\t{}".format(bleu_score, bleu_score1, bleu_score2, bleu_score3, bleu_score4))
