from subprocess import Popen, PIPE
import os
import sys
import random

tmp_dir = "/tmp"
bleu_path = "./sentence-bleu"
mode = 'average'
suffix = str(random.random())
ref_file = sys.argv[1]
hyp_file = sys.argv[2]

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

def get_bleu_score(bleu_path, all_ref_paths, d, hyp_path):
    with open(hyp_path, 'w') as ofp:
        ofp.write(d)
    read_cmd = ['cat', hyp_path]
    bleu_cmd = [bleu_path] + all_ref_paths
    rp = Popen(read_cmd, stdout=PIPE)
    bp = Popen(bleu_cmd, stdin=rp.stdout, stdout=PIPE, stderr=devnull)
    out, err = bp.communicate()
    if err is None:
        return float(out.strip())
    else:
        return None

# Read data
refs, hyps = None, None
with open(ref_file) as ifp:
    refs = read_definition_file(ifp)
with open(hyp_file) as ifp:
    hyps = read_definition_file(ifp)

# Check words
if len(refs) != len(hyps):
    print("Number of words being defined mismatched!")
words = refs.keys()

hyp_path = os.path.join(tmp_dir, 'hyp' + suffix)
to_be_deleted = set()
to_be_deleted.add(hyp_path)

# Computing BLEU
devnull = open(os.devnull, 'w')
score = 0
count = 0
total_refs = 0
total_hyps = 0
for word in words:
    if word not in refs or word not in hyps:
        continue
    wrefs = refs[word]
    whyps = hyps[word]
    # write out references
    all_ref_paths = []
    for i, d in enumerate(wrefs):
        ref_path = os.path.join(tmp_dir, 'ref' + suffix + str(i))
        with open(ref_path, 'w') as ofp:
            ofp.write(d)
            all_ref_paths.append(ref_path)
            to_be_deleted.add(ref_path)
    total_refs += len(all_ref_paths)
    # score for each output
    micro_score = 0
    micro_count = 0
    if mode == "average":
        for d in whyps:
            rhscore = get_bleu_score(bleu_path, all_ref_paths, d, hyp_path)
            if rhscore is not None:
                micro_score += rhscore
                micro_count += 1
    elif mode == "random":
        d = random.choice(whyps)
        rhscore = get_bleu_score(bleu_path, all_ref_paths, d, hyp_path)
        if rhscore is not None:
            micro_score += rhscore
            micro_count += 1
    total_hyps += micro_count
    score += micro_score / micro_count
    count += 1
devnull.close()

# delete tmp files
for f in to_be_deleted:
    os.remove(f)
print(score/count)
print(total_hyps)
print(total_refs)
