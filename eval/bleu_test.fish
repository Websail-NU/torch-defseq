#!/usr/bin/fish

vf activate tf
set references "../data/commondefs/test.txt" "../data/commondefs/models/inter/test_gcide.txt" "../data/commondefs/models/inter/test_wordnet.txt"
for r in $references
	echo $r
	echo $argv[1]"/test_samples"$argv[2]"_rank.txt"$argv[3]
	ipython bleu.py $r  $argv[1]"/test_samples"$argv[2]"_rank.txt"$argv[3]
end
vf deactivate


