#!/usr/bin/fish

vf activate tf
set temperatures "1" ".5" ".25" ".1" ".05"
for t in $temperatures
	echo $argv[1]"/valid_samples"$t"_gen.txt"
	ipython rerank.py $argv[1]"/valid_samples"$t"_gen.txt" data/commondefs/models/ngram/lm.arpa data/function_words.txt $argv[1]"/valid_samples"$t"_rank.txt"
	echo $argv[1]"/test_samples"$t"_gen.txt"
	ipython rerank.py $argv[1]"/test_samples"$t"_gen.txt" data/commondefs/models/ngram/lm.arpa data/function_words.txt $argv[1]"/test_samples"$t"_rank.txt"
end
vf deactivate


