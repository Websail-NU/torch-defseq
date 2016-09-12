#!/usr/bin/fish

vf activate tf
set temperatures "1" ".5" ".25" ".1" ".05"
for t in $temperatures
	echo $argv[1]"/valid_samples"$t"_rank.txt"
	python rerank2.py $argv[1]"/valid_samples"$t"_rank.txt" $argv[1]"/score_valid_samples"$t"_rank.txt"  data/function_words.txt $argv[1]"/valid_samples"$t"_rank2.txt.top"
	echo $argv[1]"/test_samples"$t"_rank.txt"
	python rerank2.py $argv[1]"/test_samples"$t"_rank.txt" $argv[1]"/score_test_samples"$t"_rank.txt"  data/function_words.txt $argv[1]"/test_samples"$t"_rank2.txt.top"
end
vf deactivate
