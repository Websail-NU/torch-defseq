#!/usr/bin/fish

vf activate tf
set temperatures ".5" ".25" ".1" ".05"
for t in $temperatures
	echo $argv[1]"/valid_samples"$t"_rank.txt.top"
	ipython bleu.py ../data/commondefs/valid.txt  $argv[1]"/valid_samples"$t"_rank.txt.top"
	echo $argv[1]"/valid_samples"$t"_rank.txt"
        ipython bleu.py ../data/commondefs/valid.txt  $argv[1]"/valid_samples"$t"_rank.txt"
end
vf deactivate


