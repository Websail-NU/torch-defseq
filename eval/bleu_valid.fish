#!/usr/bin/fish

vf activate tf
set temperatures ".5" ".25" ".1" ".05" "1"
for t in $temperatures
  echo $argv[1]"/valid_samples"$t"_gen.txt"
  python bleu.py ../data/commondefs/valid.txt  $argv[1]"/valid_samples"$t"_gen.txt"
  echo $argv[1]"/valid_samples"$t"_rank.txt"
  python bleu.py ../data/commondefs/valid.txt  $argv[1]"/valid_samples"$t"_rank.txt"
  echo $argv[1]"/valid_samples"$t"_rank.txt.top"
  python bleu.py ../data/commondefs/valid.txt  $argv[1]"/valid_samples"$t"_rank.txt.top"
  echo $argv[1]"/valid_samples"$t"_rank2.txt.top"
  python bleu.py ../data/commondefs/valid.txt  $argv[1]"/valid_samples"$t"_rank2.txt.top"
end
vf deactivate
