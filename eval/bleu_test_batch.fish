#!/usr/bin/fish
set m $argv[1]
set t1 $argv[2]
set t2 $argv[3]
set t3 $argv[4]
set t4 $argv[5]
#echo $m
#echo $t1
#echo $t2
#echo $t3

echo $m" gen"
./bleu_test.fish "../data/commondefs/models/"$m $t1 "gen" ""

echo $m" rank"
./bleu_test.fish "../data/commondefs/models/"$m $t2 "rank" ""

echo $m" top"
./bleu_test.fish "../data/commondefs/models/"$m $t3 "rank" ".top"

echo $m" top"
./bleu_test.fish "../data/commondefs/models/"$m $t4 "rank2" ".top"

#set models "m2" "m3" "m4" "m5" "m5_2"

#for m in $models
#  echo $m" gen"
#  ./bleu_test.fish "../data/commondefs/models/"$m ".05" "gen" ""
#  echo $m" rank"
#  ./bleu_test.fish "../data/commondefs/models/"$m ".05" "rank" ""
#  echo $m" top"
#  ./bleu_test.fish "../data/commondefs/models/"$m ".05" "rank" ".top"
#end
