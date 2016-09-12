#!/usr/bin/fish

set models "m1" "m2" "m3" "m4" "m5" "m6"
for m in $models
  ./bleu_valid.fish "../data/commondefs/models/"$m > "../data/commondefs/models/"$m"/bleu_valid.txt" &
end
