#!/usr/bin/fish

set models "m1" "m2" "m3" "m4" "m5" "m6"

for m in $models
  python format_bleu_valid.py "../data/commondefs/models/"$m"/bleu_valid.txt"
end
