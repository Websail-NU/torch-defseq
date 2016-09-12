#!/usr/bin/fish

set temperatures "1" ".5" ".25" ".1" ".05"

for t in $temperatures
  th score.lua --mode sen --cuda --modelDir "data/commondefs/models/m1" --entryFile "data/commondefs/models/m1/test_samples"$t"_rank.txt"  --batchSize 40 --outputFile "data/commondefs/models/m1/score_test_samples"$t"_rank.txt"
  th score.lua --mode sen --cuda --modelDir "data/commondefs/models/m1" --entryFile "data/commondefs/models/m1/valid_samples"$t"_rank.txt"  --batchSize 40 --outputFile "data/commondefs/models/m1/score_valid_samples"$t"_rank.txt"
end

set models "m2" "m3" "m4"

for m in $models
  for t in $temperatures
    th score.lua --mode ri --cuda --modelDir "data/commondefs/models/"$m --entryFile "data/commondefs/models/"$m"/test_samples"$t"_rank.txt"  --batchSize 40 --outputFile "data/commondefs/models/"$m"/score_test_samples"$t"_rank.txt"
    th score.lua --mode ri --cuda --modelDir "data/commondefs/models/"$m --entryFile "data/commondefs/models/"$m"/valid_samples"$t"_rank.txt"  --batchSize 40 --outputFile "data/commondefs/models/"$m"/score_valid_samples"$t"_rank.txt"
  end
end


for t in $temperatures
  th score.lua --mode ri --RICharCNN --cudnnCNN --cuda --modelDir "data/commondefs/models/m5" --entryFile "data/commondefs/models/m5/test_samples"$t"_rank.txt"  --batchSize 40 --outputFile "data/commondefs/models/m5/score_test_samples"$t"_rank.txt"
  th score.lua --mode ri --RICharCNN --cudnnCNN --cuda --modelDir "data/commondefs/models/m5" --entryFile "data/commondefs/models/m5/valid_samples"$t"_rank.txt"  --batchSize 40 --outputFile "data/commondefs/models/m5/score_valid_samples"$t"_rank.txt"
end


for t in $temperatures
  th score.lua --mode ri --RICharCNN --RIHypernym --cudnnCNN --cuda --modelDir "data/commondefs/models/m6" --entryFile "data/commondefs/models/m6/test_samples"$t"_rank.txt"  --batchSize 40 --outputFile "data/commondefs/models/m6/score_test_samples"$t"_rank.txt"
  th score.lua --mode ri --RICharCNN --RIHypernym --cudnnCNN --cuda --modelDir "data/commondefs/models/m6" --entryFile "data/commondefs/models/m6/valid_samples"$t"_rank.txt"  --batchSize 40 --outputFile "data/commondefs/models/m6/score_valid_samples"$t"_rank.txt"
end
