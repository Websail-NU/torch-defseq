#!/usr/bin/fish

set temperatures ".05" ".1" ".25" ".5"
set models "m6"
set model_dir "data/commondefs/models/"
set vshortlist "shortlist/shortlist_valid.txt"
set tshortlist "shortlist/shortlist_test.txt"
set args "--dataType" "labeled" "--mode" "ri" "--batchSize" "50" "--gen" "--genMaxLen" "30" "--genMethod" "sampling" "--genSamples" "40" "--cuda" "--cudnnCNN" "--RICharCNN" "--RIHypernym"
for m in $models
	for t in $temperatures
		echo $m" "$t
		#th test.lua --help
		th test.lua  $args "--modelDir" $model_dir$m "--genWords" $vshortlist "--genOutFile" "valid_samples"$t"_gen.txt" "--temperature" $t
		th test.lua  $args "--modelDir" $model_dir$m "--genWords" $tshortlist "--genOutFile" "test_samples"$t"_gen.txt" "--temperature" $t
	end
end
# th test.lua --dataType labeled --mode ri --RICharCNN --batchSize 50 --modelDir data/commondefs/models/m5/  --gen --genWords shortlist/shortlist_test.txt --genOutFile test_samples.05_gen.txt --genMaxLen 30 --genMethod sampling --temperature 0.05 --genSamples 40 --cuda --cudnnCNN
# th test.lua --dataType labeled --mode ri --RICharCNN --batchSize 50 --modelDir data/commondefs/models/m5/  --gen --genWords shortlist/shortlist_valid.txt --genOutFile valid_samples.05_gen.txt --genMaxLen 30 --genMethod sampling --temperature 0.05 --genSamples 40 --cuda --cudnnCNN
#th test.lua --dataType labeled --mode ri --RICharCNN --batchSize 50 --modelDir data/commondefs/models/m5/  --gen --genWords shortlist/shortlist_test.txt --genOutFile test_samples.1_gen.txt --genMaxLen 30 --genMethod sampling --temperature 0.1 --genSamples 40 --cuda --cudnnCNN
#th test.lua --dataType labeled --mode ri --RICharCNN --batchSize 50 --modelDir data/commondefs/models/m5/  --gen --genWords shortlist/shortlist_valid.txt --genOutFile valid_samples.1_gen.txt --genMaxLen 30 --genMethod sampling --temperature 0.1 --genSamples 40 --cuda --cudnnCNN
#th test.lua --dataType labeled --mode ri --RICharCNN --batchSize 50 --modelDir data/commondefs/models/m5/  --gen --genWords shortlist/shortlist_test.txt --genOutFile test_samples.25_gen.txt --genMaxLen 30 --genMethod sampling --temperature 0.25 --genSamples 40 --cuda --cudnnCNN
#th test.lua --dataType labeled --mode ri --RICharCNN --batchSize 50 --modelDir data/commondefs/models/m5/  --gen --genWords shortlist/shortlist_valid.txt --genOutFile valid_samples.25_gen.txt --genMaxLen 30 --genMethod sampling --temperature 0.25 --genSamples 40 --cuda --cudnnCNN
#th test.lua --dataType labeled --mode ri --RICharCNN --batchSize 50 --modelDir data/commondefs/models/m5/  --gen --genWords shortlist/shortlist_test.txt --genOutFile test_samples.5_gen.txt --genMaxLen 30 --genMethod sampling --temperature 0.5 --genSamples 40 --cuda --cudnnCNN
#th test.lua --dataType labeled --mode ri --RICharCNN --batchSize 50 --modelDir data/commondefs/models/m5/  --gen --genWords shortlist/shortlist_valid.txt --genOutFile valid_samples.5_gen.txt --genMaxLen 30 --genMethod sampling --temperature 0.5 --genSamples 40 --cuda --cudnnCNN
#th test.lua --dataType labeled --mode ri --RICharCNN --batchSize 50 --modelDir data/commondefs/models/m5/  --gen --genWords shortlist/shortlist_test.txt --genOutFile test_samples1_gen.txt --genMaxLen 30 --genMethod sampling --temperature 1.0 --genSamples 40 --cuda --cudnnCNN
#th test.lua --dataType labeled --mode ri --RICharCNN --batchSize 50 --modelDir data/commondefs/models/m5/  --gen --genWords shortlist/shortlist_valid.txt --genOutFile valid_samples1_gen.txt --genMaxLen 30 --genMethod sampling --temperature 1.0 --genSamples 40 --cuda --cudnnCNN

