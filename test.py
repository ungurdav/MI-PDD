import sys
import evaluating as eva
import loader
import os
import preprocessing as prep

if len(sys.argv) < 2:
    print('Invalid number of arguments, please pass filename to the program')
    sys.exit()

#redirect print into output file
filename = str(sys.argv[1])

orig_stdout = sys.stdout
outputFileName = './results/' + os.path.basename(filename) + '.results.txt'
f = open(outputFileName, 'w')
sys.stdout = f

#load dataset and replace strings
dataset = loader.DataSet(filename)
ds = loader.DataSet(filename)

prep.replaceStringsWithNumbers(ds)
eva.evaluate_all_metrics_on_all_models(ds)
i = 0.5
while i < 9:
    dataset2 = dataset.copyWithRareValues(i)
    prep.replaceStringsWithNumbers(dataset2)
    eva.evaluate_all_metrics_on_all_models(dataset2)
    i *= 2
