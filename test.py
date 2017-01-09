import sys
import evaluating as eva
import loader
import os
import preprocessing as prep

from sklearn.neighbors import KNeighborsClassifier

if len(sys.argv) < 2:
    print('Invalid number of arguments, please pass filename to the program')
    sys.exit()


dataset = loader.DataSet('./datasets/hepatitis.data')
prep.remove_useless_columns(dataset, ['18','15','17','16','14'])
prep.apply_binning_on_continuous_values(dataset, ['1','2'])
# prep.replaceStringsWithNumbers(dataset)
# dataset.basicDescription()


# sys.exit()

#
# #redirect print into output file
# filename = str(sys.argv[1])
#
# orig_stdout = sys.stdout
# outputFileName = './results/' + os.path.basename(filename) + '.results.txt'
# f = open(outputFileName, 'w')
# sys.stdout = f
#
# #load dataset and replace strings
# dataset = loader.DataSet(filename)
# ds = loader.DataSet(filename)

ds = loader.DataSet('./datasets/hepatitis.data')
prep.remove_useless_columns(ds, ['18','15','17','16','14'])
prep.apply_binning_on_continuous_values(ds, ['1','2'])

prep.replaceStringsWithNumbers(ds)
eva.evaluate_all_metrics_on_all_models(ds)
i = 0.5
while i < 9:
    dataset2 = dataset.copyWithRareValues(i)
    prep.replaceStringsWithNumbers(dataset2)
    # dataset2.basicDescription()
    eva.evaluate_all_metrics_on_all_models(dataset2)
    i *= 2
