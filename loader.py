# Load the Pima Indians diabetes dataset from CSV URL
import numpy as np
from operator import itemgetter
import pandas
import operator
import preprocessing as prep
import copy
import os

from collections import Counter
#from IPython.display import IFrame


class DataSet:
    data_set_name = ""
    rare_value_threshold = 0  # basic threshold 5%
    is_rare_values_applied = False # by default when we create a DS it is not yet transformed
    is_copy_verbose = False # whether we print out the state of copying

    @staticmethod
    def load_data_pandas(filename):
        file = open(filename, 'r')
        dataset = pandas.read_csv(file, dtype=None, delimiter=",")

        return dataset

    def __init__(self, filename):
        self.df = self.load_data_pandas(filename)
        self.data_set_name = os.path.basename(filename)

    def copyWithRareValues(self, threshold = 2, verbose = False):
        toreturn = copy.deepcopy(self)

        toreturn.rare_value_threshold = threshold
        toreturn.is_copy_verbose = verbose

        toreturn.insertRareValues()

        return toreturn


    def y(self):
        return self.df["label"]

    def X(self):
        return self.df.drop('label', axis=1, inplace=False)

    # finds the values of the rare values in the column
    # we can later use these values to remove rows containing them
    def rareValuesFromColumn(self, column):
        values = []
        c = Counter(column)
        if self.is_copy_verbose:
            print (c)
        occurences = [(i, c[i] / float(len(column)) * 100.0) for i in c]

        for i in occurences:
            if i[1] < self.rare_value_threshold:
                values.append(i[0])

        if self.is_copy_verbose:
            print (column.name, " has rare values: ", values)

        return values

    def insertRareValuesIntoColumn(self, columnName):
        col = self.df[columnName]
        rareValues = self.rareValuesFromColumn(self.df[columnName])

        for rv in rareValues:
            self.df[columnName].replace(rv, 'RV', inplace=True)

        if self.is_copy_verbose:
            print("In column ", columnName, " replaced ", (self.df[columnName] == 'RV').sum(), " values")

    def insertRareValues(self):
        df = self.df
        for col in df:
            self.insertRareValuesIntoColumn(col)

        self.is_rare_values_applied = True

    def basicDescription(self):
        print("Description of dataset:")
        print("Rare value transformation applied: ", self.is_rare_values_applied)
        if self.is_rare_values_applied:
            print("  - the threshold percentage was ", self.rare_value_threshold)

        for colName in self.df:
            counter = Counter(self.df[colName])
            print("Col: ", colName, " - ", counter)
