# -*- coding: utf-8 -*-
"""some functions for help."""

from itertools import groupby

import numpy as np
import scipy.sparse as sp
import pandas as pd

def read_txt(path):
    """read text file from path."""
    with open(path, "r") as f:
        return f.read().splitlines()

def load_data(path_dataset):
    """Load data in text format, one rating per line, as in the kaggle competition."""
    data = read_txt(path_dataset)[1:]
    return preprocess_data(data)

def preprocess_data(data):
    """preprocessing the text data, conversion to numerical array format."""
    def deal_line(line):
        pos, rating = line.split(',')
        row, col = pos.split("_")
        row = row.replace("r", "")
        col = col.replace("c", "")
        return int(row), int(col), float(rating)

    def statistics(data):
        row = set([line[0] for line in data])
        col = set([line[1] for line in data])
        return min(row), max(row), min(col), max(col)

    # parse each line
    data = [deal_line(line) for line in data]

    # do statistics on the dataset.
    min_row, max_row, min_col, max_col = statistics(data)
    print("number of items: {}, number of users: {}".format(max_row, max_col))

    # build rating matrix.
    ratings = sp.lil_matrix((max_row, max_col))
    for row, col, rating in data:
        ratings[row - 1, col - 1] = rating
    return ratings


def group_by(data, index):
    """group list of list by a specific index."""
    sorted_data = sorted(data, key=lambda x: x[index])
    groupby_data = groupby(sorted_data, lambda x: x[index])
    return groupby_data


def build_index_groups(train):
    """build groups for nnz rows and cols."""
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))

    grouped_nz_train_byrow = group_by(nz_train, index=0)
    nz_row_colindices = [(g, np.array([v[1] for v in value]))
                         for g, value in grouped_nz_train_byrow]

    grouped_nz_train_bycol = group_by(nz_train, index=1)
    nz_col_rowindices = [(g, np.array([v[0] for v in value]))
                         for g, value in grouped_nz_train_bycol]
    return nz_train, nz_row_colindices, nz_col_rowindices


def calculate_mse(real_label, prediction):
    """calculate MSE."""
    t = real_label - prediction
    return 1.0 * t.dot(t.T)


def predict(item_features, user_features, test_set):
    """ Apply MF model. Multiply matrices W and Z and select the wished predictions according to test set
    """
    # copy test set
    filled_test = sp.lil_matrix.copy(test_set)

    # compute prediction
    pred_matrix = np.dot(item_features.T, user_features)

    # fill test set with predicted label
    users, items, ratings = sp.find(filled_test)
    for row, col in zip(users, items):
        filled_test[row, col] = pred_matrix[row, col]

    return filled_test


def sp_to_df(sparse):
    """ Convert scipy.sparse matrix to pandas.DataFrame """

    row, col, rating = sp.find(sparse)
    row += 1
    col += 1

    df = pd.DataFrame({'User': row, 'Movie': col, 'Rating': rating})
    df = df[['User', 'Movie', 'Rating']].sort_values(['Movie', 'User'])
    return df

def submission_table(original_df, col_userID, col_movie, col_rate):
    """ return table according with Kaggle convention """

    def id(row):
        return 'r' + str(int(row[col_userID])) + '_c' + str(int(row[col_movie]))

    def pred(row):
        return row[col_rate]

    df = pd.DataFrame.copy(original_df)
    df['Id'] = df.apply(id, axis=1)
    df['Prediction'] = df.apply(pred, axis=1)

    return df[['Id', 'Prediction']]

#######################################################################################