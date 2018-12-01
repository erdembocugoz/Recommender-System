# -*- coding: utf-8 -*-
"""some functions for plots."""

import numpy as np
import matplotlib.pyplot as plt


def plot_raw_data(ratings):
    """plot the statistics result on raw rating data."""
    # do statistics.
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    sorted_num_movies_per_user = np.sort(num_items_per_user)[::-1]
    sorted_num_users_per_movie = np.sort(num_users_per_item)[::-1]

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sorted_num_movies_per_user, color='blue')
    ax1.set_xlabel("users")
    ax1.set_ylabel("number of ratings (sorted)")
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(sorted_num_users_per_movie)
    ax2.set_xlabel("items")
    ax2.set_ylabel("number of ratings (sorted)")
    ax2.set_xticks(np.arange(0, 12000, 3000))
    ax2.grid()

    plt.tight_layout()
    plt.savefig("stat_ratings")
    plt.show()
    # plt.close()
    return num_items_per_user, num_users_per_item


def plot_train_test_data(train, test):
    """visualize the train and test data."""
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.spy(train, precision=0.01, markersize= 0.5)
    ax1.set_xlabel("Users")
    ax1.set_ylabel("Items")
    ax1.set_title("Training data")
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.spy(test, precision=0.01, markersize= 0.5)
    ax2.set_xlabel("Users")
    ax2.set_ylabel("Items")
    ax2.set_title("Test data")
    plt.tight_layout()
    plt.savefig("train_test")
    plt.show()
    
def plot_mean_and_std_per_user(user_means, user_stds):
    data = user_means

    bins = np.arange(0, 5, 0.05) # fixed bin size
    plt.xlim([min(data) -0.5, max(data) + 0.5])
    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('mean per user')
    plt.xlabel('mean')
    plt.ylabel('count')
    plt.show()

    data = user_stds
    
    bins = np.arange(0, 2, 0.03) # fixed bin size

    plt.xlim([min(data) - 0.3, max(data) + 0.3])

    plt.hist(data, bins=bins, alpha=0.5)
    plt.title('std per user')
    plt.xlabel('std')
    plt.ylabel('count')
    plt.show()







    
    
