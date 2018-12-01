
# coding: utf-8

# In[80]:


from surprise import SVD
from surprise import *
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
import pandas as pd 
import numpy as np
import math


# In[2]:


data = Dataset.load_builtin('ml-100k')


# In[ ]:


algo = SVD()


# In[ ]:


cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


# In[ ]:


data


# In[ ]:


data.raw_ratings


# In[5]:


df = pd.read_csv("data_train.csv")


# In[8]:


df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
df['Rating'] = df['Prediction']
df = df.drop(['Id', 'Prediction'], axis=1)


# In[29]:


train_file = 'tmp_train.csv'
header = ['user_id','item_id','rating']
df.to_csv(train_file, index=False, header=False)
ratings_df = pd.read_csv('tmp_train.csv',
                          sep=',',
                         names=header,
                         
                         dtype={
                           'user_id': np.int32,
                           'item_id': np.int32,
                           'rating': np.float32,
                           'timestamp': np.int32,
                         })


# In[30]:





# In[10]:


reader = Reader(line_format='user item rating', sep=',')

# Train and test set for Surprise
# Load the data
data = Dataset.load_from_file(train_file, reader=reader)



# In[ ]:



# Algorithm
cross_validate(BaselineOnly(), data, verbose=True)


# In[12]:


param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3)

gs.fit(data)

# best RMSE score
print(gs.best_score['rmse'])

# combination of parameters that gave the best RMSE score
print(gs.best_params['rmse'])


# In[15]:


cross_validate(SVD(n_epochs=10,lr_all=0.002,reg_all=0.4), data, verbose=True,cv=3)


# In[17]:


cross_validate(BaselineOnly(), data, verbose=True,cv=3)


# In[63]:


from spotlight.cross_validation import user_based_train_test_split
from spotlight.datasets.synthetic import generate_sequential
from spotlight.evaluation import sequence_mrr_score
from spotlight.sequence.implicit import ImplicitSequenceModel


# In[64]:


dataset = generate_sequential(num_users=100,
                              num_items=1000,
                              num_interactions=10000,
                              concentration_parameter=0.01,
                              order=3)


# In[65]:


dataset


# In[105]:


import numpy as np

from spotlight.datasets.movielens import get_movielens_dataset
dataset = get_movielens_dataset(variant='100K')


# In[109]:


dataset.timestamps


# In[70]:


from spotlight.datasets import _transport
from spotlight.interactions import Interactions


# In[102]:


userid = df.values[:,0].astype(np.int32)
movieid = df.values[:,1].astype(np.int32)
rating = df.values[:,2].astype(np.int32)


# In[103]:


dataset = Interactions(userid,movieid,rating)


# In[104]:


train, test = user_based_train_test_split(dataset)

train = train.to_sequence()
test = test.to_sequence()

model = ImplicitSequenceModel(n_iter=3,
                              representation='cnn',
                              loss='bpr')
model.fit(train)


# ## Tensorflow example
# 

# In[6]:


df = pd.read_csv("data_train.csv")


# In[7]:


df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
df['Rating'] = df['Prediction']
df = df.drop(['Id', 'Prediction'], axis=1)


# In[8]:


train_file = 'tmp_train.csv'
header = ['user_id','item_id','rating']
df.to_csv(train_file, index=False, header=False)
ratings_df = pd.read_csv('tmp_train.csv',
                          sep=',',
                         names=header,
                         
                         dtype={
                           'user_id': np.int32,
                           'item_id': np.int32,
                           'rating': np.float32,
                           'timestamp': np.int32,
                         })


# In[9]:


ratings = ratings_df.as_matrix(['user_id', 'item_id', 'rating'])


# In[11]:


ratings[:,0] -= 1
ratings[:,1] -= 1


# #### The 1m and 20m MovieLens datasets skip some user and item IDs. This creates a problem: you have to map the set of unique user IDs to an index set equal to [0 ... num_users-1] and do the same for item IDs. The item mapping is accomplished using the following [numpy](http://www.numpy.org/) code. The code creates an array of size [0..max_item_id] to perform the mapping, so if the maximum item ID is very large, this method might use too much memory.

# In[12]:


np_items = ratings_df.item_id.as_matrix()
unique_items = np.unique(np_items)
n_items = unique_items.shape[0]
max_item = unique_items[-1]

# map unique items down to an array 0..n_items-1
z = np.zeros(max_item+1, dtype=int)
z[unique_items] = np.arange(n_items)
i_r = z[np_items]


# In[13]:


np_users = ratings_df.user_id.as_matrix()
unique_users = np.unique(np_users)
n_users = unique_users.shape[0]
max_user = unique_users[-1]

# map unique items down to an array 0..n_items-1
z = np.zeros(max_user+1, dtype=int)
z[unique_users] = np.arange(n_users)
i_r = z[np_users]


# #### The model code randomly selects a test set of ratings. By default, 10% of the ratings are chosen for the test set. These ratings are removed from the training set and will be used to evaluate the predictive accuracy of the user and item factors.

# In[14]:


test_set_size = int(len(ratings) / 10)
test_set_idx = np.random.choice(range(len(ratings)),
                                size=test_set_size, replace=False)
test_set_idx = sorted(test_set_idx)

ts_ratings = ratings[test_set_idx]
tr_ratings = np.delete(ratings, test_set_idx, axis=0)


# #### Finally, the code creates a scipy sparse matrix in coordinate form (coo_matrix) that includes the user and item indexes and ratings. The coo_matrix object acts as a wrapper for a sparse matrix. It also performs validation of the user and ratings indexes, checking for errors in preprocessing.

# In[85]:


from scipy.sparse import coo_matrix
u_tr, i_tr, r_tr = zip(*tr_ratings)
u_te, i_te, r_te= zip(*ts_ratings)
tr_sparse = coo_matrix((r_tr, (u_tr, i_tr)), shape=(n_users, n_items))
te_sparse = coo_matrix((r_te, (u_te, i_te)), shape=(n_users, n_items))


# In[24]:


tr_sparse


# ### Now Weighted Alternating Squares method by using Tensorflow

# In[336]:


import tensorflow as tf


# In[337]:


indice = list(zip(tr_sparse.row, tr_sparse.col))


# In[338]:


def make_wts(data, wt_type, obs_wt, feature_wt_exp, axis):
    """Generate observed item weights.
    Args:
    data:             coo_matrix of ratings data
    wt_type:          weight type, LOG_RATINGS or LINEAR_RATINGS
    obs_wt:           linear weight factor
    feature_wt_exp:   logarithmic weight factor
    axis:             axis to make weights for, 1=rows/users, 0=cols/items
    Returns:
    vector of weights for cols (items) or rows (users)
    """
    # recipricol of sum of number of items across rows (if axis is 0)
    frac = np.array(1.0/(data > 0.0).sum(axis))

    # filter any invalid entries
    frac[np.ma.masked_invalid(frac).mask] = 0.0

    # normalize weights according to assumed distribution of ratings
    if wt_type == "LOG_RATINGS":
        wts = np.array(np.power(frac, feature_wt_exp)).flatten()
    else:
        wts = np.array(obs_wt * frac).flatten()

    # check again for any numerically unstable entries
    assert np.isfinite(wts).sum() == wts.shape[0]
    return wts


# In[348]:


dim = 34
num_iters = 20
reg = 9.83
unobs = 0.001
wt_type = 1
feature_wt_factor =  189.8
obs_wt = 100
num_rows = tr_sparse.shape[0]
num_cols = tr_sparse.shape[1]


# In[341]:


##Note that data is sparse matrix created by coo_matrix
input_tensor = tf.SparseTensor(indices=indice,
                                values=(tr_sparse.data).astype(np.float32),
                                dense_shape=tr_sparse.shape)
assert feature_wt_exp is not None
row_wts = np.ones(num_rows)
col_wts = make_wts(tr_sparse, "LINEAR_RATINGS", feature_wt_factor , None, 0)


# In[342]:


model =  tf.contrib.factorization.WALSModel(num_rows, num_cols, dim,
                                    unobserved_weight=unobs,
                                    regularization=reg,
                                    row_weights=row_wts,
                                    col_weights=col_wts)


# In[343]:


# retrieve the row and column factors
row_factor = model.row_factors[0]
col_factor = model.col_factors[0]


# In[349]:


row_update_op = model.update_row_factors(sp_input=input_tensor)[1]
col_update_op = model.update_col_factors(sp_input=input_tensor)[1]
sess = tf.Session(graph=input_tensor.graph)
sess.run(model.initialize_op)
sess.run(model.worker_init)
for _ in range(num_iters):
    sess.run(model.row_update_prep_gramian_op)
    sess.run(model.initialize_row_update_op)
    sess.run(row_update_op)
    sess.run(model.col_update_prep_gramian_op)
    sess.run(model.initialize_col_update_op)
    sess.run(col_update_op)


# In[350]:


# evaluate output factor matrices
output_row = row_factor.eval(session=sess)
output_col = col_factor.eval(session=sess)
sess.close()


# In[351]:


def get_rmse(output_row, output_col, actual):
    """Compute rmse between predicted and actual ratings.
      Args:
        output_row: evaluated numpy array of row_factor
        output_col: evaluated numpy array of col_factor
        actual: coo_matrix of actual (test) values
      Returns:
        rmse
      """
    mse = 0
    for i in range(actual.data.shape[0]):
        row_pred = output_row[actual.row[i]]
        col_pred = output_col[actual.col[i]]
        err = actual.data[i] - np.dot(row_pred, col_pred)
        mse += err * err
    mse /= actual.data.shape[0]
    rmse = math.sqrt(mse)
    return rmse


# In[352]:


get_rmse(output_row, output_col, te_sparse)


# In[353]:


from skopt.space import Real, Integer
from skopt.utils import use_named_args

