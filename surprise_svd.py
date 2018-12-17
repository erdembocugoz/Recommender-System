
# coding: utf-8

# In[14]:


from surprise import SVD
from surprise import *
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV
import pandas as pd 
import numpy as np
import math
from implementations import *
import pickle


# In[15]:


#Load Train& Test Data


# In[23]:


df = pd.read_csv("data_train.csv")
df_test = pd.read_csv("data_test.csv")


# In[24]:


df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
df['Rating'] = df['Prediction']
df = df.drop(['Id', 'Prediction'], axis=1)


# In[25]:


df_test['User'] = df_test['Id'].apply(lambda x: int(x.split('_')[0][1:]))
df_test['Movie'] = df_test['Id'].apply(lambda x: int(x.split('_')[1][1:]))
df_test['Rating'] = df_test['Prediction']
df_test = df_test.drop(['Id', 'Prediction'], axis=1)


# In[26]:


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


# In[27]:


test_file = 'tmp_test.csv'
header = ['user_id','item_id','rating']
df_test.to_csv(test_file, index=False, header=False)
ratings_df_test = pd.read_csv('tmp_test.csv',
                          sep=',',
                         names=header,
                         
                         dtype={
                           'user_id': np.int32,
                           'item_id': np.int32,
                           'rating': np.float32,
                           'timestamp': np.int32,
                         })



# In[28]:


reader = Reader(line_format='user item rating', sep=',')

# Train and test set for Surprise
# Load the data
data = Dataset.load_from_file(train_file, reader=reader)



# In[30]:


reader = Reader(line_format='user item rating', sep=',')

# Train and test set for Surprise
# Load the data
data_test = Dataset.load_from_file(test_file, reader=reader)



# In[38]:


from surprise.model_selection import PredefinedKFold

fold = [(train_file, test_file)]
data = Dataset.load_from_folds(fold, reader=reader)
pkf = PredefinedKFold()
    # Algorithm
algo = SVDpp(n_epochs=30,lr_all=0.001,reg_all=0.001)

    # Go through 1 fold
for trainset, testset in  pkf.split(data):
    # Train
    algo.fit(trainset)

    # Predict
    predictions = algo.test(testset)


# In[46]:


pred = np.zeros(len(predictions))
for i in range(len(predictions)):
    val = predictions[i].est
    if val > 5:
        pred[i] = 5
    elif val < 1:
        pred[i] = 1
    else:
        pred[i] = val

# Copy the test
df_return = df_test.copy()

df_return.Rating = np.round(pred)


# In[58]:


df_return.Rating = (np.round(pred)).astype(int)


# In[59]:


submission = submission_table(df_return, 'User', 'Movie', 'Rating')


# In[60]:


file_name = 'prediction_svdpp.csv'
submission.to_csv(file_name, index=False)
with open('fm_model.pkl', 'wb') as f:
    pickle.dump(algo, f)

