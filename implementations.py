import pandas as pd
import numpy as np



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
    
def split(df, cut):
	""" split table into train and test set """
	size = df.shape[0]
	keys = list(range(size))
	np.random.shuffle(keys)

	key_cut = int(size * cut)
	train_key = keys[:key_cut]
	test_key = keys[key_cut:]

	test = df.loc[test_key]
	train = df.loc[train_key]

	return train, test

def load_data(fileName):
	df = pd.read_csv(fileName)
	df['User'] = df['Id'].apply(lambda x: int(x.split('_')[0][1:]))
	df['Movie'] = df['Id'].apply(lambda x: int(x.split('_')[1][1:]))
	df['Rating'] = df['Prediction']
	df = df.drop(['Id', 'Prediction'], axis=1)
	return df

def create_input_pyfm(df):
	data = []
	y = list(df.Rating)
	usrs = list(df.User)
	mvies = list(df.Movie)
	for i in range(len(df)):
		y[i] = float(y[i])
		data.append({"user_id": str(usrs[i]), "movie_id": str(mvies[i])})
	return data,np.array(y)
