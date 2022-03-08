"""
Converts |user_id|item_id|is_train| to 2 txt files for LightGCN model
"""

import pandas as pd
import numpy as np
from typing import Sequence
from sklearn.preprocessing import LabelEncoder
import os
import gc
from tqdm import tqdm
import joblib


def get_encoder(array: Sequence):
    encoder = LabelEncoder()
    encoder.fit(array)

    return encoder

def write_to_txt(df,path):
    """
    |user_id|item_id|
    |  int  |  int  |
    -----------------
    |
    |
    |
    |
    v
    txt file where each row is:
    user_id item1 item2 item3 ... itemN
    """
    try:
        tqdm.pandas()
        df = df.groupby(['user_map'])['item_map'].progress_apply(list).reset_index()
        df['item_map'] = df['item_map'].progress_apply(lambda x: ' '.join(list(map(str, x))))
    except:
        df = df.groupby(['user_map'])['item_map'].apply(list).reset_index()
        df['item_map'] = df['item_map'].apply(lambda x: ' '.join(list(map(str, x))))
    df.to_csv(path, header=None, sep=' ', index=False)

    
path_for_save = 'my_data/'

path_to_transactions = 'my_transactions.csv'  # path to transactions |user_id|item_id|is_train|
interactions = pd.read_csv(path_to_transactions)



user_column = 'user_id'
item_column = 'instrument_tiker'

#drop "new" users
train_users = interactions[interactions['is_train']][user_column].unique()

new_uids = interactions[~interactions[user_column].isin(train_users)][user_column]
interactions.drop(interactions[interactions[user_column].isin(new_uids)].index, inplace=True)


user_encoder = get_encoder(interactions[user_column])
item_encoder = get_encoder(interactions[item_column])

firs_time = False
#save encoders
joblib.dump(user_encoder,f'{path_for_save}user_encoder.pkl')
joblib.dump(item_encoder,f'{path_for_save}item_encoder.pkl')

user_ids = user_encoder.transform(interactions[user_column])
item_ids = item_encoder.transform(interactions[item_column])

df = pd.DataFrame()
df['user_map'] = user_ids
df['item_map'] = item_ids
df['is_train'] = interactions['is_train'].values


items_labels = item_encoder.transform(interactions[item_column].unique())
items = interactions[item_column].unique()
item_list = pd.DataFrame()
item_list['org_id'] = items
item_list['remap_id'] = items_labels
item_list = item_list.sort_values(by='remap_id')
item_list.to_csv(f'{path_for_save}item_list.txt', header=item_list.columns, index=None, sep=' ', mode='a')

print('Item mapping is saved')


user_labels = user_encoder.transform(interactions[user_column].unique())
users = interactions[user_column].unique()
user_list = pd.DataFrame()
user_list['org_id'] = users
user_list['remap_id'] = user_labels
user_list = user_list.sort_values(by='remap_id')
user_list.to_csv(f'{path_for_save}user_list.txt', header=item_list.columns, index=None, sep=' ', mode='a')

print('User mapping is saved')
del interactions
gc.collect()

train_df = df[df['is_train']]

write_to_txt(train_df,f'{path_for_save}train.txt')
print('Train data is saved')


test_df = df[~df['is_train']]

write_to_txt(test_df,f'{path_for_save}test.txt')


print('Data is ready')


