import pandas as pd
import numpy as np

def gran_date(row: pd.Series) -> pd.Series:
    size = len(row['Date'])
    if size == 10:
        month = row['Date'].split('-')[1]
        return month
    else:
        return np.nan

def gran_node(row: pd.Series) -> pd.Series:
    return 'Major' if row['Mode']==1 else 'Minor'

def first_process_songs(songs: pd.DataFrame) -> pd.DataFrame:
    drop_columns = ['Track Name', 'Album Name', 'Markets', 'Date']
    songs = songs.drop(columns='Unnamed: 0').dropna()
    songs['Month'] = songs.apply(gran_date, axis=1)
    songs['Mode'] = songs.apply(gran_node, axis=1)
    songs = songs.drop(columns=drop_columns)
    return songs

def create_model_input_table(songs: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(songs, columns=['Artist Name', 'Key',
                        'Mode', 'TSignature', 'Month', 'Country'])

