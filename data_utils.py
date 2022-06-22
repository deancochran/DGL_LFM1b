import os
import random
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

def getType(col):
    '''
    Description:
    getType returns dtype of a specified column name
    
    Params:
    col (str) - the specified column name of a list of data in the LFM1b dataset

    Returns:
    (str) - dtype of 'str' or 'int'
    '''
    if col in ['user_id', 'age', 'playcount', 'registered_unixtime','album_id', 'artist_id','track_id','last_played', 'genre_id']:
        return 'int'
    else:
        return 'str'

def setType(x, col):
    '''
    Description:
    setType returns a converted value corresponding to the specified column name
    
    Params:
    x - the specified valuse that needs to be converted
    col (str) - the specified column name of a list of data in the LFM1b dataset

    Returns:
    (str/int)- value of converted input
    '''
    if col in ['user_id', 'age', 'playcount', 'registered_unixtime','album_id', 'artist_id','track_id','last_played', 'genre_id']:
        return int(x)
    else:
        return str(x)

def isValid(x,col):
    '''
    Description:
    isValid returns bool if value corresponding to the specified column name is of the correct dtype
    
    Params:
    x - the specified valuse that needs to be converted
    col (str) - the specified column name of a list of data in the LFM1b dataset

    Returns:
    (bool)- value indicasting a in valid value (True=Invalid)
    '''
    if col in ['user_id', 'age', 'playcount', 'registered_unixtime','album_id', 'artist_id','track_id','last_played', 'genre_id']:
        try:
            x=setType(x, col)
            return True
        except:
            return False
    else:
        try:
            x=setType(x, col)
            return True
        except:
            return False
        
def get_col_names(type):
    '''
    Description:
    get_col_names returns a list of the column names corresponding to the type of data being observed
    
    Params:
    type (str) - the specified subscript of the txt file ('user', 'artist', 'album', 'track', 'le', 'genre')

    Returns:
    (list)- a array of str column anmes correponding to the data file
    '''
    if type=='user':
        return ['user_id', 'country', 'age', 'gender', 'playcount', 'registered_unixtime']
    elif type=='album':
        return ['album_id', 'album_name', 'artist_id']
    elif type=='artist':
        return ['artist_id', 'artist_name']
    elif type=='track':
        return ['track_id', 'track_name', 'artist_id']
    elif type=='le':
        return ['user_id', 'artist_id', 'album_id', 'track_id', 'timestamp']
    elif type=='genre':
        return ['genre_id', 'genre_name']
    else:
        raise Exception('bad "type" parameter in get_col_names')


def get_fileSize(path):
    '''
    Description:
    get_fileSize returns a integer val of te size of any specified file
    
    Params:
    path (str) - the specified path of the file to size

    Returns:
    (int)- a int value describing the byte size of the file
    '''
    proc=subprocess.Popen(f'wc -l {path}', shell=True, stdout=subprocess.PIPE)
    return int(bytes.decode(proc.communicate()[0]).split(' ')[0])

def get_raw_df(file_path, type):
    '''
    Description:
    get_raw_df returns a pandas dataframe that the file path and type name correspond to
    
    Params:
    file_path (str) - the specified path of the file 
    type (str) - the type subscript of the file ('user', 'artist', 'album', 'track', 'le', 'genre')

    Returns:
    (pandas>Dataframe)- a dataframe of the files information
    '''
    chunksize=100000 # set chunksize
    if len(get_col_names(type))==6: # if there is a header on the first line of the txt file
        df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', header = 0, chunksize=chunksize)
    else:# no header on the first line of the txt file
        df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', chunksize=chunksize)

    # print(f'loading df at {file_path}')
    chunks=[]
    size = get_fileSize(file_path) # get size of file
    for chunk in tqdm(df_chunks, total=size//chunksize): # load tqdm progress 
        for col in chunk: 
            try:
                chunk[col]=chunk[col].astype(getType(col)).copy() # try to set type
            except:
                try:
                    chunk[col]=[setType(x,col) for x in chunk[col]].copy() # try to force type
                    chunk[col]=chunk[col].astype(getType(col)).copy()
                except: # remove bad vals and force type
                    bad_vals=[]
                    for x in chunk[col].unique().tolist():
                        if isValid(x,col) ==False:
                            bad_vals.append(x)
                    bad_vals=np.unique(bad_vals)
                    chunk=chunk[~chunk[col].isin(bad_vals)].copy()
                    chunk[col]=[setType(x,col) for x in chunk[col]]
                    chunk[col]=chunk[col].astype(getType(col)).copy()
        # all the columns are of the correct datatype now
        chunks.append(chunk) # append chunk in preparation for concatenation
    df = pd.concat(chunks) # concatenate chunks 
    return df

def get_bad_ids(key_ids, ids_list, type):
    '''
    Description:
    get_bad_ids returns a list of ids of type id_type whose id is not in the id_type file, but is in the list of all available ids
    
    Params:
    key_ids (list) - the array of ids that should be unique, no other ids should exist in the database
    ids_list (list) - the array of arrays that indicate all other collections of the key ids found
    type (str) - the type subscript of the file ('user', 'artist', 'album', 'track', 'le', 'genre')

    Returns:
    (list)- a list of all ids of id_type that should be removed from the rest of the database
    '''
    id_type = type+'_id'
    found_ids=[]
    for list in ids_list:
        found_ids+=[int(x) for x in list if isinstance(x,int)]
    found_ids=np.unique(found_ids)
    df=pd.DataFrame({id_type:pd.Series(found_ids)})
    result = df[~df[id_type].isin(key_ids)][id_type].tolist()
    return result 

def get_bad_name_ids(file_path, type):
    '''
    Description:
    get_bad_name_ids returns a list of ids of type id_type whose name is unparsable
    
    Params:
    file_path (str) - the specified path of the file 
    type (str) -  the string abbreviation name of the file

    Returns:
    (list)- a list of ids of type id_type whose name is unparsable
    '''
    col_name_id = type+'_name'
    chunksize=100000 # set chunksize 
    if len(get_col_names(type))==6: # if there is a header on the first line of the txt file
        df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', header = 0, chunksize=chunksize)
    else:# no header on the first line of the txt file
        df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', chunksize=chunksize)
    # if we only want the values in certain columns
    bad_ids=[] 
    size = get_fileSize(file_path) # get size of file
    for chunk in tqdm(df_chunks, total=size//chunksize): # load tqdm progress 
        for col in chunk: 
            try:
                chunk[col]=chunk[col].astype(getType(col)).copy() # try to set type
            except:
                try:
                    chunk[col]=[setType(x,col) for x in chunk[col]].copy() # try to force type
                    chunk[col]=chunk[col].astype(getType(col)).copy()
                except: # remove bad vals and force type
                    bad_vals=[]
                    for x in chunk[col].unique().tolist():
                        if isValid(x,col) ==False:
                            bad_vals.append(x)
                    bad_vals=np.unique(bad_vals)
                    chunk=chunk[~chunk[col].isin(bad_vals)].copy()
                    chunk[col]=[setType(x,col) for x in chunk[col]]
                    chunk[col]=chunk[col].astype(getType(col)).copy()
        chunk=chunk[chunk[col_name_id]=='nan']
        bad_ids+=list(chunk[type+'_id'].values) # set unique values of the column 
    return list(set(bad_ids))

def get_raw_ids(file_path, type, return_unique_ids=False, id_list=None):
    '''
    Description:
    get_raw_ids returns a dictionary of the specified columns explitly for the non preprocessed LFM1b files. 
    Additionally, if indicated, the function will return a unique list of the ids found.
    
    Params:
    file_path (str) - the specified path of the file 
    type (str) -  the string abbreviation name of the file ('user', 'artist', 'album', 'track', 'le', 'genre')
    return_unique_ids (bool) - inidcator if unique ids are wanted
    id_list (list) - a list holding the column names that will be returned ['artist_id', 'album_id', 'track_id, user_id']

    Returns:
    (dict)- a dictionry of ids with the column names specified in the id_list as keys
    '''
    print(f'---------------------------- Loading Raw {type} file  ----------------------------')
    chunksize=100000 # set chunksize 
    if len(get_col_names(type))==6: # if there is a header on the first line of the txt file
        df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', header = 0, chunksize=chunksize)
    else:# no header on the first line of the txt file
        df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', chunksize=chunksize)
    # if we only want the values in certain columns
    ids_dict={k: list() for k in id_list}
    size = get_fileSize(file_path) # get size of file
    for chunk in tqdm(df_chunks, total=size//chunksize): # load tqdm progress 
        for col in chunk: 
            try:
                chunk[col]=chunk[col].astype(getType(col)).copy() # try to set type
            except:
                try:
                    chunk[col]=[setType(x,col) for x in chunk[col]].copy() # try to force type
                    chunk[col]=chunk[col].astype(getType(col)).copy()
                except: # remove bad vals and force type
                    bad_vals=[]
                    for x in chunk[col].unique().tolist():
                        if isValid(x,col) ==False:
                            bad_vals.append(x)
                    bad_vals=np.unique(bad_vals)
                    chunk=chunk[~chunk[col].isin(bad_vals)].copy()
                    chunk[col]=[setType(x,col) for x in chunk[col]]
                    chunk[col]=chunk[col].astype(getType(col)).copy()
        if return_unique_ids:
            for id_col in id_list: # for every column we want to return
                ids_dict[id_col]+=list(chunk[id_col].unique()) # set unique values of the column 
        else:
            for id_col in id_list: # for every column we want to return
                ids_dict[id_col]+=list(chunk[id_col].values) # set unique values of the column 
            
    if return_unique_ids: 
        for k,v in ids_dict.items(): 
            ids_dict[k]=list(set(v)) # make final unique values of every column in case of redundant ids
        return ids_dict
    else:
        return ids_dict

def get_preprocessed_ids(file_path, type, return_unique_ids=False, id_list=None):
    '''
    Description:
    This is the same function as get_raw_ids except, this assumes that the file is in the preprocessed directory and has a header row. (See get_raw_ids)
    
    Params:
    file_path (str) - the specified path of the file 
    type (str) -  the string abbreviation name of the file ('user', 'artist', 'album', 'track', 'le', 'genre')
    return_unique_ids (bool) - inidcator if unique ids are wanted
    id_list (list) - a list holding the column names that will be returned ['artist_id', 'album_id', 'track_id, user_id']

    Returns:
    (dict)- a dictionry of ids with the column names specified in the id_list as keys
    '''
    # print(f'---------------------------- Loading Preprocessed {type} file  ----------------------------')
    chunksize=100000 # set chunksize
    df_chunks = pd.read_csv(file_path, names=get_col_names(type), sep="\t", encoding='utf8', header = 0, chunksize=chunksize)
    # if we only want the values in certain columns
    ids_dict={k: list() for k in id_list}
    size = get_fileSize(file_path) # get size of file
    for chunk in tqdm(df_chunks, total=size//chunksize): # load tqdm progress 
        for col in chunk: 
            try:
                chunk[col]=chunk[col].astype(getType(col)).copy() # try to set type
            except:
                raise Exception('type not correct in preprocessed file')
        if return_unique_ids:
            # all the columns are of the correct datatype now
            for id_col in id_list: # for every column we want to return
                ids_dict[id_col]+=list(chunk[id_col].unique().tolist()) # set unique values of the column 
        else:
            for id_col in id_list: # for every column we want to return
                ids_dict[id_col]+=list(chunk[id_col].values) # set unique values of the column 
    if return_unique_ids: 
        for k,v in ids_dict.items(): 
            ids_dict[k]=list(set(v)) # make final unique values of every column in case of redundant ids
        return ids_dict
    else:
        return ids_dict

global verifyIds_count
verifyIds_count=0

def get_le_playcount(file_path, type, user_mapping, groupby_mapping, relative_playcount=False):
    '''
    Description:
    get_le_playcount returns a list of users, destination_node_ids, and playcounts.
    Additionally, if specified the fucntion will return a normalized relative playcount value for each instance if specified
    
    Params:
    file_path (str) - the specified path of the file 
    type (str) -  the string abbreviation name of the file ('user', 'artist', 'album', 'track', 'le', 'genre')
    return_unique_ids (bool) - inidcator if unique ids are wanted
    id_list (list) - a list holding the column names that will be returned ['artist_id', 'album_id', 'track_id, user_id']

    Returns:
    (dict)- a dictionry of ids with the column names specified in the id_list as keys
    '''
    type_id = type+'_id'
    # print(f'loading {type} playcounts for every user')
    chunksize=10000000
    df_chunks = pd.read_csv(file_path, names=get_col_names('le'), sep="\t", encoding='utf8', header = 0, chunksize=chunksize)
    size = get_fileSize(file_path) # get size of file
    playcount_dict={}
    total_user_plays={}
    for chunk in tqdm(df_chunks, total=size//chunksize): # load tqdm progress 
        for col in chunk: 
            chunk[col]=chunk[col].astype(getType(col)).copy() 
        for user_id, groupby_id in zip(chunk['user_id'].values, chunk[type_id].values): 
            try:
                user_id=user_mapping[user_id]
                groupby_id=groupby_mapping[groupby_id]
                if (user_id,groupby_id) not in playcount_dict.keys():
                    playcount_dict[(user_id,groupby_id)]=1  
                else:
                    playcount_dict[(user_id,groupby_id)]+=1

                if user_id not in total_user_plays.keys():
                    total_user_plays[user_id]=1  
                else:
                    total_user_plays[user_id]+=1
            except:
                pass    
    
    if relative_playcount:
        return [user_id for user_id, groupby_id in playcount_dict.keys()], [groupby_id for user_id, groupby_id in playcount_dict.keys()], [val/total_user_plays[u_id] for (u_id,g_id), val in playcount_dict.items()]
    else:
        return [user_id for user_id, groupby_id in playcount_dict.keys()], [groupby_id for user_id, groupby_id in playcount_dict.keys()], [val for (u_id,g_id), val in playcount_dict.items()]


def remap_ids(col_dict, ordered_cols, mappings):
    '''
    Description:
    remap_ids a dictionary of the input dict such that for every column name and mapping specified the values are remapped accordingly
    
    
    Params:
    col_dict (dict) - the specified dictionary of items
    ordered_cols (list) -  a list of the columns that should be remapped
    mappings (list) - a list of every mapping to remap the columns with

    Returns:
    (dict)- a dictionry of the remapped items
    '''
    new_mapping={col_name:list() for col_name in col_dict.keys()}
    length=len(col_dict[ordered_cols[0]])
    skipped=0
    for row_index in tqdm(range(length), total=length):
        bad_row=False
        for mapping, col_name in zip(mappings,ordered_cols):
            try:
                val=col_dict[col_name][row_index]
                new_mapping[col_name].append(mapping[val])
            except:
                bad_row=True
        if bad_row ==False:
            for col_name in col_dict.keys():
                if col_name not in ordered_cols:
                    val=col_dict[col_name][row_index]
                    new_mapping[col_name].append(val)
        else:
            skipped+=1
    # print(f'remapped {col_dict.keys()} skipped {skipped} bad ids')
    return new_mapping


def get_artist_genre_df(artist_genres_allmusic_path, artist_name_to_id_mapping):
    '''
    Description:
    get_artist_genre_df is a special function that reads into the genre collections of provied by the LFM1b databased 
    and returns the dataframe of the specified file


    Params:
    artist_genres_allmusic_path (str) - the specified path to the file
    artist_name_to_id_mapping (mapping) -  a unique mapping of artist names to already mapped artist ids

    Returns:
    (pandas.Dataframe)- a dataframe of the remapped items
    '''
    file=open(artist_genres_allmusic_path, 'r')
    lines=file.readlines()
    data={'artist_id':list(),'genre_id':list()}
    
    for line in lines:
        info=line.strip().split('\t')
        name=str(info[0])
        genre_list=np.array([int(x) for x in info[1:]])
        if len(genre_list) != 0 and name in artist_name_to_id_mapping.keys():
            for genre in genre_list:
                data['artist_id'].append(artist_name_to_id_mapping[name])
                data['genre_id'].append(genre)

    return pd.DataFrame(data)



def filterRaw(type, ids, df_path, output_path, fix_user_entires=False, artist_ids=None):
    print(f'----------------------------                 Filtering Original {type} File                    ----------------------------')
    col_names=get_col_names(type)
    df = get_raw_df(df_path, type=type)
    df = df[df[col_names[0]].isin(ids)].copy()
    if artist_ids!=None:
        df = df[df['artist_id'].isin(artist_ids)].copy()
    if fix_user_entires == True:
        df['country']=df['country'].replace('nan','NoCountry').copy()
        df['gender']=df['gender'].replace('nan','NoGender').copy()
    df.to_csv(output_path, columns=col_names, sep="\t", encoding='utf8', mode='w', index=False, line_terminator='\n')

def filterLEs(input_path, type, output_path, bad_ids, cols_to_filter, load_raw=True):
    ''' 
    the filterLEs reads the LEs from a specified input file and filters the ids based on each 
    list in bad_ids specified by the id column in cols_to_filter. After filtered the data is saved in the output path
    '''
    print(f'----------------------------                 Filtering Les                    ----------------------------')
    chunksize=10000
    column_names=get_col_names(type='le')
    if load_raw==False:
        df_chunks = pd.read_csv(input_path, names=column_names, sep="\t", encoding='utf8', header = 0, chunksize=chunksize)
    else:
        df_chunks = pd.read_csv(input_path, names=column_names, sep="\t", encoding='utf8', chunksize=chunksize)
    size = get_fileSize(input_path)
    for i, chunk in enumerate(tqdm(df_chunks, total=size//chunksize)):
        for col, ids_list in zip(cols_to_filter,bad_ids):
            try:
                chunk[col]=chunk[col].astype(getType(col)).copy()
            except:
                try:
                    chunk[col]=[setType(x,col) for x in chunk[col]].copy()
                    chunk[col]=chunk[col].astype(getType(col)).copy()
                except:
                    bad_vals=[]
                    for x in chunk[col].unique().tolist():
                        if isValid(x,col) ==False:
                            bad_vals.append(x)
                    bad_vals=np.unique(bad_vals)
                    # print('chunk bad vals',bad_vals)
                    chunk=chunk[~chunk[col].isin(bad_vals)].copy()
                    chunk[col]=[setType(x,col) for x in chunk[col]]
                    chunk[col]=chunk[col].astype(getType(col)).copy()
            chunk = chunk[~chunk[col].isin(ids_list)].copy()
        if i==0:
            chunk.to_csv(output_path, columns=get_col_names(type), sep="\t", encoding='utf8', index=False, header=True, mode='w')
        else:
            chunk.to_csv(output_path, columns=get_col_names(type), sep="\t", encoding='utf8', index=False, header=False, mode='a')

def get_user_info(user_info, u_id, country_percs, gender_percs):
    u_id_index=user_info['user_id'].index(u_id)

    u_id_country=user_info['country'][u_id_index]
    if u_id_country not in country_percs:
        u_id_country='NA'

    u_id_age=user_info['age'][u_id_index]
    if u_id_age < 0:
        u_id_age='NA'

    u_id_gender=user_info['gender'][u_id_index]
    if u_id_gender not in gender_percs:
        u_id_gender='n'

    u_id_playcount=user_info['playcount'][u_id_index]

    return u_id_country, u_id_age, u_id_gender, u_id_playcount


def preprocess_raw(raw_path, preprocessed_path, n_users=None):
    ''' 
    Description:
    The preprocess_raw fucniton works in two ways....

    1) When called by the process method inside the LFM1b DGLDataset class to pre process 
    and save all the listening events, users, artists, slbums, and tracks such that
    a fully connected database is stored in the prprocessed file direcotry

    2) when the number of users is specified this function will produce a subset of the full 
    LFM1b dataset according to the distrubtion of users country age, gender, and playcount

        Additionally, the subset does not evaluate the ditribution of unqieu artists, albums, and or tracks that the collected users interact with
        The fundamental purpose of the subset is primarly to offer a simplified full connected graph that can be utilized for graph neural network analysis,
        without compromizing the current distributions of the datset that may cause bias, unfair classification/prediciton, or privacy issues within the dataset. 

    Parameters:
    raw_path (str) - the string path to the raw LFM1b directory holding the contents of the LFM1b.zip file
    preprocessed_path (str) - the string path to the preprocessed LFM1b directory where new txt files will be stored
    n_users (int) - an integer to specify the number of users to collect for a specified subsets
    '''
    
    les_raw_path=raw_path+'/LFM-1b_LEs.txt'
    artists_raw_path=raw_path+'/LFM-1b_artists.txt'
    albums_raw_path=raw_path+'/LFM-1b_albums.txt'
    tracks_raw_path=raw_path+'/LFM-1b_tracks.txt'
    users_raw_path=raw_path+'/LFM-1b_users.txt'

    les_pre_path=preprocessed_path+'/LFM-1b_LEs.txt'
    artists_pre_path=preprocessed_path+'/LFM-1b_artists.txt'
    albums_pre_path=preprocessed_path+'/LFM-1b_albums.txt'
    tracks_pre_path=preprocessed_path+'/LFM-1b_tracks.txt'
    users_pre_path=preprocessed_path+'/LFM-1b_users.txt'

    condition= os.path.exists(preprocessed_path+'/LFM-1b_LEs.txt') == False
    if condition == True and n_users==None:
        print(f'making subset of all users')

        unique_le_ids = get_raw_ids(les_raw_path, type='le', return_unique_ids=True, id_list=['artist_id', 'album_id', 'track_id','user_id'])
        artist_id_dict = get_raw_ids(artists_raw_path, type='artist', return_unique_ids=True, id_list=['artist_id'])
        bad_artist_name_ids = get_bad_name_ids(artists_raw_path, type='artist')
        album_id_dict=get_raw_ids(albums_raw_path, type='album', return_unique_ids=True, id_list=['album_id','artist_id'])
        bad_album_name_ids = get_bad_name_ids(albums_raw_path, type='album')
        track_id_dict=get_raw_ids(tracks_raw_path, type='track', return_unique_ids=True, id_list=['track_id','artist_id'])
        bad_track_name_ids = get_bad_name_ids(tracks_raw_path, type='track')
        
        print('---------------------------- Filtering All Bad Ids From LEs and Collecting remaining "ids"   ----------------------------')
        total_bad_artist_ids = np.unique(get_bad_ids(artist_id_dict['artist_id'], [album_id_dict['artist_id'],track_id_dict['artist_id'], unique_le_ids['artist_id']], type='artist')+bad_artist_name_ids)
        total_bad_album_ids = np.unique(get_bad_ids(album_id_dict['album_id'], [unique_le_ids['album_id']], type='album')+bad_album_name_ids)
        total_bad_track_ids = np.unique(get_bad_ids(track_id_dict['track_id'], [unique_le_ids['track_id']], type='track')+bad_track_name_ids)
        filterLEs(les_raw_path, type='le', output_path=les_pre_path, bad_ids=[total_bad_artist_ids,total_bad_album_ids,total_bad_track_ids], cols_to_filter=['artist_id','album_id','track_id'])
        unique_le_ids = get_preprocessed_ids(les_pre_path, type='le', return_unique_ids=True, id_list=['artist_id', 'album_id', 'track_id','user_id'])
        
        filterRaw('user', unique_le_ids['user_id'], users_raw_path, users_pre_path, fix_user_entires=True)
        filterRaw('artist', unique_le_ids['artist_id'], artists_raw_path, artists_pre_path)
        filterRaw('album', unique_le_ids['album_id'], albums_raw_path, albums_pre_path, artist_ids=unique_le_ids['artist_id'])
        filterRaw('track', unique_le_ids['track_id'], tracks_raw_path, tracks_pre_path, artist_ids=unique_le_ids['artist_id'])
    
        file_path=raw_path+'_UGP/genres_allmusic.txt'
        df = pd.read_csv(file_path, names=['genre_name'])
        df['genre_id']=df['genre_name'].index
        df=df.reindex(columns=['genre_id', 'genre_name'])
        output_path=preprocessed_path+'/genres_allmusic.txt'
        df.to_csv(output_path, columns=get_col_names('genre'), sep="\t", encoding='utf8', mode='w', index=False, line_terminator='\n')

        del df
    else:
        n_users=int(n_users)
        print(f'making subset of {n_users} users')

        required_country_count={
            'US':18.581,
            'RU':9.103,
            'DE':8.295,
            'UK':8.295,
            'PL':7.987,
            'BR':7.041,
            'FI':2.553,
            'NL':2.491,
            'ES':2.252,
            'SE':2.230,
            'UA':2.071,
            'CA':1.951,
            'FR':1.912,
            'NA':54.131
        }
        required_gender_count={
            'm':33.218,
            'f':13.13,
            'n':53.649
        }
        required_valid_age_count={
            'NA':61.69,
            'good':38.31
        }
        age_dist={
            'mean':25.4,
            'std':9.7,
        }
        playcount_dist={
            'mean':15962,
            'std':8879
        }            
        
        required_country_count={k:round(int(float(v/100)*float(n_users))) for k,v in required_country_count.items()}
        size=sum(required_country_count.values())
        if size > n_users:
            while size != n_users:
                rand_key = random.choice([k for k,v in required_country_count.items()])  
                required_country_count[rand_key]-=1
                size=sum(required_country_count.values())

        required_gender_count={k:round(int(float(v/100)*float(n_users))) for k,v in required_gender_count.items()}
        size=sum(required_gender_count.values())
        if size < n_users:
            while size != n_users:
                rand_key = random.choice([k for k,v in required_gender_count.items()])  
                required_gender_count[rand_key]+=1
                size=sum(required_gender_count.values())

        
        required_valid_age_count={k:round(int(float(v/100)*float(n_users))) for k,v in required_valid_age_count.items()}
        size=sum(required_valid_age_count.values())
        if size < n_users:
            while size != n_users:
                rand_key = random.choice([k for k,v in required_valid_age_count.items()])  
                required_valid_age_count[rand_key]+=1
                size=sum(required_valid_age_count.values())

        user_info = get_raw_ids(users_raw_path, type='user', id_list=['user_id', 'country', 'age', 'gender', 'playcount'])
        user_info_ids=user_info['user_id'].copy()
        
        user_id_collection=[]
        pbar = tqdm(total=n_users)
        ids_left=len(user_info_ids)
        while len(user_id_collection)<n_users and ids_left > 0:
            ids_left=len(user_info_ids)
            random_index = random.choice(range(ids_left))  
            user_id = user_info_ids.pop(random_index)
            country, age, gender, playcount = get_user_info(user_info, user_id, required_country_count, required_gender_count)
            country_condition = required_country_count[country] != 0
            if str(age) == 'NA':
                age_key='NA'
                age_condition = required_valid_age_count[age_key] != 0
                age_dist_condition=True
            else:
                age_key='good'
                age_condition = required_valid_age_count[age_key] != 0
                age_dist_condition= (age > age_dist['mean'] - age_dist['std']) and (age < age_dist['mean'] + age_dist['std'])
            gender_condition = required_gender_count[gender] != 0
            playcount_condition = (playcount > playcount_dist['mean'] - playcount_dist['std']) and (playcount < playcount_dist['mean'] + playcount_dist['std'])

            if country_condition and age_condition and age_dist_condition and gender_condition and playcount_condition:
                required_country_count[country]-=1
                required_valid_age_count[age_key]-=1
                required_gender_count[gender]-=1
                user_id_collection.append(user_id)
                pbar.update(1)
            else:
                pass

        pbar.close()

        total_bad_user_ids=[]
        unique_le_user_id_ids_size=len(user_info['user_id'])
        for id in tqdm(user_info['user_id'], total=unique_le_user_id_ids_size):
            if id not in user_id_collection:
                total_bad_user_ids.append(id)

        unique_le_ids = get_raw_ids(les_raw_path, type='le', return_unique_ids=True, id_list=['artist_id', 'album_id', 'track_id','user_id'])
        artist_id_dict = get_raw_ids(artists_raw_path, type='artist', return_unique_ids=True, id_list=['artist_id'])
        bad_artist_name_ids = get_bad_name_ids(artists_raw_path, type='artist')
        album_id_dict=get_raw_ids(albums_raw_path, type='album', return_unique_ids=True, id_list=['album_id','artist_id'])
        bad_album_name_ids = get_bad_name_ids(albums_raw_path, type='album')
        track_id_dict=get_raw_ids(tracks_raw_path, type='track', return_unique_ids=True, id_list=['track_id','artist_id'])
        bad_track_name_ids = get_bad_name_ids(tracks_raw_path, type='track')
        
        print('---------------------------- Filtering All Bad Ids From LEs and Collecting remaining "ids"   ----------------------------')
        total_bad_artist_ids = np.unique(get_bad_ids(artist_id_dict['artist_id'], [album_id_dict['artist_id'],track_id_dict['artist_id'], unique_le_ids['artist_id']], type='artist')+bad_artist_name_ids)
        total_bad_album_ids = np.unique(get_bad_ids(album_id_dict['album_id'], [unique_le_ids['album_id']], type='album')+bad_album_name_ids)
        total_bad_track_ids = np.unique(get_bad_ids(track_id_dict['track_id'], [unique_le_ids['track_id']], type='track')+bad_track_name_ids)
        filterLEs(les_raw_path, type='le', output_path=les_pre_path, bad_ids=[total_bad_user_ids,total_bad_album_ids,total_bad_track_ids,total_bad_artist_ids], cols_to_filter=['user_id','track_id','album_id','artist_id'])
        unique_le_ids = get_preprocessed_ids(les_pre_path, type='le', return_unique_ids=True, id_list=['artist_id', 'album_id', 'track_id','user_id'])
        
        filterRaw('user', unique_le_ids['user_id'], users_raw_path, users_pre_path, fix_user_entires=True)
        filterRaw('artist', unique_le_ids['artist_id'], artists_raw_path, artists_pre_path)
        filterRaw('album', unique_le_ids['album_id'], albums_raw_path, albums_pre_path, artist_ids=unique_le_ids['artist_id'])
        filterRaw('track', unique_le_ids['track_id'], tracks_raw_path, tracks_pre_path, artist_ids=unique_le_ids['artist_id'])
    
        file_path=raw_path+'_UGP/genres_allmusic.txt'
        df = pd.read_csv(file_path, names=['genre_name'])
        df['genre_id']=df['genre_name'].index
        df=df.reindex(columns=['genre_id', 'genre_name'])
        output_path=preprocessed_path+'/genres_allmusic.txt'
        df.to_csv(output_path, columns=get_col_names('genre'), sep="\t", encoding='utf8', mode='w', index=False, line_terminator='\n')

        del df