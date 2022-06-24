import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import argparse
import torch as th

from dgl import heterograph
from dgl.data import DGLDataset, download, extract_archive
from dgl.data.utils import save_graphs, load_graphs
from .data_utils import get_artist_genre_df, get_le_playcount, get_les, get_preprocessed_ids, remap_ids, preprocess_raw, get_fileSize
from .encoders import CategoricalEncoder, IdentityEncoder, SequenceEncoder

class LFM1b(DGLDataset):
    def __init__(self, 
    hash_key=(), 
    force_reload=False, 
    verbose=False, 
    n_users=None, 
    device='cpu', 
    overwrite_preprocessed=False, 
    overwrite_processed=False,
    artists=True,
    albums=True,
    tracks=True,
    playcount_weight=False
    ):
        name='DGL_LFM1b'
        self.root_dir = './data/'+name
        self.preprocessed_dir = './data/'+name+'/preprocessed'
        self.raw_ugp_dir='./data/'+name+f'/{name}_UGP'
        self.lfm1b_ugp_url='http://www.cp.jku.at/datasets/LFM-1b/LFM-1b_UGP.zip'
        self.n_users=n_users
        self.device=device
        self.overwrite_preprocessed=overwrite_preprocessed
        self.overwrite_processed=overwrite_processed
        self.artists=artists
        self.albums=albums
        self.tracks=tracks
        self.playcount_weight=playcount_weight
        super().__init__(
            name=name, 
            url='http://drive.jku.at/ssf/s/readFile/share/1056/266403063659030189/publicLink/LFM-1b.zip', 
            raw_dir=self.root_dir+'/'+name, 
            save_dir=self.root_dir+'/processed',
            hash_key=hash_key, 
            force_reload=force_reload, 
            verbose=verbose
            ) 

    def download(self):
        """Download and extract Zip file from LFM1b"""
        if self.url is not None:
            extract_archive(download(self.url, path = self.root_dir, overwrite = False), target_dir = self.root_dir+'/'+self.name, overwrite = False)
            extract_archive(download(self.lfm1b_ugp_url, path = self.root_dir, overwrite = False), target_dir = self.root_dir, overwrite = False)            
            if not os.path.exists(self.preprocessed_dir):
                os.mkdir(self.preprocessed_dir)     
            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)              
        else:
            raise Exception("self.url is None! This should point to the LastFM1b zip download path: 'http://drive.jku.at/ssf/s/readFile/share/1056/266403063659030189/publicLink/LFM-1b.zip'")

    def load(self):
        """load graph list and graph labels with load_graphs"""
        return load_graphs(self.save_dir+'/lastfm1b.bin')

    def save(self):
        """save file to processed directory"""
        if os.path.exists(os.path.join(self.save_dir+'/lastfm1b.bin')) == False:
            print('saving graph...')
            glist=[self.graph]
            glabels={"glabel": th.tensor([0])}
            save_graphs(self.save_dir+'/lastfm1b.bin',glist,glabels)
            print('loading graph memory size....')
            size=get_fileSize(self.save_dir+'/lastfm1b.bin')
            print(f'graph is {size} bytes large')
            print('saved!')

    def process(self):
        if os.path.exists(self.save_dir+'/lastfm1b.bin') == True and self.overwrite_processed == True:
            os.remove(self.save_dir+'/lastfm1b.bin')
        processed_condition = os.path.exists(os.path.join(self.save_dir+'/lastfm1b.bin')) == False
        if processed_condition == True:
            preprocessed_condition = os.path.exists(os.path.join(self.preprocessed_dir+'/LFM-1b_LEs.txt')) == False or self.overwrite_preprocessed == True
            if preprocessed_condition == True:
                preprocess_raw(self.raw_dir,self.preprocessed_dir, n_users=self.n_users, overwrite=self.overwrite_preprocessed)
            
            graph_data = {}
            mappings={} 
            device = th.device(self.device)
            id_encoder = IdentityEncoder(device=device) # used to encode floats f(x)==2, where x = 2 
            # bin_encoder = BinaryEncoder(device=device)
            cat_encoder = CategoricalEncoder(device=device) # used to encode categories f(x)==[0,0,1,0], where x = 2, of possible types 0,1,2,3
            seq_encoder = SequenceEncoder(device=device) # used to encode strs f(x)==[0.213,0.254,...,134,.893], where x = 'dean', and shape is (1x254)
            
            
            # -------------------------USER MAPPING-------------------------
            file_path=self.preprocessed_dir+'/'+'LFM-1b_users.txt'
            print('\t','------------------- Loading Mapping Data from',file_path.split('_')[-1],'-------------------')
            df = get_preprocessed_ids(file_path, type='user', id_list=['user_id'])
            mappings['user_mapping']= {int(id): i for i, id in enumerate(df['user_id'])}

            # -------------------------ARTIST MAPPING -------------------------
            file_path=self.preprocessed_dir+'/'+'LFM-1b_artists.txt'
            print('\t','------------------- Loading Mapping Data from',file_path.split('_')[-1],'-------------------')
            df = get_preprocessed_ids(file_path, type='artist', id_list=['artist_id','artist_name'])
            mappings['artist_mapping'] = {int(id): i for i, id in enumerate(df['artist_id'])}
            df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
            mappings['artist_name_mapping'] = {artist_name: int(artist_id)  for artist_id, artist_name in zip(df['artist_id'],df['artist_name'])}
            del df

            # -------------------------ALBUM MAPPING -------------------------
            file_path=self.preprocessed_dir+'/'+'LFM-1b_albums.txt'
            print('\t','------------------- Loading Mapping Data from',file_path.split('_')[-1],'-------------------')
            df = get_preprocessed_ids(file_path, type='album', id_list=['album_id','artist_id'])
            df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
            mappings['album_mapping'] = {int(id): i for i, id in enumerate(df['album_id'])}
            del df

            # -------------------------TRACK MAPPING -------------------------
            file_path=self.preprocessed_dir+'/'+'LFM-1b_tracks.txt'
            print('\t','------------------- Loading Graph Data from',file_path.split('_')[-1],'-------------------')
            df = get_preprocessed_ids(file_path, type='track', id_list=['track_id', 'artist_id'])
            df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
            mappings['track_mapping'] = {int(id): i for i, id in enumerate(df['track_id'])}
            del df

            if self.artists==True:
                # -------------------------GENRES->ARTISTS GRAPH DATA-------------------------
                file_path=self.raw_ugp_dir+'/LFM-1b_artist_genres_allmusic.txt'
                print('\t','------------------- Loading Graph Data from',file_path.split('_')[-1],'-------------------')
                df = get_artist_genre_df(file_path, mappings['artist_name_mapping'])
                graph_data[('artist', 'in_genre', 'genre')]=(th.tensor(df['artist_id'].values), th.tensor(df['genre_id'].values))
                graph_data[('genre', 'is_genre_of', 'artist')]=(th.tensor(df['genre_id'].values), th.tensor(df['artist_id'].values))
                del df

                if self.albums==True:
                    # -------------------------ALBUMS->ARTISTS GRAPH DATA-------------------------
                    file_path=self.preprocessed_dir+'/'+'LFM-1b_albums.txt'
                    print('\t','------------------- Loading Graph Data from',file_path.split('_')[-1],'-------------------')
                    df = get_preprocessed_ids(file_path, type='album', id_list=['album_id','artist_id'])
                    df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
                    df=remap_ids(df, ordered_cols=['album_id'], mappings=[mappings['album_mapping']])
                    graph_data[('artist', 'produced', 'album')]=(th.tensor(df['artist_id']), th.tensor(df['album_id']))
                    graph_data[('album', 'produced_by', 'artist')]=(th.tensor(df['album_id']), th.tensor(df['artist_id']))
                    del df

                if self.tracks==True:
                    # -------------------------TRACKS->ARTISTS GRAPH DATA-------------------------
                    file_path=self.preprocessed_dir+'/'+'LFM-1b_tracks.txt'
                    print('\t','------------------- Loading Graph Data from',file_path.split('_')[-1],'-------------------')
                    df = get_preprocessed_ids(file_path, type='track', id_list=['track_id', 'artist_id'])
                    df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
                    df=remap_ids(df, ordered_cols=['track_id'], mappings=[mappings['track_mapping']])
                    graph_data[('track', 'preformed_by', 'artist')]=(th.tensor(df['track_id']), th.tensor(df['artist_id']))
                    graph_data[('artist', 'preformed', 'track')]=(th.tensor(df['artist_id']), th.tensor(df['track_id']))
                    del df

                # -------------------------USER->ARTISTS GRAPH DATA-------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_LEs.txt'
                print('\t','------------------- Loading Graph Data from',file_path.split('_')[-1],'-------------------')
                if self.playcount_weight:
                    playcounts, user_id_list, groupby_id_list=get_le_playcount(
                        file_path,type='artist',
                        user_mapping=mappings['user_mapping'], 
                        groupby_mapping=mappings['artist_mapping'],
                        relative_playcount=True
                        )
                    del playcounts
                else:
                    timestamps, user_id_list, groupby_id_list=get_les(
                        file_path,type='artist',
                        user_mapping=mappings['user_mapping'], 
                        groupby_mapping=mappings['artist_mapping']
                    )
                    del timestamps
                graph_data[('user', 'listened_to_artist', 'artist')]=(
                    th.tensor(user_id_list), 
                    th.tensor(groupby_id_list)
                        )
                graph_data[('artist', 'artist_listened_by', 'user')]=(
                        th.tensor(groupby_id_list), 
                        th.tensor(user_id_list)
                        )
                del user_id_list
                del groupby_id_list
                

            if self.albums==True:
                # -------------------------USER->ALBUMS GRAPH DATA-------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_LEs.txt'
                print('\t','------------------- Loading Graph Data from',file_path.split('_')[-1],'-------------------')
                if self.playcount_weight:
                    playcounts, user_id_list, groupby_id_list=get_le_playcount(
                        file_path,type='album',
                        user_mapping=mappings['user_mapping'], 
                        groupby_mapping=mappings['album_mapping'],
                        relative_playcount=True
                        )
                    del playcounts
                else:
                    timestamps, user_id_list, groupby_id_list=get_les(
                        file_path,type='album',
                        user_mapping=mappings['user_mapping'], 
                        groupby_mapping=mappings['album_mapping']
                    )
                    del timestamps
                graph_data[('user', 'listened_to_album', 'album')]=(
                    th.tensor(user_id_list), 
                    th.tensor(groupby_id_list)
                    )
                graph_data[('album', 'album_listened_by', 'user')]=(
                    th.tensor(groupby_id_list), 
                    th.tensor(user_id_list)
                    )
                del user_id_list
                del groupby_id_list
                
            if self.tracks==True:
                # -------------------------USER->TRACKS GRAPH DATA-------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_LEs.txt'
                print('\t','------------------- Loading Graph Data from',file_path.split('_')[-1],'-------------------')
                if self.playcount_weight:
                    playcounts, user_id_list, groupby_id_list=get_le_playcount(
                        file_path,type='track',
                        user_mapping=mappings['user_mapping'], 
                        groupby_mapping=mappings['track_mapping'],
                        relative_playcount=True
                        )
                    del playcounts
                else:
                    timestamps, user_id_list, groupby_id_list=get_les(
                        file_path,type='track',
                        user_mapping=mappings['user_mapping'], 
                        groupby_mapping=mappings['track_mapping']
                    )
                    del timestamps
                graph_data[('user', 'listened_to_track', 'track')]=(
                    th.tensor(user_id_list), 
                    th.tensor(groupby_id_list)
                    )
                graph_data[('track', 'track_listened_by', 'user')]=(
                    th.tensor(groupby_id_list), 
                    th.tensor(user_id_list)
                    )
                del user_id_list
                del groupby_id_list
                

            # -------------------------DGL HETERO GRAPH OBJECT-------------------------
            print('\t','-------------------  Creating DGL HeteroGraph from Graph Data  -------------------')
            self.graph = heterograph(graph_data)
            print(self.graph)
            del graph_data

            if self.artists==True:
                # -------------------------ARTIST NODE DATA-------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_artists.txt'
                print('\t','------------------- Loading features from',file_path.split('_')[-1],'-------------------')
                df = get_preprocessed_ids(file_path, type='artist', id_list=['artist_id','artist_name'])
                df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
                self.graph.nodes['artist'].data['id']=cat_encoder(df['artist_id'])
                self.graph.nodes['artist'].data['name']=seq_encoder(df['artist_name'])
                del df

                # -------------------------GENRE NODE DATA-------------------------
                file_path=self.preprocessed_dir+'/'+'genres_allmusic.txt'
                print('\t','------------------- Loading features from',file_path.split('_')[-1],'-------------------')
                df = get_preprocessed_ids(file_path,type='genre' ,id_list=['genre_id','genre_name'])
                self.graph.nodes['genre'].data['id']=cat_encoder(df['genre_id'])
                self.graph.nodes['genre'].data['name']=seq_encoder(df['genre_name'])
                del df
                del mappings['artist_name_mapping']
            
            if self.albums==True:
                # -------------------------ALBUM ID RE-MAPPING-------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_albums.txt'
                print('\t','------------------- Loading features from',file_path.split('_')[-1],'-------------------')
                df = get_preprocessed_ids(file_path, type='album', id_list=['album_id','album_name','artist_id'])
                df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
                df=remap_ids(df, ordered_cols=['album_id'], mappings=[mappings['album_mapping']])
                # -------------------------ALBUM NODE DATA-------------------------
                self.graph.nodes['album'].data['id']=cat_encoder(df['album_id'])
                self.graph.nodes['album'].data['name']=seq_encoder(df['album_name'])
                del df

            if self.tracks==True:
                # -------------------------TRACK ID RE-MAPPING-------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_tracks.txt'
                print('\t','------------------- Loading features from',file_path.split('_')[-1],'-------------------')
                df = get_preprocessed_ids(file_path, type='track', id_list=['track_id', 'track_name', 'artist_id'])
                df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
                df=remap_ids(df, ordered_cols=['track_id'], mappings=[mappings['track_mapping']])
                # -------------------------TRACK NODE DATA-------------------------
                self.graph.nodes['track'].data['id']=cat_encoder(df['track_id'])
                self.graph.nodes['track'].data['name']=seq_encoder(df['track_name'])

                del df

            # -------------------------USER ID RE-MAPPING-------------------------
            file_path=self.preprocessed_dir+'/'+'LFM-1b_users.txt'
            print('\t','------------------- Loading features from',file_path.split('_')[-1],'-------------------')
            df = get_preprocessed_ids(file_path, type='user', id_list=['user_id','country','age','gender','playcount'])
            df=remap_ids(df, ordered_cols=['user_id'], mappings=[mappings['user_mapping']])
            # -------------------------USER NODE DATA-------------------------
            self.graph.nodes['user'].data['id']=cat_encoder(df['user_id'])
            self.graph.nodes['user'].data['country']=cat_encoder(df['country'])
            self.graph.nodes['user'].data['age']=th.tensor(df['age'], dtype=th.float)
            self.graph.nodes['user'].data['gender']=cat_encoder(df['gender'])
            self.graph.nodes['user'].data['playcount']=th.tensor(df['playcount'], dtype=th.float)
            del df

            if self.artists==True:
                # -------------------------USER->ARTISTS EDGE DATA-------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_LEs.txt'
                if self.playcount_weight:
                    playcounts, user_id_list, groupby_id_list=get_le_playcount(
                        file_path,type='artist',
                        user_mapping=mappings['user_mapping'], 
                        groupby_mapping=mappings['artist_mapping'],
                        relative_playcount=True
                        )
                    self.graph.edges['listened_to_artist'].data['weight']=id_encoder(playcounts)
                    self.graph.edges['artist_listened_by'].data['weight']=id_encoder(playcounts)
                    del playcounts
                else:
                    timestamps, user_id_list, groupby_id_list=get_les(
                    file_path,type='artist',
                    user_mapping=mappings['user_mapping'], 
                    groupby_mapping=mappings['artist_mapping']
                    )
                    self.graph.edges['listened_to_artist'].data['timestamp']=th.tensor((timestamps), dtype=th.float)
                    self.graph.edges['artist_listened_by'].data['timestamp']=th.tensor((timestamps), dtype=th.float)
                    del timestamps

                del mappings['artist_mapping']
                del user_id_list
                del groupby_id_list
            

            if self.albums==True:
                # -------------------------USER->ALBUMS EDGE DATA-------------------------    
                file_path=self.preprocessed_dir+'/'+'LFM-1b_LEs.txt'        
                if self.playcount_weight:        
                    playcounts, user_id_list, groupby_id_list=get_le_playcount(
                        file_path,type='album',
                        user_mapping=mappings['user_mapping'], 
                        groupby_mapping=mappings['album_mapping'],
                        relative_playcount=True
                        )
                    self.graph.edges['listened_to_album'].data['weight']=id_encoder(playcounts)
                    self.graph.edges['album_listened_by'].data['weight']=id_encoder(playcounts)
                    del playcounts
                else:
                    timestamps, user_id_list, groupby_id_list=get_les(
                    file_path,type='album',
                    user_mapping=mappings['user_mapping'], 
                    groupby_mapping=mappings['album_mapping']
                    )
                    self.graph.edges['listened_to_album'].data['timestamp']=id_encoder(timestamps)
                    self.graph.edges['album_listened_by'].data['timestamp']=id_encoder(timestamps)
                    del timestamps
                
                del mappings['album_mapping']
                del user_id_list
                del groupby_id_list
                
            
            if self.tracks==True:
                # -------------------------USER->TRACKS EDGE DATA-------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_LEs.txt'
                if self.playcount_weight:     
                    playcounts, user_id_list, groupby_id_list=get_le_playcount(
                        file_path,type='track',
                        user_mapping=mappings['user_mapping'], 
                        groupby_mapping=mappings['track_mapping'],
                        relative_playcount=True
                        )
                    self.graph.edges['listened_to_track'].data['weight']=id_encoder(playcounts)
                    self.graph.edges['track_listened_by'].data['weight']=id_encoder(playcounts)
                    del playcounts
                else:
                    timestamps, user_id_list, groupby_id_list=get_les(
                    file_path,type='track',
                    user_mapping=mappings['user_mapping'], 
                    groupby_mapping=mappings['track_mapping']
                    )
                    self.graph.edges['listened_to_track'].data['timestamp']=id_encoder(timestamps)
                    self.graph.edges['track_listened_by'].data['timestamp']=id_encoder(timestamps)
                    del timestamps

                del mappings['track_mapping']
                del user_id_list
                del groupby_id_list
                
            
                    
            del mappings


    def __getitem__(self, idx):
        glist,_=self.load()
        return glist[idx]

    def __len__(self):
        return 1
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='DGL_LFM1b', type=str, help='name of directory in data folder')
    parser.add_argument('--n_users', default=None, type=str, help="number of LE rows rto collect for a subset of the full dataset")
    parser.add_argument('--device', default='cpu', type=str, help='torch device to use for categorical encoding of ids')
    parser.add_argument('--overwrite_preprocessed', default=False, type=bool, help='boolean indication wheter to overwrite preprocessed ')
    parser.add_argument('--overwrite_processed', default=False, type=bool, help='boolean indication wheter to overwrite processed')
    parser.add_argument('--artists', default=True, type=bool, help='boolean indication wheter to use the artist and genre nodes in the graph')
    parser.add_argument('--albums', default=True, type=bool, help='boolean indication wheter to use the albums and genre nodes in the graph')
    parser.add_argument('--tracks', default=True, type=bool, help='boolean indication wheter to use the tracks and genre nodes in the graph')

    args = parser.parse_args()
    print('running with args...')
    print(args)
    
    LFM1b(name=args.name, n_users=args.n_users, device=args.device)