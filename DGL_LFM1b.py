
from enum import unique
import warnings
import os
import argparse
from dgl import heterograph
from dgl.data import DGLDataset, download, extract_archive
from dgl.data.utils import save_graphs, load_graphs
from torch_sparse import transpose
from torch_geometric.nn import MetaPath2Vec
from .data_utils import *
from .meta_paths import *
warnings.filterwarnings("ignore", category=FutureWarning)
th.cuda.empty_cache()

class LFM1b(DGLDataset):
    def __init__(self, 
    n_users=None, 
    popular_artists=False,
    device='cpu', 
    overwrite_preprocessed=False, 
    overwrite_processed=False,
    artists=True,
    albums=True,
    tracks=True,
    playcount_weight=False,
    norm_playcount_weight=True,
    metapath2vec=True,
    emb_dim=32,
    walk_length=64,
    context_size=7,
    walks_per_node=3,
    num_negative_samples=10,
    batch_size=512,
    learning_rate=0.001,
    epochs=5,
    logs=100
    ):
        name='DGL_LFM1b'
        self.root_dir = './data/'+name
        self.preprocessed_dir = './data/'+name+'/preprocessed'
        self.raw_ugp_dir='./data/'+name+f'/{name}_UGP'
        self.lfm1b_ugp_url='http://www.cp.jku.at/datasets/LFM-1b/LFM-1b_UGP.zip'
        self.n_users=n_users
        self.popular_artists=popular_artists
        self.device=device
        self.overwrite_preprocessed=overwrite_preprocessed
        self.overwrite_processed=overwrite_processed
        self.artists=artists
        self.albums=albums
        self.tracks=tracks
        self.emb_dim=emb_dim
        self.playcount_weight=playcount_weight
        self.norm_playcount_weight=norm_playcount_weight
        self.walk_length=walk_length
        self.context_size=context_size
        self.walks_per_node=walks_per_node
        self.num_negative_samples=num_negative_samples
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.logs=logs
        self.metapath2vec=metapath2vec
        super().__init__(
            name=name, 
            url='http://drive.jku.at/ssf/s/readFile/share/1056/266403063659030189/publicLink/LFM-1b.zip', 
            raw_dir=self.root_dir+'/'+name, 
            save_dir=self.root_dir+'/processed',
            hash_key=(), 
            force_reload=False, 
            verbose=False
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
        print('\n','Processing LFM1b')
        if os.path.exists(self.save_dir+'/lastfm1b.bin') == True and (self.overwrite_processed == True or self.overwrite_processed== True):
            os.remove(self.save_dir+'/lastfm1b.bin')
        processed_condition = os.path.exists(os.path.join(self.save_dir+'/lastfm1b.bin')) == False
        if processed_condition == True:
            preprocessed_files_dont_exist = os.path.exists(os.path.join(self.preprocessed_dir+'/LFM-1b_LEs.txt')) == False or os.path.exists(os.path.join(self.preprocessed_dir+'/LFM-1b_albums.txt')) == False or os.path.exists(os.path.join(self.preprocessed_dir+'/LFM-1b_artists.txt')) == False or os.path.exists(os.path.join(self.preprocessed_dir+'/LFM-1b_tracks.txt')) == False or os.path.exists(os.path.join(self.preprocessed_dir+'/LFM-1b_users.txt')) == False
            if preprocessed_files_dont_exist == True or self.overwrite_preprocessed == True:
                preprocess_raw(self.raw_dir,self.preprocessed_dir, n_users=self.n_users, popular_artists=self.popular_artists)

            graph_data = {}
            num_nodes_dict = {}
            mappings={} 
            # device = th.device(self.device)
            id_encoder = IdentityEncoder(device='cpu') # used to encode floats f(x)==2, where x = 2 
            # -------------------------USER MAPPING-------------------------
            file_path=self.preprocessed_dir+'/'+'LFM-1b_users.txt'
            print('\t','Loading Mapping Data from',file_path.split('_')[-1])
            df = get_preprocessed_ids(file_path, type='user', id_list=['user_id'])
            mappings['user_mapping']= {int(id): i for i, id in enumerate(df['user_id'])}
            num_nodes_dict['user']=len(mappings['user_mapping'].values())
            del df

            if self.artists==True:
                # -------------------------GENRE MAPPING-------------------------
                file_path=self.preprocessed_dir+'/'+'genres_allmusic.txt'
                print('\t','Loading Mapping Data from',file_path.split('_')[-1])
                df = get_preprocessed_ids(file_path,type='genre' ,id_list=['genre_id','genre_name'])
                num_nodes_dict['genre']=len(df['genre_id'])
                del df

                # -------------------------ARTIST MAPPING -------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_artists.txt'
                print('\t','Loading Mapping Data from',file_path.split('_')[-1])
                df = get_preprocessed_ids(file_path, type='artist', id_list=['artist_id','artist_name'])
                mappings['artist_name_mapping'] = {artist_name: int(artist_id)  for artist_id, artist_name in zip(df['artist_id'],df['artist_name'])}
                mappings['artist_mapping'] = {int(id): i for i, id in enumerate(df['artist_id'])}
                df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
                num_nodes_dict['artist']=len(mappings['artist_mapping'].values())
                del df


            if self.albums==True:
                # -------------------------ALBUM MAPPING -------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_albums.txt'
                print('\t','Loading Mapping Data from',file_path.split('_')[-1])
                df = get_preprocessed_ids(file_path, type='album', id_list=['album_id','artist_id'])
                if self.artists==True:
                    df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
                mappings['album_mapping'] = {int(id): i for i, id in enumerate(df['album_id'])}
                num_nodes_dict['album']=len(mappings['album_mapping'].values())
                del df


            if self.tracks==True:
                # -------------------------TRACK MAPPING -------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_tracks.txt'
                print('\t','Loading Mapping Data from',file_path.split('_')[-1])
                df = get_preprocessed_ids(file_path, type='track', id_list=['track_id', 'artist_id'])
                if self.artists==True:
                    df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
                mappings['track_mapping'] = {int(id): i for i, id in enumerate(df['track_id'])}
                num_nodes_dict['track']=len(mappings['track_mapping'].values())
                del df

            if self.artists==True:
                # -------------------------GENRES->ARTISTS GRAPH DATA-------------------------
                file_path=self.raw_ugp_dir+'/LFM-1b_artist_genres_allmusic.txt'
                print('\t','Loading Graph Data from',file_path.split('_')[-1])
                df = get_artist_genre_df(file_path, mappings['artist_name_mapping'], mappings['artist_mapping'], self.preprocessed_dir)
                graph_data[('artist', 'in_genre', 'genre')]=(th.tensor(df['artist_id']), th.tensor(df['genre_id']))
                graph_data[('genre', 'is_genre_of', 'artist')]=(th.tensor(df['genre_id']), th.tensor(df['artist_id']))
                del df

                if self.albums==True:
                    # -------------------------ALBUMS->ARTISTS GRAPH DATA-------------------------
                    file_path=self.preprocessed_dir+'/'+'LFM-1b_albums.txt'
                    print('\t','Loading Graph Data from',file_path.split('_')[-1])
                    df = get_preprocessed_ids(file_path, type='album', id_list=['album_id','artist_id'])
                    df=remap_ids(df, ordered_cols=['album_id'], mappings=[mappings['album_mapping']])
                    if self.artists==True:
                        df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
                    print('number of artist album edges:',len(df['album_id']))
                    graph_data[('artist', 'produced', 'album')]=(th.tensor(df['artist_id']), th.tensor(df['album_id']))
                    graph_data[('album', 'produced_by', 'artist')]=(th.tensor(df['album_id']), th.tensor(df['artist_id']))
                    del df

                if self.tracks==True:
                    # -------------------------TRACKS->ARTISTS GRAPH DATA-------------------------
                    file_path=self.preprocessed_dir+'/'+'LFM-1b_tracks.txt'
                    print('\t','Loading Graph Data from',file_path.split('_')[-1])
                    df = get_preprocessed_ids(file_path, type='track', id_list=['track_id', 'artist_id'])
                    df=remap_ids(df, ordered_cols=['track_id'], mappings=[mappings['track_mapping']])
                    if self.artists==True:
                        df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
                    print('number of artist track edges:',len(df['track_id']))
                    graph_data[('track', 'preformed_by', 'artist')]=(th.tensor(df['track_id']), th.tensor(df['artist_id']))
                    graph_data[('artist', 'preformed', 'track')]=(th.tensor(df['artist_id']), th.tensor(df['track_id']))
                    del df

                # -------------------------USER->ARTISTS GRAPH DATA-------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_LEs.txt'
                # print('\t','Loading Graph Data from',file_path.split('_')[-1])
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
                
                print('number of user artist edges:',len(groupby_id_list))
                del user_id_list
                del groupby_id_list
                

            if self.albums==True:
                # -------------------------USER->ALBUMS GRAPH DATA-------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_LEs.txt'
                print('\t','Loading Graph Data from',file_path.split('_')[-1])
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
                print('number of user album edges:',len(groupby_id_list))
                del user_id_list
                del groupby_id_list
                
            if self.tracks==True:
                # -------------------------USER->TRACKS GRAPH DATA-------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_LEs.txt'
                print('\t','Loading Graph Data from',file_path.split('_')[-1])
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
                print('number of user track edges:',len(groupby_id_list))
                del user_id_list
                del groupby_id_list
                

            # -------------------------DGL HETERO GRAPH OBJECT-------------------------
            print('\t','Creating DGL HeteroGraph from Graph Data')
            self.graph = heterograph(graph_data,num_nodes_dict)
            print(self.graph)
            del graph_data


            # ------------------------- METAPATH2VEC NODE EMBEDDING ENCODER -------------------------
            if self.metapath2vec==True:
                print('\t','Creating metapath2vec node embeddings')
                metapath=get_metapath(num_nodes_dict)
                print('using metapath',metapath)
                metapath2vec_model = MetaPath2Vec(
                    {(s,e,d):th.stack(self.graph[e].adj_sparse('coo')) for s,e,d in self.graph.canonical_etypes}, 
                    embedding_dim=self.emb_dim,
                    metapath=metapath, 
                    walk_length=self.walk_length,
                    context_size=self.context_size,
                    walks_per_node=self.walks_per_node,
                    num_negative_samples=self.num_negative_samples).to(self.device)

                print('training...')
                loader = metapath2vec_model.loader(batch_size=self.batch_size, shuffle=True, num_workers=4)
                optimizer = th.optim.Adam(metapath2vec_model.parameters(), lr=self.learning_rate)
                metapath2vec_model.train()
                for epoch in range(1, self.epochs + 1):
                    for i, (pos_rw, neg_rw) in enumerate(loader):
                        optimizer.zero_grad()
                        loss = metapath2vec_model.loss(pos_rw.to(self.device), neg_rw.to(self.device))
                        loss.backward()
                        optimizer.step()
                        print('\r',f'Epoch: {epoch:02d} of {self.epochs+1}, Step: {i + 1:03d}/{len(loader)}, 'f'Loss: {loss:.4f}', end=' ')
                del loader, optimizer
                print('loading...')
                embedding_dict = {}
                for node_type in metapath2vec_model.num_nodes_dict:
                    # get embedding of node with specific type
                    embedding_dict[node_type] = metapath2vec_model(node_type).detach().cpu()
                del metapath2vec_model
                nodes_embedding_path = self.preprocessed_dir+'/LFM-1b_nodes_embedding.pt'
                th.save(embedding_dict, nodes_embedding_path)
                print('saved! embedding_dict')             
            if self.artists==True:
                # -------------------------ARTIST NODE DATA-------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_artists.txt'
                print('\t','Loading features from',file_path.split('_')[-1])
                if self.metapath2vec==True:
                    self.graph.nodes['artist'].data['feat']=embedding_dict['artist']
                else:
                    df = get_preprocessed_ids(file_path, type='artist', id_list=['artist_id','artist_name'])
                    df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
                    print("artist_size",len(df['artist_id']))
                    self.graph.nodes['artist'].data['feat']=id_encoder(df['artist_id'])
                    del df

                # -------------------------GENRE NODE DATA-------------------------
                file_path=self.preprocessed_dir+'/'+'genres_allmusic.txt'
                print('\t','Loading features from',file_path.split('_')[-1])
                if self.metapath2vec==True:
                    self.graph.nodes['genre'].data['feat']=embedding_dict['genre']
                else:
                    df = get_preprocessed_ids(file_path,type='genre' ,id_list=['genre_id','genre_name'])
                    self.graph.nodes['genre'].data['feat']=id_encoder(df['genre_id'])
                    del df
                    del mappings['artist_name_mapping']
                
            if self.albums==True:
                # -------------------------ALBUM NODE DATA-------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_albums.txt'
                print('\t','Loading features from',file_path.split('_')[-1])
                if self.metapath2vec==True:
                    self.graph.nodes['album'].data['feat']=embedding_dict['album']
                else:
                    df = get_preprocessed_ids(file_path, type='album', id_list=['album_id','album_name','artist_id'])
                    if self.artists==True:
                        df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
                    df=remap_ids(df, ordered_cols=['album_id'], mappings=[mappings['album_mapping']])
                    self.graph.nodes['album'].data['feat']=id_encoder(df['album_id'])
                    del df

            if self.tracks==True:
                # -------------------------TRACK NODE DATA-------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_tracks.txt'
                print('\t','Loading features from',file_path.split('_')[-1])
                if self.metapath2vec==True:
                    self.graph.nodes['track'].data['feat']=embedding_dict['track']
                else:
                    df = get_preprocessed_ids(file_path, type='track', id_list=['track_id', 'track_name', 'artist_id'])
                    if self.artists==True:
                        df=remap_ids(df, ordered_cols=['artist_id'], mappings=[mappings['artist_mapping']])
                    df=remap_ids(df, ordered_cols=['track_id'], mappings=[mappings['track_mapping']])     
                    self.graph.nodes['track'].data['feat']=(df['track_id'])
                    del df

            # -------------------------USER NODE DATA-------------------------
            file_path=self.preprocessed_dir+'/'+'LFM-1b_users.txt'
            print('\t','Loading features from',file_path.split('_')[-1])
            if self.metapath2vec==True:
                self.graph.nodes['user'].data['feat']=embedding_dict['user']
            else:
                df = get_preprocessed_ids(file_path, type='user', id_list=['user_id','country','age','gender','playcount'])
                df=remap_ids(df, ordered_cols=['user_id'], mappings=[mappings['user_mapping']])
                self.graph.nodes['user'].data['feat']=id_encoder(df['user_id'])
                del df

            if self.artists==True:
                # -------------------------USER->ARTISTS EDGE DATA-------------------------
                file_path=self.preprocessed_dir+'/'+'LFM-1b_LEs.txt'
                if self.playcount_weight:
                    playcounts, user_id_list, groupby_id_list=get_le_playcount(
                        file_path,type='artist',
                        user_mapping=mappings['user_mapping'], 
                        groupby_mapping=mappings['artist_mapping'],
                        relative_playcount=False
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
                    self.graph.edges['listened_to_artist'].data['timestamp']=id_encoder(timestamps)
                    self.graph.edges['artist_listened_by'].data['timestamp']=id_encoder(timestamps)
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
                        relative_playcount=self.norm_playcount_weight
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
                        relative_playcount=False
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

  

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_users', default=None, type=str, help="number of LE rows rto collect for a subset of the full dataset")
    parser.add_argument('--popular_artists', default=False, type=str2bool, nargs='?', const=True, help="number of LE rows rto collect for a subset of the full dataset")
    parser.add_argument('--device', default='cpu', type=str, help='GPU or CPU device specification')
    parser.add_argument('--overwrite_preprocessed', default=False, type=str2bool, nargs='?', const=True, help='indication to overwrite preprocessed ')
    parser.add_argument('--overwrite_processed', default=False, type=str2bool, nargs='?', const=True, help='indication to overwrite processed')
    parser.add_argument('--artists', default=True, type=str2bool, nargs='?', const=True, help='indication to use the artist and genre nodes in the graph')
    parser.add_argument('--albums', default=True, type=str2bool, nargs='?', const=True, help='indication to use the albums and genre nodes in the graph')
    parser.add_argument('--tracks', default=True, type=str2bool, nargs='?', const=True, help='indication to use the tracks and genre nodes in the graph')
    parser.add_argument('--playcount_weight', default=False, type=str2bool, nargs='?', const=True, help='indication to use the a single edge with weight feature, or every edge with timestamp features between a user and their unique listen events')
    parser.add_argument('--norm_playcount_weight', default=True, type=str2bool, nargs='?', const=True, help='indication give every edge a "normalized playcount weight" feature, or "total playcount weight"')
    parser.add_argument('--metapath2vec', default=True, type=str2bool, nargs='?', const=True, help='indication to use metapath2vec to encode node embeddings (recommended, otherwise manual adjustment may be required)')
    parser.add_argument('--emb_dim', default=6, type=int,  help='node embedding vector size')
    parser.add_argument('--walk_length', default=32, type=int,  help='length of metapath2vec walks')
    parser.add_argument('--context_size', default=4, type=int,  help='context_size of metapath2vec')
    parser.add_argument('--walks_per_node', default=3, type=int,  help='context_size of metapath2vec')
    parser.add_argument('--num_negative_samples', default=5, type=int,  help='num_negative_samples of metapath2vec')
    parser.add_argument('--metapath2vec_epochs_batch_size', default=56, type=int,  help='batch_size of metapath2vec')
    parser.add_argument('--learning_rate', default=0.01, type=float,  help='learning_rate of metapath2vec')
    parser.add_argument('--metapath2vec_epochs', default=5, type=int,  help='epochs of metapath2vec')
    parser.add_argument('--logs', default=100, type=int,  help='logs of metapath2vec')

    args = parser.parse_args()
    print('\n','running with args...')
    print(args)

    LFM1b(
        n_users=args.n_users, 
        popular_artists=args.popular_artists,
        device=args.device, 
        overwrite_preprocessed=args.overwrite_preprocessed,
        overwrite_processed=args.overwrite_processed,
        artists=args.artists,
        albums=args.albums,
        tracks=args.tracks,
        playcount_weight=args.playcount_weight,
        norm_playcount_weight=args.norm_playcount_weight,
        metapath2vec=args.metapath2vec,
        emb_dim=args.emb_dim, 
        walk_length=args.walk_length,
        context_size=args.context_size,
        walks_per_node=args.walks_per_node,
        num_negative_samples=args.num_negative_samples,
        batch_size=args.metapath2vec_epochs_batch_size,
        learning_rate=args.learning_rate,
        epochs=args.metapath2vec_epochs,
        logs=args.logs
        )



