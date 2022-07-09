def get_metapath(num_nodes_dict):
    if ('artist' in num_nodes_dict.keys()) and ('album' not in num_nodes_dict.keys()) and ('track' not in num_nodes_dict.keys()):
        return [
        ('user','listened_to_artist','artist'),
        ('artist','in_genre','genre'),
        ('genre','is_genre_of','artist'),
        ('artist','artist_listened_by','user'),
        ]
    elif ('artist' in num_nodes_dict.keys()) and ('album' not in num_nodes_dict.keys()) and ('track' in num_nodes_dict.keys()):
        return [
        # ('user','listened_to_track','track'),
        # ('track','preformed_by','artist'),
        # ('artist','artist_listened_by','user'),
        # ('user','listened_to_artist','artist'),
        # ('artist','in_genre','genre'),
        # ('genre','is_genre_of','artist'),
        # ('artist','preformed','track'),
        # ('track','track_listened_by','user'),
        ('user','listened_to_track','track'),
        ('track','track_listened_by','user'), 
        ('user','listened_to_artist','artist'),
        ('artist','in_genre','genre'),
        ('genre','is_genre_of','artist'),
        ('artist','artist_listened_by','user'),
        ]
    elif ('artist' in num_nodes_dict.keys()) and ('album' in num_nodes_dict.keys()) and ('track' not in num_nodes_dict.keys()):
        return [
        # ('user','listened_to_album','album'),
        # ('album','produced_by','artist'),
        # ('artist','artist_listened_by','user'),
        # ('user','listened_to_artist','artist'),
        # ('artist','in_genre','genre'),
        # ('genre','is_genre_of','artist'),
        # ('artist','produced','album'),
        # ('album','album_listened_by','user'),
        ('user','listened_to_album','album'),
        ('album','album_listened_by','user'),
        ('user','listened_to_artist','artist'),
        ('artist','in_genre','genre'),
        ('genre','is_genre_of','artist'),
        ('artist','artist_listened_by','user'),
        ]

    elif ('artist' not in num_nodes_dict.keys()) and ('album' in num_nodes_dict.keys()) and ('track' not in num_nodes_dict.keys()):
        return [
        ('user','listened_to_album','album'),
        ('album','album_listened_by','user'),
        ]
    elif ('artist' not in num_nodes_dict.keys()) and ('album' in num_nodes_dict.keys()) and ('track' in num_nodes_dict.keys()):
        return [
        ('user','listened_to_album','album'),
        ('album','album_listened_by','user'),
        ('user','listened_to_track','track'),
        ('track','track_listened_by','user'),
        ]

    elif ('artist' not in num_nodes_dict.keys()) and ('album' not in num_nodes_dict.keys()) and ('track' in num_nodes_dict.keys()):
        return [
        ('user','listened_to_track','track'),
        ('track','track_listened_by','user'),
        ]


    elif ('artist' in num_nodes_dict.keys()) and ('album' in num_nodes_dict.keys()) and ('track' in num_nodes_dict.keys()):
        return [
        # ('user','listened_to_track','track'),
        # ('track','preformed_by','artist'),

        # ('artist','produced','album'),
        # ('album','album_listened_by','user'),

        # ('user','listened_to_artist','artist'),
        # ('artist','in_genre','genre'),

        # ('genre','is_genre_of','artist'),
        # ('artist','artist_listened_by','user'),

        # ('user','listened_to_album','album'),
        # ('album','produced_by','artist'),

        # ('artist','preformed','track'),
        # ('track','track_listened_by','user'),
        ('user','listened_to_track','track'),
        ('track','track_listened_by','user'),
        
        ('user','listened_to_album','album'),
        ('album','album_listened_by','user'),

        ('user','listened_to_artist','artist'),
        ('artist','in_genre','genre'),
        ('genre','is_genre_of','artist'),
        ('artist','artist_listened_by','user'),
        
        
        ]
    else:
        raise Exception(f'no metapath described for graph with {num_nodes_dict.keys()}')