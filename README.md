Python 3.8.10
# lfm1b-dgl-hetero
This repository is a custom DGL datatset created from the LFM-1b database. 
The database downloads and processes the full database to create one singular DGL heterogeneous graph.

The node types of the graph:
- User (120K)
- Artsit (3M)
- Album (15M)
- Track (32M)
- Genre (20)



# The Data

The LFM-1b dataset collection more than one billion listening events, intended to be used for various music retrieval and recommendation tasks. 
The [paper](http://www.cp.jku.at/people/schedl/Research/Publications/pdf/schedl_icmr_2016.pdf) written by Schedl, M. was published in 2016 
for ICMR and is directly available through the [website](http://www.cp.jku.at/datasets/LFM-1b/). 

In case you make use of the LFM-1b dataset in your own research, please cite the following paper:


    The LFM-1b Dataset for Music Retrieval and Recommendation
    Schedl, M.
    Proceedings of the ACM International Conference on Multimedia Retrieval (ICMR 2016), New York, USA, April 2016.

Additionally, the [paper](http://www.cp.jku.at/people/schedl/Research/Publications/pdf/schedl_ism_mam_2017.pdf) written by Schedl, M. and Ferwerda, B. discussing
the LFM1b User Genre Profile dataset was published in 2017 for ISM. It uses Last.fm artist tags indexed with two dictionaries of genre and style descriptors 
(from Allmusic and Freebase) to create, for each user in LFM-1b, a preference profile as a vector over genres.


In case you make use of the LFM-1b UGP dataset in your own research, please cite the following paper:


    Large-scale Analysis of Group-specific Music Genre Taste From Collaborative Tags
    Schedl, M. and Ferwerda, B.
    Proceedings of the 19th IEEE International Symposium on Multimedia (ISM 2017), Taichung, Taiwan, December 2017.