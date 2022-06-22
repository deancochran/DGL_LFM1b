# lfm1b-dgl-hetero
This repository is a custom DGL datatset created from the LFM-1b database. 
The database downloads and processes the full database to create one singular DGL heterogeneous graph.

The node types of the graph:
- User (120K)
- Artsit (3M)
- Album (15M)
- Track (32M)
- Genre (20)

The Edge types of the graph :
- User -> Artsit (61411336)
- Artsit -> User (61411336)
- User -> Album (na)
- Album -> User (na)
- User -> Track (na)
- Track -> User (na)
- Artsit -> Genre (414379)
- Genre -> Artsit (414379)
- Album -> Artsit (14184326)
- Artsit -> Album (14184326)
- Track -> Artsit (27258365)
- Artsit -> Track (27258365)


Additionally, for all the user edges:

- User -> Artsit
- Artsit -> User
- User -> Album 
- Album -> User 
- User -> Track 
- Track -> User

There is 'norm_weight' edge data indicating the normalized realtive interaction a user had with a specified target artist, album, track. 
The 'norm_weight' edge data for all other edges is represented as a 1 

# Requirements

This repository was built with Python 3.8.10 to install the requirements.txt file, ensure you have the correct lilbraries pre-installed:

- [torch](https://pytorch.org/) 1.11.0
- [dgl](https://www.dgl.ai/) 0.8.2

Follow these manual imports with:

    pip install -r requirements.txt



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

# Compile the dataset

To compile the LFM1b database, simply 'cd' into the root of the repository and run:

    python LFM1b.py

### **Precurser warning**: 

I, the author of the repository, am using a Linux Machine with 30GB of RAM and 12GB of GPU.  To run the above script, it will take the machine ~2hrs, and I am unable to store the full knowledge graph in memory


## Compile a subset

To create a subset of the LFM1b database, simply 'cd' into the root of the repository and run:

    python LFM1b.py --n_users 50

`This provides a subset of 50 users with their correspoing listen events, and the artists/albums/tracks associated with their listening habits`


# The DGL Framework

The Deep Graph library ([DGL]((https://www.dgl.ai/))) framework provides the ability to utilize the DGLDataset object
to generate a customizeable dataset for the purpose of node/link/graph down stream tasks.

Once the dataset is compiled you may import the class into any file and load the precompiled graph for DGL based analysis.

    from LFM1b import LFM1b 
    dataset = LFM1b()
    glist, glabels = dataset.load()
    hg=glist[0]

    print(hg)





