# PinSAGE
pinsage for wine recommendation
This is the PinSAGE package applied to the wine recommendation system prepared by the 11th Tobigs Conference "투믈리에".
It was implemented based on the DGL library and modified to fit the project in the PinSAGE example.

PinSAGE paper: https://arxiv.org/pdf/1806.01973.pdf <br>
DGL: https://docs.dgl.ai/# <br>
DGL PinSAGE example: https://github.com/dmlc/dgl/tree/master/examples/pytorch/pinsage <br>

## Paper Explanation
### Intro
We develop a dataefficient Graph Convolutional Network (GCN) algorithm PinSage, which combines efficient random walks and graph convolutions to generate embeddings of nodes (i.e. items) that incorporate both graph structure as well as node feature information.<br>
these learned representations have high utility because they can be re-used in various recommendation tasks. For example, item embeddings learned using a deep model can be used for item-item recommendation.<br>
	
### Method
To generate the embedding for a node (i.e., an item), we apply multiple convolutional modules that aggregate feature information (e.g., visual, textual features) from the node’s local graph neighborhood (Figure 1). Importantly, parameters of these localized convolutional modules are shared across all nodes, making the parameter complexity of our approach independent of the input graph size.<br>
<br>
<b>a. Setup</b><br>
Pinterest is a content discovery application where users interact with pins, which are visual bookmarks to online content (e.g., recipes they want to cook, or clothes they want to purchase). Users organize these pins into boards, which contain collections of pins that the user deems to be thematically related. Our task is to generate high-quality embeddings or representations of pins that can be used for recommendation. In order to learn these embeddings, we model the Pinterest environment as a bipartite graph consisting of nodes in two disjoint sets, I (containing pins) and C (containing boards). <br>
<br>
In addition to the graph structure, we also assume that the pins/items u ∈ I are associated with real-valued attributes, xu ∈ R d . In general, these attributes may specify metadata or content information about an item, and in the case of Pinterest, we have that pins are associated with both rich text and image features. Our goal is to leverage both these input attributes as well as the structure of the bipartite graph to generate high-quality embeddings. These embeddings are then used for recommender system candidate generation via nearest neighbor lookup (i.e., given a pin, find related pins). <br>
<br>
For notational convenience and generality, when we describe the PinSage algorithm, we simply refer to the node set of the full graph with V = I ∪ C and do not explicitly distinguish between pin and board nodes.<br>
<br>
<b>b. Model Architecture</b><br>
We use localized convolutional modules to generate embeddings for nodes. We start with input node features and then learn neural networks that transform and aggregate features over the graph to compute the node embeddings (Figure 1).<br>
Algorithm 1: <br>
convolve Input :<br>
Current embedding zu for node u; <br>
set of neighbor embeddings {zv |v ∈ N (u)}, <br>
set of neighbor weights α; <br>
symmetric vector function γ (·) <br>
<br>
Output :New embedding z_new_u for node u <br>
1) nu ← γ ({ReLU (Qhv + q) | v ∈ N (u)} ,α); <br>
2) z_new_u ← ReLU (W · concat(zu, nu ) + w); <br>
3) z_new_u ←  z_new_u /∥ z_new_u ∥2<br>
<br>
The basic idea is that we transform the representations zv , ∀v ∈ N (u) of u’s neighbors through a dense neural network and then apply a aggregator/pooling fuction (e.g., a element-wise mean or weighted sum, denoted as γ ) on the resulting set of vectors (Line 1). This aggregation step provides a vector representation, nu , of u’s local neighborhood, N (u). We then concatenate the aggregated neighborhood vector nu with u’s current representation hu and transform the concatenated vector through another dense neural network layer (Line 2). Empirically we observe significant performance gains when using concatenation operation instead of the average operation as in [21]. Additionally, the normalization in Line 3 makes training more stable. The output of the algorithm is a representation of u that incorporates both information about itself and its local graph neighborhood.<br>
<br>
Importance-based neighborhoods. An important innovation in our approach is how we define node neighborhoods N (u), i.e., how we select the set of neighbors to convolve over in Algorithm 1. In PinSage we define importance-based neighborhoods, where the neighborhood of a node u is defined as the T nodes that exert the most influence on node u. Concretely, we simulate random walks starting from node u and compute the L1-normalized visit count of nodes visited by the random walk [14].2 The neighborhood of u is then defined as the top T nodes with the highest normalized visit counts with respect to node u. We implement γ in Algorithm 1 as a weighted-mean, with weights defined according to the L1 normalized visit counts. We refer to this new approach as importance pooling.<br>
<br>
<b>c. Model Training</b><br>
We train PinSage in a supervised fashion using a max-margin ranking loss (Max margin loss, also known as hinge loss, is a loss function used in machine learning for binary classification problems. The goal of this loss function is to maximize the margin between positive and negative samples in the training data). In this setup, we assume that we have access to a set of labeled pairs of items L, where the pairs in the set, (q,i) ∈ L, are assumed to be related—i.e., we assume that if (q,i) ∈ L then item i is a good recommendation candidate for query item q. The goal of the training phase is to optimize the PinSage parameters so that the output embeddings of pairs (q,i) ∈ L in the labeled set are close together.<br>
Loss function. In order to train the parameters of the model, we use a max-margin-based loss function. The basic idea is that we want to maximize the inner product of positive examples, i.e., the embedding of the query item and the corresponding related item. At the same time we want to ensure that the inner product of negative examples—i.e., the inner product between the embedding of the query item and an unrelated item—is smaller than that of the positive sample by some pre-defined margin. The loss function for a single pair of node embeddings (zq, zi) : (q,i) ∈ L is thus 
JG(zqzi) = Enk∼Pn(q) max{0, zq · znk − zq · zi + ∆}, (1)<br>
where Pn(q) denotes the distribution of negative examples for item q, and ∆ denotes the margin hyper-parameter. Here negative samples are taken as easy and hard. Easy samples are the one that are unrelated to the given item while hard negative samples are the one that are very close to given item but still unrelated (these are examples that have page rank 2000-5000 corresponding to given item)<br>
<br>
<b>d. Node Embeddings via MapReduce</b><br>
We observe that inference of node embeddings very nicely lends itself to MapReduce computational model. Figure 3 details the data flow on the bipartite pin-to-board Pinterest graph, where we assume the input (i.e., “layer-0”) nodes are pins/items (and the layer-1 nodes are boards/contexts).<br>
<br>
<b>e. Efficient nearest-neighbor lookups</b><br>
given a query item q, the we can recommend items whose embeddings are the K-nearest neighbors of the query item’s embedding. Approximate KNN can be obtained efficiently via locality sensitive hashing <br>
<br>
Features used for learning. Each pin at Pinterest is associated with an image and a set of textual annotations (title, description). To generate feature representation xq for each pin q, we concatenate visual embeddings (4,096 dimensions), textual annotation embeddings (256 dimensions), and the log degree of the node/pin in the graph. The visual embeddings are the 6-th fully connected layer of a classification network using the VGG-16 architecture [28]. Textual annotation embeddings are trained using a Word2Vec-based model [23], where the context of an annotation consists of other annotations that are associated with each pin.<br>
<br>


## Dataset

### Vivino
11,900,000 Wines & 42,000,000 Users
User feature: userID, user_follower_count, user_rating_count
Item feature: wine_id, body, acidity, alcohol, rating_average, grapes_id


We have a request to share data, so we provide it for you to use in part.
* 100,000 review data
* User Metadata
* Wine Metadata

As much as it's not the entire data, when you learn it yourself, performance may not come out as much as you want.
**process_wine.py** is code that preprocesses collected data for DGL If you use the data provided, please refer to it.


## Training model

### Nearest-neighbor recommendation

This model recommends wine as Knearst Neighbors for all users.
This method finds the center of the embedding vector of the wine consumed by a specific user and recommends the K wines closest to the center vector.<br>
I have also added another method that utilizes cosine distance for each pin attached to board to find its relevant pins in the rest of the data and that appears as another metrics in evaluation

```
python model.py -d data.pkl -s model -k 500 --eval-epochs 100 --save-epochs 100 --num-epochs 500 --device 0 --hidden-dims 128 --batch-size 64 --batches-per-epoch 512
```

- d: Data Files
- s: The name of the model to
- k: top K count
- eval epochs: performance output epoch interval (0 = output X)
- save epochs: storage epoch interval (0 = storage X)
- - num epochs: epoch 횟수
- hidden dims: embedding dimension
- batch size: batch size
- - batches per epoch: iteration 횟수

In addition, there are parameters applied by PinSAGE, so please refer to the model.py code.

## Inference
Use method called get_recommendations in model.py<br>
<br> 
There are two methods that I have applied here-<br>
1. We average the node embeddings of a particular user for inference to obtain a central embedding vector, and obtain Distance with all embeddings and matrix operations. We extract as many as K embeddings in the order of small distances and present them as final recommendations. 
2. Take cosine ditance of each node embedding of a particular user for inference and obtain distance with all the embeddings. For this as well extract as many as K embeddings in the order of small distance and present them as final recommendations.<br>
<br>
Evaluate with Recall and Hitrate whether the selected items belong to the verification data.<br>


## Performance

Model | Hitrate | Recall
------------ | ------------- | -------------
SVD | 0.854 | 0.476
PinSAGE | 0.942 | 0.693
