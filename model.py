# %%
import pickle

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torchtext
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.auto import tqdm as tq_

import layers
import sampler as sampler_module
import evaluation
from argparse import Namespace
from sklearn.metrics.pairwise  import cosine_distances


def get_recommendations(data_dict, args, model_path='epoch.pt'):

    data_dict = prepare_dataset(data_dict, args)
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('Current using CPUs')
    else:
        print ('Current cuda device ', torch.cuda.current_device()) # check

    gnn = PinSAGEModel(data_dict['graph'], data_dict['item_ntype'], data_dict['textset'], args.hidden_dims, args.num_layers).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=args.lr)
    checkpoint = torch.load(model_path, map_location=device)
    gnn.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])

    g = data_dict['graph']
    item_ntype = data_dict['item_ntype']
    user_ntype = data_dict['user_ntype']
    user_to_item_etype = data_dict['user_to_item_etype']
    timestamp = data_dict['timestamp']
    nid_uid_dict = {v: k for v, k in enumerate(list(g.ndata['userID'].values())[0].numpy())}
    nid_wid_dict = {nid.item(): wid.item() for wid, nid in  zip(g.ndata['wine_id']['wine'], g.ndata['id']['wine'])}


    gnn = gnn.to(device)

    neighbor_sampler = sampler_module.NeighborSampler(
        g, user_ntype, item_ntype, args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)
    
    # get embedding of all items (wine)
    h_item = evaluation.get_all_emb(gnn, g.ndata['id'][item_ntype], data_dict['textset'], item_ntype, neighbor_sampler, args.batch_size, device)
    # get interacted node ids for all users
    item_batch = evaluation.item_by_user_batch(g, user_ntype, item_ntype, user_to_item_etype, timestamp, args)
    users = []

    for i, nodes in tq_(enumerate(item_batch)):
        '''
        nodes : Actual interaction nodes per user [train node, test node (8: 2 ratio)]
        '''
        # get Real user ID from given ID
        category = nid_uid_dict[i]
        user_id = data_dict['user_category'][category]  # real user id
        label = data_dict['testset'][user_id]  # test label
        users.append(user_id)

        # Explore Real Wine IDs
        item = evaluation.node_to_item(nodes, nid_wid_dict, data_dict['item_category'])  # wine ID

        # get interacted wine IDs that are available in testset
        label_idx = [i for i, x in enumerate(item) if x in label]  # label index

        # Item Recommendation
        # get nodes that are not in test set i.e. they are in training set
        test_nodes = [x for i, x in enumerate(nodes) if i in label_idx]  # label index Nodes for training without input
        train_nodes = [x for i, x in enumerate(nodes) if i not in label_idx]  # label index Nodes for training without input
        if len(train_nodes)>0 and len(test_nodes)>0:
            # get embedding of above nodes i.e. embedding of training data
            h_nodes_train = h_item[train_nodes]
            # get center embedding of training data
            h_center = torch.mean(h_nodes_train, axis=0) 

            # method 1: distance between above embedding with rest of the embeddings 
            dist = h_center @ h_item.t()  # matrix multiplication
            # dist Extract k items in order of size
            topk = dist.topk(300)[1].cpu().numpy()  
            topk = evaluation.node_to_item(topk, nid_wid_dict, data_dict['item_category'])  # ID conversion
            tp = [x for x in label if x in topk]
            print('method 1', tp)

            # method 2: cosine distance
            cos_dist = cosine_distances(h_nodes_train, h_item)
            top_distk = np.argsort(cos_dist)[:, :300].flatten()#[:args.k]
            top_distk = evaluation.node_to_item(top_distk, nid_wid_dict, data_dict['item_category'])
            tp = [x for x in label if x in top_distk]
            print('method 2', tp)

class PinSAGEModel(nn.Module):
    def __init__(self, full_graph, ntype, textsets, hidden_dims, n_layers):
        super().__init__()

        self.proj = layers.LinearProjector(full_graph, ntype, textsets, hidden_dims)
        self.sage = layers.SAGENet(hidden_dims, n_layers)
        self.scorer = layers.ItemToItemScorer(full_graph, ntype)

    def forward(self, pos_graph, neg_graph, blocks):
        h_item = self.get_repr(blocks)
        pos_score = self.scorer(pos_graph, h_item)
        neg_score = self.scorer(neg_graph, h_item)
        return (neg_score - pos_score + 1).clamp(min=0)

    def get_repr(self, blocks):
        h_item = self.proj(blocks[0].srcdata)
        h_item_dst = self.proj(blocks[-1].dstdata)
        return h_item_dst + self.sage(blocks, h_item)
        
def load_model(data_dict, device, args):
    gnn = PinSAGEModel(data_dict['graph'], data_dict['item_ntype'], data_dict['textset'], args.hidden_dims, args.num_layers).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=args.lr)
    if args.retrain:
        checkpoint = torch.load(args.save_path + '.pt', map_location=device)
    else:
        checkpoint = torch.load(args.save_path, map_location=device)
   
    gnn.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['optimizer_state_dict'])

    return gnn, opt, checkpoint['epoch']

def prepare_dataset(data_dict, args):
    g = data_dict['graph']
    item_texts = data_dict['item_texts']
    user_ntype = data_dict['user_ntype']
    item_ntype = data_dict['item_ntype']

    # Assign IDs to user & wine under 'id' feature
    # (to learn an individual trainable embedding for each entity)
    g.nodes[user_ntype].data['id'] = torch.arange(g.number_of_nodes(user_ntype))
    g.nodes[item_ntype].data['id'] = torch.arange(g.number_of_nodes(item_ntype))
    data_dict['graph'] = g

    # Prepare torchtext dataset and vocabulary
    if not len(item_texts):
        data_dict['textset'] = None
    else:
        fields = {} # {'name': <torchtext.data.field.Field at 0x7ffaf4fb45b0>}
        examples = [] # [[wine_name, [('name', fields['name'])]]]
        for key, texts in item_texts.items():
            fields[key] = torchtext.data.Field(include_lengths=True, lower=True, batch_first=True)
        for i in range(g.number_of_nodes(item_ntype)):
            example = torchtext.data.Example.fromlist(
                [item_texts[key][i] for key in item_texts.keys()],
                [(key, fields[key]) for key in item_texts.keys()])
            examples.append(example)
            
        textset = torchtext.data.Dataset(examples, fields)
        for key, field in fields.items():
            field.build_vocab(getattr(textset, key))
            #field.build_vocab(getattr(textset, key), vectors='fasttext.simple.300d')
        data_dict['textset'] = textset

    return data_dict

def prepare_dataloader(data_dict, args):
    g = data_dict['graph']
    user_ntype = data_dict['user_ntype']
    item_ntype = data_dict['item_ntype']
    textset = data_dict['textset']
    # batch_sampler >> this will act as ground truth value
    # Sampler returns [[starting item nodes (heads)], [related item nodes (pos tails)], [not related item nodes (neg tails)]] >> size is per given batch
    batch_sampler = sampler_module.ItemToItemBatchSampler(
        g, user_ntype, item_ntype, args.batch_size)
    
    # get input >> provides n number of neighbors for the given node as per random walks
    # last argument is optional which is not available here but is the weight for edges or connections
    neighbor_sampler = sampler_module.NeighborSampler(
        g, user_ntype, item_ntype, args.random_walk_length,
        args.random_walk_restart_prob, args.num_random_walks, args.num_neighbors,
        args.num_layers)
    
    # this will call batch_sampler and then provide neighbors on heads
    collator = sampler_module.PinSAGECollator(neighbor_sampler, g, item_ntype, textset)
    dataloader = DataLoader(
        batch_sampler,
        collate_fn=collator.collate_train,
        num_workers=args.num_workers)

    dataloader_test = DataLoader(
        torch.arange(g.number_of_nodes(item_ntype)),
        batch_size=args.batch_size,
        collate_fn=collator.collate_test,
        num_workers=args.num_workers)
    dataloader_it = iter(dataloader)

    return dataloader_it, dataloader_test, neighbor_sampler
    
def train(data_dict, args):
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('Current using CPUs')
    else:
        print ('Current cuda device ', torch.cuda.current_device()) # check

    # Dataset (adds another key with text encoded data)
    data_dict = prepare_dataset(data_dict, args)
    dataloader_it, dataloader_test, neighbor_sampler = prepare_dataloader(data_dict, args)
    
    # Model
    if args.retrain:
        print('Loading pretrained model...')
        gnn, opt, start_epoch = load_model(data_dict, device, args)
    else:
        gnn = PinSAGEModel(data_dict['graph'], data_dict['item_ntype'], data_dict['textset'], args.hidden_dims, args.num_layers)
        opt = torch.optim.Adam(gnn.parameters(), lr=args.lr)
        start_epoch = 0


    if args.eval_epochs:
        g = data_dict['graph']
        item_ntype = data_dict['item_ntype']
        user_ntype = data_dict['user_ntype']
        user_to_item_etype = data_dict['user_to_item_etype']
        timestamp = data_dict['timestamp']
        nid_uid_dict = {v: k for v, k in enumerate(list(g.ndata['userID'].values())[0].numpy())}
        nid_wid_dict = {nid.item(): wid.item() for wid, nid in  zip(g.ndata['wine_id']['wine'], g.ndata['id']['wine'])}


    gnn = gnn.to(device)
    for epoch in tqdm(range(start_epoch, args.num_epochs + start_epoch)):
        gnn.train()
        for batch in range(args.batches_per_epoch):
            pos_graph, neg_graph, blocks = next(dataloader_it)
            for i in range(len(blocks)):
                blocks[i] = blocks[i].to(device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)

            loss = gnn(pos_graph, neg_graph, blocks).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Evaluate
        if not epoch:
            continue
        
        if args.eval_epochs and not epoch % args.eval_epochs:
            print('\nEvaluating...')
            # get embedding of all items (wine)
            h_item = evaluation.get_all_emb(gnn, g.ndata['id'][item_ntype], data_dict['textset'], item_ntype, neighbor_sampler, args.batch_size, device)
            # get interacted node ids for all users
            item_batch = evaluation.item_by_user_batch(g, user_ntype, item_ntype, user_to_item_etype, timestamp, args)
            recalls = []
            precisions = [] 
            hitrates = []

            recalls1 = []
            precisions1 = [] 
            hitrates1 = []

            users = []

            for i, nodes in tq_(enumerate(item_batch)):
                '''
                nodes : Actual interaction nodes per user [train node, test node (8: 2 ratio)]
                '''
                # get Real user ID from given ID
                category = nid_uid_dict[i]
                user_id = data_dict['user_category'][category]  # real user id
                label = data_dict['testset'][user_id]  # test label
                users.append(user_id)

                # Explore Real Wine IDs
                item = evaluation.node_to_item(nodes, nid_wid_dict, data_dict['item_category'])  # wine ID
                # get interacted wine IDs index that are available in testset
                label_idx = [i for i, x in enumerate(item) if x in label]  # label index

                # Item Recommendation
                test_nodes = [x for i, x in enumerate(nodes) if i in label_idx]  # label index Nodes for testing without input
                # get nodes that are not in test set i.e. they are in training set
                train_nodes = [x for i, x in enumerate(nodes) if i not in label_idx]  # label index Nodes for training without input
                if len(train_nodes)>0 and len(test_nodes)>0:
                        
                    # get embedding of above nodes i.e. embedding of training data
                    h_nodes = h_item[train_nodes]

                    # method 1
                    # get center embedding of training data
                    h_center = torch.mean(h_nodes, axis=0) 
                    # get distance between above embedding with rest of the embeddings 
                    dist = h_center @ h_item.t()  # matrix multiplication
                    # dist Extract k items in order of size
                    topk = dist.topk(args.k)[1].cpu().numpy()  
                    topk = evaluation.node_to_item(topk, nid_wid_dict, data_dict['item_category'])  # ID conversion

                    # now check if any recommendations were from wine IDs that are in test set
                    tp = [x for x in label if x in topk]
                    if not tp:
                        # if none of the recommendations were in testset
                        recall, precision, hitrate = 0, 0, 0
                    else:
                        # if recommendations were available in testset
                        # out of total items in testset we recommended
                        recall = len(tp) / len(label) 
                        # out of total recommendations how many were in testset
                        precision = len(tp) / len(topk)
                        hitrate = 1  # There is at least one

                    recalls.append(recall)
                    precisions.append(precision)
                    hitrates.append(hitrate)
                
                    # method 2: cosine distance
                    cos_dist = cosine_distances(h_nodes, h_item)
                    top_distk = np.argsort(cos_dist)[:, :args.k].flatten()
                    top_distk = evaluation.node_to_item(top_distk, nid_wid_dict, data_dict['item_category'])
                    tp1 = [x for x in label if x in top_distk]
                    if not tp1:
                        # if none of the recommendations were in testset
                        recall1, precision1, hitrate1 = 0, 0, 0
                    else:
                        # if recommendations were available in testset
                        # out of total items in testset we recommended
                        recall1 = len(set(tp1)) / len(label) 
                        # out of total recommendations how many were in testset
                        precision1 = len(set(tp1)) / len(set(top_distk))
                        hitrate1 = 1  # There is at least one

                    recalls1.append(recall1)
                    precisions1.append(precision1)
                    hitrates1.append(hitrate1)
            
            result_df = pd.DataFrame({'recall': recalls, 'precision': precisions, 'hitrate': hitrates})
            result_df = result_df.mean().apply(lambda x: round(x, 3))
            recall, precision, hitrate = result_df['recall'], result_df['precision'], result_df['hitrate']
            print(f'\tEpoch:{epoch}\tRecall:{recall}\tHitrate:{hitrate}\tPrecision:{precision}')

            result_df1 = pd.DataFrame({'recall1': recalls1, 'precision1': precisions1, 'hitrate1': hitrates1})
            result_df1 = result_df1.mean().apply(lambda x: round(x, 3))
            recall1, precision1, hitrate1 = result_df1['recall1'], result_df1['precision1'], result_df1['hitrate1']
            print(f'\tEpoch:{epoch}\tRecall1:{recall1}\tHitrate1:{hitrate1}\tPrecision1:{precision1}')

        if args.save_epochs:
            if not epoch % args.save_epochs:
                torch.save({
                'epoch': epoch,
                'model_state_dict': gnn.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss
                        }, args.save_path + '_' + str(epoch) + 'epoch.pt')

    return gnn, epoch+1, opt, loss

# %%
if __name__ == '__main__':
    
    # Namespace(dataset_path='data.pkl', save_path='', random_walk_length=2, 
    #                  random_walk_restart_prob=0.5, num_random_walks=10,
    #                  num_neighbors=3, num_layers=2, hidden_dims=16, batch_size=64,
    #                  device='cpu', num_epochs=25, batches_per_epoch=5000, num_workers=0, 
    #                  lr=3e-4, eval_epochs=1, save_epochs=0, retrain=0, k=10)
    
    args = Namespace(dataset_path='data.pkl', save_path='', random_walk_length=2, 
                     random_walk_restart_prob=0.5, num_random_walks=10,
                     num_neighbors=3, num_layers=2, hidden_dims=128, batch_size=128,
                     device='cpu', num_epochs=499, batches_per_epoch=256, num_workers=4, 
                     lr=3e-5, eval_epochs=10, save_epochs=10, retrain=0, k=500)
    
    with open(args.dataset_path, 'rb') as f:
        dataset = pickle.load(f)

    data_dict = {
        'graph': dataset['train-graph'],
        'val_matrix': None,
        'test_matrix': None,
        'item_texts': dataset['item-texts'],
        'testset': dataset['testset'], 
        'new_ratings': dataset['rating-data'],
        'user_ntype': dataset['user-type'],
        'item_ntype': dataset['item-type'],
        'user_to_item_etype': dataset['user-to-item-type'],
        'timestamp': dataset['timestamp-edge-column'],
        'user_category': dataset['user-category'], 
        'item_category': dataset['item-category']
    }
    
   
    # Training
    gnn, epoch, opt, loss = train(data_dict, args)


    torch.save({
                'epoch': epoch,
                'model_state_dict': gnn.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss
            }, args.save_path + '_' + str(epoch) + 'epoch.pt')

    #get_recommendations(data_dict, args, '_500epoch.pt')
