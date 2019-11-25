import torch
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import torch.utils.data
import itertools
from collections import Counter
from random import shuffle
import json
#
from networkx.readwrite import json_graph
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import pdb
import time
import random
import pickle
import os.path
import torch_geometric as tg
import torch_geometric.datasets
import time

from torch_geometric.data import Data, DataLoader

from utils import precompute_dist_data, get_link_mask, duplicate_edges, deduplicate_edges


def get_tg_dataset(args, dataset_name, use_cache=True, remove_feature=False):
    # "Cora", "CiteSeer" and "PubMed"
    if dataset_name in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = tg.datasets.Planetoid(root='datasets/' + dataset_name, name=dataset_name)
    else:
        try:
            dataset = load_tg_dataset(dataset_name)
        except:
            raise NotImplementedError

    # precompute shortest path
    if not os.path.isdir('datasets'):
        os.mkdir('datasets')
    if not os.path.isdir('datasets/cache'):
        os.mkdir('datasets/cache')
    f1_name = 'datasets/cache/' + dataset_name + str(args.approximate) + '_dists.dat'
    f2_name = 'datasets/cache/' + dataset_name + str(args.approximate)+ '_dists_removed.dat'
    f3_name = 'datasets/cache/' + dataset_name + str(args.approximate)+ '_links_train.dat'
    f4_name = 'datasets/cache/' + dataset_name + str(args.approximate)+ '_links_val.dat'
    f5_name = 'datasets/cache/' + dataset_name + str(args.approximate)+ '_links_test.dat'

    if use_cache and ((os.path.isfile(f2_name) and args.task=='link') or (os.path.isfile(f1_name) and args.task!='link')):
        with open(f3_name, 'rb') as f3, \
            open(f4_name, 'rb') as f4, \
            open(f5_name, 'rb') as f5:
            links_train_list = pickle.load(f3)
            links_val_list = pickle.load(f4)
            links_test_list = pickle.load(f5)
        if args.task=='link':
            with open(f2_name, 'rb') as f2:
                dists_removed_list = pickle.load(f2)
        else:
            with open(f1_name, 'rb') as f1:
                dists_list = pickle.load(f1)

        print('Cache loaded!')
        data_list = []
        for i, data in enumerate(dataset):
            if args.task == 'link':
                data.mask_link_positive = deduplicate_edges(data.edge_index.numpy())
            data.mask_link_positive_train = links_train_list[i]
            data.mask_link_positive_val = links_val_list[i]
            data.mask_link_positive_test = links_test_list[i]
            get_link_mask(data, resplit=False)

            if args.task=='link':
                data.dists = torch.from_numpy(dists_removed_list[i]).float()
                data.edge_index = torch.from_numpy(duplicate_edges(data.mask_link_positive_train)).long()
            else:
                data.dists = torch.from_numpy(dists_list[i]).float()
            if remove_feature:
                data.x = torch.ones((data.x.shape[0],1))
            data_list.append(data)
    else:
        data_list = []
        dists_list = []
        dists_removed_list = []
        links_train_list = []
        links_val_list = []
        links_test_list = []
        for i, data in enumerate(dataset):
            if 'link' in args.task:
                get_link_mask(data, args.remove_link_ratio, resplit=True,
                              infer_link_positive=True if args.task == 'link' else False)
            links_train_list.append(data.mask_link_positive_train)
            links_val_list.append(data.mask_link_positive_val)
            links_test_list.append(data.mask_link_positive_test)
            if args.task=='link':
                dists_removed = precompute_dist_data(data.mask_link_positive_train, data.num_nodes,
                                                     approximate=args.approximate)
                dists_removed_list.append(dists_removed)
                data.dists = torch.from_numpy(dists_removed).float()
                data.edge_index = torch.from_numpy(duplicate_edges(data.mask_link_positive_train)).long()

            else:
                dists = precompute_dist_data(data.edge_index.numpy(), data.num_nodes, approximate=args.approximate)
                dists_list.append(dists)
                data.dists = torch.from_numpy(dists).float()
            if remove_feature:
                data.x = torch.ones((data.x.shape[0],1))
            data_list.append(data)

        with open(f1_name, 'wb') as f1, \
            open(f2_name, 'wb') as f2, \
            open(f3_name, 'wb') as f3, \
            open(f4_name, 'wb') as f4, \
            open(f5_name, 'wb') as f5:

            if args.task=='link':
                pickle.dump(dists_removed_list, f2)
            else:
                pickle.dump(dists_list, f1)
            pickle.dump(links_train_list, f3)
            pickle.dump(links_val_list, f4)
            pickle.dump(links_test_list, f5)
        print('Cache saved!')
    return data_list


def nx_to_tg_data(graphs, features, edge_labels=None):
    data_list = []
    for i in range(len(graphs)):
        feature = features[i]
        graph = graphs[i].copy()
        graph.remove_edges_from(graph.selfloop_edges())

        # relabel graphs
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))
        nx.relabel_nodes(graph, mapping, copy=False)

        x = np.zeros(feature.shape)
        graph_nodes = list(graph.nodes)
        for m in range(feature.shape[0]):
            x[graph_nodes[m]] = feature[m]
        x = torch.from_numpy(x).float()

        # get edges
        edge_index = np.array(list(graph.edges))
        edge_index = np.concatenate((edge_index, edge_index[:,::-1]), axis=0)
        edge_index = torch.from_numpy(edge_index).long().permute(1,0)

        data = Data(x=x, edge_index=edge_index)
        # get edge_labels
        if edge_labels[0] is not None:
            edge_label = edge_labels[i]
            mask_link_positive = np.stack(np.nonzero(edge_label))
            data.mask_link_positive = mask_link_positive
        data_list.append(data)
    return data_list



def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def Graph_load_batch(min_num_nodes = 20, max_num_nodes = 1000, name = 'ENZYMES',node_attributes = True,graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    print('Loading graph dataset: '+str(name))
    G = nx.Graph()
    # load data
    path = 'data/'+name+'/'
    data_adj = np.loadtxt(path+name+'_A.txt', delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(path+name+'_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path+name+'_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path+name+'_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path+name+'_graph_labels.txt', delimiter=',').astype(int)


    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]
        if G_sub.number_of_nodes()>=min_num_nodes and G_sub.number_of_nodes()<=max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
    print('Loaded')
    return graphs, data_node_att, data_node_label



# main data load function
def load_graphs(dataset_str):
    node_labels = [None]
    edge_labels = [None]
    idx_train = [None]
    idx_val = [None]
    idx_test = [None]

    if dataset_str == 'grid':
        graphs = []
        features = []
        for _ in range(1):
            graph = nx.grid_2d_graph(20, 20)
            graph = nx.convert_node_labels_to_integers(graph)

            feature = np.identity(graph.number_of_nodes())
            graphs.append(graph)
            features.append(feature)

    elif dataset_str == 'communities':
        graphs = []
        features = []
        node_labels = []
        edge_labels = []
        for i in range(1):
            community_size = 20
            community_num = 20
            p=0.01

            graph = nx.connected_caveman_graph(community_num, community_size)

            count = 0

            for (u, v) in graph.edges():
                if random.random() < p:  # rewire the edge
                    x = random.choice(list(graph.nodes))
                    if graph.has_edge(u, x):
                        continue
                    graph.remove_edge(u, v)
                    graph.add_edge(u, x)
                    count += 1
            print('rewire:', count)

            n = graph.number_of_nodes()
            label = np.zeros((n,n),dtype=int)
            for u in list(graph.nodes):
                for v in list(graph.nodes):
                    if u//community_size == v//community_size and u>v:
                        label[u,v] = 1
            rand_order = np.random.permutation(graph.number_of_nodes())
            feature = np.identity(graph.number_of_nodes())[:,rand_order]
            graphs.append(graph)
            features.append(feature)
            edge_labels.append(label)

    elif dataset_str == 'protein':

        graphs_all, features_all, labels_all = Graph_load_batch(name='PROTEINS_full')
        features_all = (features_all-np.mean(features_all,axis=-1,keepdims=True))/np.std(features_all,axis=-1,keepdims=True)
        graphs = []
        features = []
        edge_labels = []
        for graph in graphs_all:
            n = graph.number_of_nodes()
            label = np.zeros((n, n),dtype=int)
            for i,u in enumerate(graph.nodes()):
                for j,v in enumerate(graph.nodes()):
                    if labels_all[u-1] == labels_all[v-1] and u>v:
                        label[i,j] = 1
            if label.sum() > n*n/4:
                continue

            graphs.append(graph)
            edge_labels.append(label)

            idx = [node-1 for node in graph.nodes()]
            feature = features_all[idx,:]
            features.append(feature)

        print('final num', len(graphs))


    elif dataset_str == 'email':

        with open('data/email.txt', 'rb') as f:
            graph = nx.read_edgelist(f)

        label_all = np.loadtxt('data/email_labels.txt')
        graph_label_all = label_all.copy()
        graph_label_all[:,1] = graph_label_all[:,1]//6


        for edge in list(graph.edges()):
            if graph_label_all[int(edge[0])][1] != graph_label_all[int(edge[1])][1]:
                graph.remove_edge(edge[0], edge[1])

        comps = [comp for comp in nx.connected_components(graph) if len(comp)>10]
        graphs = [graph.subgraph(comp) for comp in comps]

        edge_labels = []
        features = []

        for g in graphs:
            n = g.number_of_nodes()
            feature = np.ones((n, 1))
            features.append(feature)

            label = np.zeros((n, n),dtype=int)
            for i, u in enumerate(g.nodes()):
                for j, v in enumerate(g.nodes()):
                    if label_all[int(u)][1] == label_all[int(v)][1] and i>j:
                        label[i, j] = 1
            label = label
            edge_labels.append(label)


    elif dataset_str == 'ppi':
        dataset_dir = 'data/ppi'
        print("Loading data...")
        G = json_graph.node_link_graph(json.load(open(dataset_dir + "/ppi-G.json")))
        edge_labels_internal = json.load(open(dataset_dir + "/ppi-class_map.json"))
        edge_labels_internal = {int(i): l for i, l in edge_labels_internal.items()}

        train_ids = [n for n in G.nodes()]
        train_labels = np.array([edge_labels_internal[i] for i in train_ids])
        if train_labels.ndim == 1:
            train_labels = np.expand_dims(train_labels, 1)

        print("Using only features..")
        feats = np.load(dataset_dir + "/ppi-feats.npy")
        ## Logistic gets thrown off by big counts, so log transform num comments and score
        feats[:, 0] = np.log(feats[:, 0] + 1.0)
        feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
        feat_id_map = json.load(open(dataset_dir + "/ppi-id_map.json"))
        feat_id_map = {int(id): val for id, val in feat_id_map.items()}
        train_feats = feats[[feat_id_map[id] for id in train_ids]]

        node_dict = {}
        for id,node in enumerate(G.nodes()):
            node_dict[node] = id

        comps = [comp for comp in nx.connected_components(G) if len(comp)>10]
        graphs = [G.subgraph(comp) for comp in comps]

        id_all = []
        for comp in comps:
            id_temp = []
            for node in comp:
                id = node_dict[node]
                id_temp.append(id)
            id_all.append(np.array(id_temp))

        features = [train_feats[id_temp,:]+0.1 for id_temp in id_all]

    else:
        raise NotImplementedError

    return graphs, features, edge_labels, node_labels, idx_train, idx_val, idx_test


def load_tg_dataset(name='communities'):
    graphs, features, edge_labels,_,_,_,_ = load_graphs(name)
    return nx_to_tg_data(graphs, features, edge_labels)