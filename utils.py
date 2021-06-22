import numpy as np
import scipy.sparse as sp
import torch
import json


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

'''
def load_data():
    print('Loading dataset...')
    # adj = np.load('NASDAQ_wiki_relation.npy')
    adj = np.load('graph.npy')
    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj
'''
# def build_graph(connection_file, tic_wiki_file,
                        # sel_path_file):
def load_data():
    print('Loading dataset...')
    connection_file='Data/relation/NASDAQ_connections.json'
    tic_wiki_file='Data/relation/NASDAQ_wiki.csv' 
    sel_path_file='Data/relation/selected_wiki_connections.csv'

    # readin tickers => col1: abbreviation, col2: wikidata index 
    idx_labels = np.genfromtxt(tic_wiki_file, dtype=str, delimiter=',',
                            skip_header=False)
    # build graph   
    idx_map = {j[1]: i for i, j in enumerate(idx_labels) if j[1] != 'unknown'}


    # readin selected paths/connections
    sel_paths = np.genfromtxt(sel_path_file, dtype=str, delimiter=' ',
                              skip_header=False)
    sel_paths = set(sel_paths[:, 0])

    # readin connections
    with open(connection_file, 'r') as fin:
        connections = json.load(fin)
    
    # get occured paths
    edges_unordered = []
    for key1, conns in connections.items():
        for key2, paths in conns.items():
            for p in paths:
                path_key = '_'.join(p)
                if path_key in sel_paths:
                    edges_unordered.append([key1, key2])
                    continue

    edges_unordered = np.array(edges_unordered)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)

    adj = np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])
    adj = sp.coo_matrix(adj, dtype=np.float32)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = torch.FloatTensor(np.array(adj.todense()))
    return adj

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

if __name__ == '__main__':
    load_data()