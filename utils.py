import argparse

import numpy as np
import torch
import torch.nn as nn
from torch import optim as optim

import dgl
from sklearn import metrics
from libs.my_munkres import Munkres


from scipy.optimize import linear_sum_assignment
def kmeans_gpu(X, n_clusters, max_iter=300, tol=1e-4, device='cuda'):

    centers = X[torch.randint(0, X.shape[0], (n_clusters,))].clone()
    
    for _ in range(max_iter):
        dists = torch.cdist(X, centers)  # (N, K)
        labels = torch.argmin(dists, dim=1)
        
        new_centers = torch.stack([
            X[labels == k].mean(dim=0) if (labels == k).any() else centers[k]
            for k in range(n_clusters)
        ])
        
        if torch.norm(new_centers - centers) < tol:
            break
        centers = new_centers
    
    return labels

def build_args():
    parser = argparse.ArgumentParser(description="GAT")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--missing_rate", type=float, required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--max_epoch", type=int, required=True,
                        help="number of training epochs")

    parser.add_argument("--APA_para", type=float, default=10)
    parser.add_argument("--DEG_para", type=float, default=10)
    parser.add_argument("--decoder_AH_type", type=str, required=True, choices=["cat", "mean"])
    parser.add_argument("--loss_DEG_A_para", type=float, default=1)
    parser.add_argument("--loss_DEG_Z_para", type=float, default=250)
    parser.add_argument("--loss_DEG_H_para", type=float, default=0.5)
    
    parser.add_argument("--loss_APA_A2H_para", type=float, default=0.1)
    parser.add_argument("--loss_APA_H2A_para", type=float, default=0.9)

    parser.add_argument("--hyperbuild", type=int, default=3, help="hypergraph build type")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0,
                        help="weight decay")
    parser.add_argument("--optimizer", type=str, default="adam")

    parser.add_argument("--num_heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--num_out_heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num_hidden", type=int, default=512,
                        help="number of hidden units")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--attn_drop", type=float, default=.1,
                        help="attention dropout")
    parser.add_argument("--in_drop", type=float, default=.2,
                        help="input feature dropout")
    parser.add_argument("--norm", type=str, default=None)
    parser.add_argument("--negative_slope", type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument("--encoder", type=str, default="gat")
    parser.add_argument("--decoder", type=str, default="mlp")
    parser.add_argument("--activation", type=str, default="prelu")
    parser.add_argument("--num_dec_layers", type=int, default=1)

    args = parser.parse_args()
    return args


def create_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "selu":
        return nn.SELU()
    elif name == "elu":
        return nn.ELU()
    elif name == "silu":
        return nn.SiLU()
    elif name is None:
        return nn.Identity()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def identity_norm(x):
    def func(x):
        return x

    return func


def create_norm(name):
    if name == "layernorm":
        return nn.LayerNorm
    elif name == "batchnorm":
        return nn.BatchNorm1d
    elif name == "identity":
        return identity_norm
    else:
        return None


def create_optimizer(opt, model, lr, weight_decay, get_num_layer=None, get_layer_scale=None):
    opt_lower = opt.lower()
    parameters = model.parameters()
    opt_args = dict(lr=lr, weight_decay=weight_decay)

    opt_split = opt_lower.split("_")
    opt_lower = opt_split[-1]

    if opt_lower == "adam":
        optimizer = optim.Adam(parameters, **opt_args)
    elif opt_lower == "adamw":
        optimizer = optim.AdamW(parameters, **opt_args)
    elif opt_lower == "adadelta":
        optimizer = optim.Adadelta(parameters, **opt_args)
    elif opt_lower == "sgd":
        opt_args["momentum"] = 0.9
        return optim.SGD(parameters, **opt_args)
    else:
        raise NotImplementedError("Invalid optimizer")

    return optimizer

def create_scheduler(optimizer):
    return optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=10,
        gamma=0.9
    )

# -------------------
def mask_edge(graph, mask_prob):
    E = graph.num_edges()

    mask_rates = torch.ones(E) * mask_prob
    masks = torch.bernoulli(1 - mask_rates)
    mask_idx = masks.nonzero().squeeze(1)
    return mask_idx

from scipy.sparse import lil_matrix
def load_missing_graph_dataset(dataset_name, missing_rate=0.6, show_details=False, Hmask=1):
    load_path = "data/" + dataset_name + "/" + dataset_name
    
    feat = np.load(f'{load_path}_feat_missing_{missing_rate}.npy', allow_pickle=True)
    random_values = np.random.rand(*feat.shape) * 1e-6 + 1e-6
    feat[feat == -1] = random_values[feat == -1]
    
    zero_rows = np.all(feat == 0, axis=1)
    feat[zero_rows] = random_values[zero_rows]
    
    label = np.load(load_path + "_label.npy", allow_pickle=True)
    adj = np.load(load_path + "_adj.npy", allow_pickle=True)
    
    for i in range(adj.shape[0]):
        if np.all(adj[i, :] == 0):
            adj[i, i] = 1
    
    node_mask = np.load(f'{load_path}_missing_mask_{missing_rate}.npy', allow_pickle=True)
    missing_index = np.where(node_mask)[0]  
    
    cluster_num = len(np.unique(label))
    node_num = feat.shape[0]
    show_details=True
    if show_details:
        print("++++++++++++++++++++++++++++++")
        print("---details of graph dataset---")
        print("++++++++++++++++++++++++++++++")
        print("dataset name:   ", dataset_name)
        print("missing rate:   ", missing_rate)
        print("missing nodes:  ", len(missing_index), "/", node_num)
        print("feature shape:  ", feat.shape)
        print("label shape:    ", label.shape)
        print("adj shape:      ", adj.shape)
        print("undirected edge num:   ", int(np.nonzero(adj)[0].shape[0] / 2))
        print("category num:          ", cluster_num)
        print("category distribution: ")
        for i in range(cluster_num):
            count = np.sum(label == i)
            print(f"label {i}: {count} nodes")
        print("++++++++++++++++++++++++++++++")


    feat_tensor = torch.tensor(feat, dtype=torch.float32)
    adj_tensor = torch.tensor(adj, dtype=torch.int64)
    label_tensor = torch.tensor(label, dtype=torch.int64)

    # build DGL
    src_adj, dst_adj = adj_tensor.nonzero(as_tuple=True)

    graph_adj = dgl.graph((src_adj, dst_adj))
    graph_adj.ndata['feat'] = feat_tensor
    graph_adj.ndata['label'] = label_tensor
    graph_adj = dgl.add_self_loop(graph_adj)

    hypergraph_incidence = get_similarity_hypergraph(feat, missing_index, adj, Hmask)
    src, dst = hypergraph_incidence.nonzero()
    graph_hyper = dgl.heterograph({
        ('node', 'in', 'hyperedge'): (src, dst),
        ('hyperedge', 'has', 'node'): (dst, src)
    })
    missing_mask = torch.zeros(graph_hyper.num_nodes('node'), dtype=torch.bool)
    missing_mask[missing_index] = True
    graph_hyper.nodes['node'].data.update({
        'feat': feat_tensor,
        'label': label_tensor,
        'missing': missing_mask
    })
    return graph_adj, graph_hyper, missing_index, (feat.shape[1], cluster_num)

'''
def get_similarity_hypergraph(feat, missing_index, adj, K=5):
    N = feat.shape[0]
    H = np.zeros((N, 0), dtype=int)
    
    complete_nodes = np.setdiff1d(np.arange(N), missing_index)
    for u in complete_nodes:
        feat_u = feat[u] / (np.linalg.norm(feat[u]) + 1e-8)
        sim = np.dot(feat, feat_u) / (np.linalg.norm(feat, axis=1) + 1e-8)
        topk = np.argsort(-sim)[:K+1]
        H = np.hstack([H, np.zeros((N, 1), dtype=int)])
        H[topk, -1] = 1
    
    for u in missing_index:
        neighbors = np.where(adj[u] > 0)[0]
        if len(neighbors) == 0:
            neighbors = [u]
        H = np.hstack([H, np.zeros((N, 1), dtype=int)])
        H[neighbors, -1] = 1
    
    unique_edges = np.unique(H.T, axis=0)
    H = unique_edges.T.astype(int)
    return H 
'''

def get_similarity_hypergraph(feat, missing_index, adj, Hmask):
    print(Hmask)
    N = feat.shape[0]
    H_list = []
    adj_list=[]
    for u in range(N):
        adj_list.append(set(np.where(adj[u] > 0)[0]))

    hyperedge_set= set()
    # using direct neighbors as the hyperedge of each node
    for u in range(N):
        neighbors = adj_list[u]
        second_neighbors = set()
        for neighbor in neighbors:
            second_neighbors |= adj_list[neighbor]
        second_neighbors -= neighbors | {u}

        third_neighbors = set()
        for second_neighbor in second_neighbors:
            third_neighbors |= adj_list[second_neighbor]
        third_neighbors -= neighbors | second_neighbors | {u}
        
        all_neighbors = set()
        if (Hmask   )&1: all_neighbors |= {u}
        if (Hmask>>1)&1: all_neighbors |= neighbors
        if (Hmask>>2)&1: all_neighbors |= second_neighbors
        if (Hmask>>3)&1: all_neighbors |= third_neighbors
        if not neighbors:all_neighbors |= {u}
        if not all_neighbors:continue
        hyperedge_tuple = tuple(sorted(all_neighbors))
        
        if hyperedge_tuple not in hyperedge_set:
            hyperedge_set.add(hyperedge_tuple)
            
            hyperedge_vector = np.zeros(N, dtype=int)
            hyperedge_vector[list(all_neighbors)] = 1
            
            H_list.append(hyperedge_vector)

    H = np.array(H_list).T if H_list else np.zeros((N, 0), dtype=int)
    return H


def cluster_probing_full_batch(model, graph, x, device):
    model.eval()
    with torch.no_grad():
        x = model.embed(graph.to(device), x.to(device))

    labels = graph.ndata["label"]

    #print(x.shape)
    results_str = clustering(x, labels, device)

    return results_str


def clustering(embeds, labels, device):
    if not isinstance(embeds, torch.Tensor):
        embeds = torch.tensor(embeds, device=device)
    else:
        embeds = embeds.to(device)

    num_classes = torch.max(labels).item() + 1
    results = {"acc": 0, "nmi": 0, "ari": 0, "f1": 0}
    
    best_metrics = torch.zeros(10, 4, device='cpu')  
    
    for i in range(10):
        all_labels = []
        for _ in range(10):
            predY = kmeans_gpu(embeds, num_classes, device=device)
            all_labels.append(predY.cpu().numpy())
        
        for j, predY_np in enumerate(all_labels):
            gnd_Y = bestMap(predY_np, labels.cpu().numpy())
            acc, f1, nmi, ari, _ = clustering_metrics(gnd_Y, predY_np)
            if acc > best_metrics[i, 0]:
                best_metrics[i] = torch.tensor([acc, nmi, ari, f1])
    
    avg_metrics = best_metrics.mean(dim=0)
    results = {
        "acc": avg_metrics[0].item(),
        "nmi": avg_metrics[1].item(),
        "ari": avg_metrics[2].item(),
        "f1": avg_metrics[3].item()
    }
    
    return results


def bestMap(L1, L2):
    '''
    bestmap: permute labels of L2 to match L1 as good as possible
        INPUT:
            L1: labels of L1, shape of (N,) vector
            L2: labels of L2, shape of (N,) vector
        OUTPUT:
            new_L2: best matched permuted L2, shape of (N,) vector
    version 1.0 --December/2018
    Modified from bestMap.m (written by Deng Cai)
    '''
    if L1.shape[0] != L2.shape[0] or len(L1.shape) > 1 or len(L2.shape) > 1:
        raise Exception('L1 shape must equal L2 shape')
        return
    Label1 = np.unique(L1)
    nClass1 = Label1.shape[0]
    Label2 = np.unique(L2)
    nClass2 = Label2.shape[0]
    nClass = max(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        for j in range(nClass2):
            G[j, i] = np.sum((np.logical_and(L1 == Label1[i], L2 == Label2[j])).astype(np.int64))
    c, t = linear_sum_assignment(-G)
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        try:
            newL2[L2 == Label2[i]] = Label1[t[i]]
        except:
            pass # some labels may not be matched
    return newL2


def clustering_metrics(true_label, pred_label):
    l1 = list(set(true_label))
    numclass1 = len(l1)

    l2 = list(set(pred_label))
    numclass2 = len(l2)
    if numclass1 != numclass2:
        print('Class Not equal, Error!!!!')
        return 0, 0, 0, 0, 0

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(true_label) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if pred_label[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()

    indexes = m.compute(cost)
    idx = indexes[2][1]
    # get the match results
    new_predict = np.zeros(len(pred_label))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]
        # ai is the index with label==c2 in the predict list
        ai = [ind for ind, elm in enumerate(pred_label) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(true_label, new_predict)
    f1_macro = metrics.f1_score(true_label, new_predict, average='macro')
    nmi = metrics.normalized_mutual_info_score(true_label, pred_label)
    ari = metrics.adjusted_rand_score(true_label, pred_label)

    return acc * 100, f1_macro * 100, nmi * 100, ari * 100, idx