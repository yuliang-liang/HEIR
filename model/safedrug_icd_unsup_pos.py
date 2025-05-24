from collections import defaultdict
from copy import deepcopy
from typing import List, Tuple, Dict, Optional
import os

import numpy as np
import rdkit.Chem.BRICS as BRICS
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from pyhealth.datasets import SampleEHRDataset
from pyhealth.medcode import ATC
from pyhealth.metrics import ddi_rate_score
from pyhealth.models import BaseModel
from pyhealth.models.utils import get_last_visit
from pyhealth import BASE_CACHE_PATH as CACHE_PATH
from pyhealth.medcode import InnerMap

from torch_geometric.utils import subgraph,k_hop_subgraph

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torch_cluster import random_walk
from sklearn.linear_model import LogisticRegression

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, GCNConv
from torch_geometric.data import NeighborSampler as RawNeighborSampler

import matplotlib.pyplot as plt
import seaborn as sns
from pyhealth.medcode import InnerMap
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool

from transformers import BertTokenizer, BertModel


class MaskLinear(nn.Module):
    """MaskLinear layer.

    This layer wraps the PyTorch linear layer and adds a hard mask for
    the parameter matrix. It is used in the SafeDrug model.

    Args:
        in_features: input feature size.
        out_features: output feature size.
        bias: whether to use bias. Default is True.
    """

    def __init__(self, in_features: int, out_features: int, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / self.weight.size(1) ** 0.5
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input: torch.tensor, mask: torch.tensor) -> torch.tensor:
        """
        Args:
            input: input feature tensor of shape [batch size, ..., input_size].
            mask: mask tensor of shape [input_size, output_size], i.e., the same
                size as the weight matrix.

        Returns:
            Output tensor of shape [batch size, ..., output_size].
        """
        weight = torch.mul(self.weight, mask)
        output = torch.mm(input, weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class MolecularGraphNeuralNetwork(nn.Module):
    """Molecular Graph Neural Network.

    Paper: Masashi Tsubaki et al. Compound-protein interaction
    prediction with end-to-end learning of neural networks for
    graphs and sequences. Bioinformatics, 2019.

    Args:
        num_fingerprints: total number of fingerprints.
        dim: embedding dimension of the fingerprint vectors.
        layer_hidden: number of hidden layers.
    """

    def __init__(self, num_fingerprints, dim, layer_hidden):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.layer_hidden = layer_hidden
        self.embed_fingerprint = nn.Embedding(num_fingerprints, dim)
        self.W_fingerprint = nn.ModuleList(
            [nn.Linear(dim, dim) for _ in range(layer_hidden)]
        )

    def update(self, matrix, vectors, layer):
        hidden_vectors = torch.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def sum(self, vectors, axis):
        sum_vectors = [torch.sum(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(sum_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, fingerprints, adjacencies, molecular_sizes):
        """
        Args:
            fingerprints: a list of fingerprints
            adjacencies: a list of adjacency matrices
            molecular_sizes: a list of the number of atoms in each molecule
        """
        """MPNN layer (update the fingerprint vectors)."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for layer in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, layer)
            # fingerprint_vectors = F.normalize(hs, 2, 1)  # normalize.
            fingerprint_vectors = hs

        """Molecular vector by sum or mean of the fingerprint vectors."""
        molecular_vectors = self.sum(fingerprint_vectors, molecular_sizes)
        # molecular_vectors = self.mean(fingerprint_vectors, molecular_sizes)

        return molecular_vectors

class SafeDrugLayer(nn.Module):
    """SafeDrug model.

    Paper: Chaoqi Yang et al. SafeDrug: Dual Molecular Graph Encoders for
    Recommending Effective and Safe Drug Combinations. IJCAI 2021.

    This layer is used in the SafeDrug model. But it can also be used as a
    standalone layer. Note that we improve the layer a little bit to make it
    compatible with the package. Original code can be found at 
    https://github.com/ycq091044/SafeDrug/blob/main/src/models.py.

    Args:
        hidden_size: hidden feature size.
        mask_H: the mask matrix H of shape [num_drugs, num_substructures].
        ddi_adj: an adjacency tensor of shape [num_drugs, num_drugs].
        num_fingerprints: total number of different fingerprints.
        molecule_set: a list of molecule tuples (A, B, C) of length num_molecules.
            - A <torch.tensor>: fingerprints of atoms in the molecule
            - B <torch.tensor>: adjacency matrix of the molecule
            - C <int>: molecular_size
        average_projection: a tensor of shape [num_drugs, num_molecules] representing
            the average projection for aggregating multiple molecules of the
            same drug into one vector.
        kp: correcting factor for the proportional signal. Default is 0.5.
        target_ddi: DDI acceptance rate. Default is 0.08.
    """

    def __init__(
        self,
        hidden_size: int,
        mask_H: torch.Tensor,
        ddi_adj: torch.Tensor,
        num_fingerprints: int,
        molecule_set: List[Tuple],
        average_projection: torch.Tensor,
        kp: float = 0.05,
        target_ddi: float = 0.08,
    ):
        super(SafeDrugLayer, self).__init__()
        self.hidden_size = hidden_size
        self.kp = kp
        self.target_ddi = target_ddi

        self.mask_H = nn.Parameter(mask_H, requires_grad=False)
        self.ddi_adj = nn.Parameter(ddi_adj, requires_grad=False)

        # medication space size
        label_size = mask_H.shape[0]

        # local bipartite encoder
        self.bipartite_transform = nn.Linear(hidden_size, mask_H.shape[1])
        # self.bipartite_output = MaskLinear(mask_H.shape[1], label_size, False)
        self.bipartite_output = nn.Linear(mask_H.shape[1], label_size)

        # global MPNN encoder (add fingerprints and adjacency matrix to parameter list)
        mpnn_molecule_set = list(zip(*molecule_set))

        # process three parts of information
        fingerprints = torch.cat(mpnn_molecule_set[0])
        self.fingerprints = nn.Parameter(fingerprints, requires_grad=False)
        adjacencies = self.pad(mpnn_molecule_set[1], 0)
        self.adjacencies = nn.Parameter(adjacencies, requires_grad=False)
        self.molecule_sizes = mpnn_molecule_set[2]
        self.average_projection = nn.Parameter(average_projection, requires_grad=False)

        self.mpnn = MolecularGraphNeuralNetwork(
            num_fingerprints, hidden_size, layer_hidden=2
        )
        self.mpnn_output = nn.Linear(label_size, label_size)
        self.mpnn_layernorm = nn.LayerNorm(label_size)

        self.test = nn.Linear(hidden_size, label_size)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def pad(self, matrices, pad_value):
        """Pads the list of matrices.

        Padding with a pad_value (e.g., 0) for batch processing.
        For example, given a list of matrices [A, B, C], we obtain a new
        matrix [A00, 0B0, 00C], where 0 is the zero (i.e., pad value) matrix.
        """
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        zeros = torch.FloatTensor(np.zeros((M, N)))
        pad_matrices = pad_value + zeros
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i : i + m, j : j + n] = matrix
            i += m
            j += n
        return pad_matrices

    def calculate_loss(
        self, logits: torch.Tensor, y_prob: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        mul_pred_prob = y_prob.T @ y_prob  # (voc_size, voc_size)
        batch_ddi_loss = (
            torch.sum(mul_pred_prob.mul(self.ddi_adj)) / self.ddi_adj.shape[0] ** 2
        )

        y_pred = y_prob.clone().detach().cpu().numpy()
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        y_pred = [np.where(sample == 1)[0] for sample in y_pred]

        cur_ddi_rate = ddi_rate_score(y_pred, self.ddi_adj.cpu().numpy())
        if cur_ddi_rate > self.target_ddi:
            beta = max(0.0, 1 + (self.target_ddi - cur_ddi_rate) / self.kp)
            add_loss, beta = batch_ddi_loss, beta
        else:
            add_loss, beta = 0, 1

        # obtain target, loss, prob, pred
        bce_loss = self.loss_fn(logits, labels)

        #loss = beta * bce_loss + (1 - beta) * add_loss
        loss = bce_loss
        return loss

    def forward(
        self,
        patient_emb: torch.tensor,
        drugs: torch.tensor,
        mask: Optional[torch.tensor] = None,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward propagation.

        Args:
            patient_emb: a tensor of shape [patient, visit, input_size].
            drugs: a multihot tensor of shape [patient, num_labels].
            mask: an optional tensor of shape [patient, visit] where 1
                indicates valid visits and 0 indicates invalid visits.

        Returns:
            loss: a scalar tensor representing the loss.
            y_prob: a tensor of shape [patient, num_labels] representing
                the probability of each drug.
        """
        if mask is None:
            mask = torch.ones_like(patient_emb[:, :, 0])

        query = get_last_visit(patient_emb, mask)  # (batch, dim)

        # MPNN Encoder
        MPNN_emb = self.mpnn(
            self.fingerprints, self.adjacencies, self.molecule_sizes
        )  # (#molecule, hidden_size)
        MPNN_emb = torch.mm(self.average_projection, MPNN_emb)  # (#med, hidden_size)
        MPNN_match = torch.sigmoid(torch.mm(query, MPNN_emb.T))  # (patient, #med)
        MPNN_att = self.mpnn_layernorm(
            MPNN_match + self.mpnn_output(MPNN_match)
        )  # (batch, #med)

        # Bipartite Encoder (use the bipartite encoder only for now)
        bipartite_emb = torch.sigmoid(self.bipartite_transform(query))  # (batch, dim)
        bipartite_att = self.bipartite_output(
            bipartite_emb
        )  # (batch, #med)

        # combine
        logits = bipartite_att * MPNN_att

        # calculate the ddi_loss by PID stragegy and add to final loss
        y_prob = torch.sigmoid(logits)
        loss = self.calculate_loss(logits, y_prob, drugs)

        return loss, y_prob


def discretize_icd_code(code, max_len=5):
    """
    Discretizes the ICD code.
    "123.45" -> ["-1","1", "2", "3", "4", "5"]
    """
    assert len(code) <= max_len, "The length of the code is greater than max_len."
    icd_array = ['-1'] * (max_len+1)

    if code.startswith("E"):
        # if len(code) <= 4:
        #     icd_array = code
        icd_array[:len(code)] = code
    else:
        #starts with v or digit
        # if len(code) <= 3:
        #     icd_array = code
        icd_array[1:len(code)+1] =  code
    return icd_array

def get_digit_embedding_dict(tokens):
    #Multi-Feature Multi-Label Encoder for each digit of icd code
    from sklearn.preprocessing import OneHotEncoder
    icd_feat =[]
    for code in tokens:
        d = discretize_icd_code(code, max_len=7) # icd9 max_len=5, icd10 max_len=7
        icd_feat.append(d)
    encoder = OneHotEncoder(sparse_output = False)
    encoder.fit(icd_feat)
    encodings = encoder.transform(icd_feat)
    enc_dict = {}
    num_features = len(encodings[0])
    for i, code in enumerate(tokens):
        enc_dict[code] = encodings[i]
    return enc_dict, num_features, encoder

# broken up such as E001-E009
# def get_digit_embedding_dict(vocab="ICD9CM"):
#     icd9cm = InnerMap.load(vocab)
#     tokens = list(icd9cm.graph.nodes)
#     tokens = [t.replace(".","") for t in tokens]
#     return discretize_icd_code_dictionary(tokens)


def get_icd_description(code,vocab="ICD9CM"):
    icd9cm = InnerMap.load(vocab)
    #ICD9PROC = InnerMap.load("ICD9PROC")
    return icd9cm.lookup(code)


# Alternative implementation for tree model
class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels,))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x

    def full_forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
        return x


class NeighborSampler(RawNeighborSampler):
    def sample(self, batch):
        batch = torch.tensor(batch)
        row, col, _ = self.adj_t.coo()

        # For each node in `batch`, we sample a direct neighbor (as positive
        # example) and a random node (as negative example):
        pos_batch = random_walk(row, col, batch, walk_length=1,
                                coalesced=False)[:, 1]

        neg_batch = torch.randint(0, self.adj_t.size(1), (batch.numel(), ),
                                dtype=torch.long)

        batch = torch.cat([batch, pos_batch, neg_batch], dim=0)
        return super(NeighborSampler, self).sample(batch)



def get_icd_graph_embedding_dict(vocab="ICD9CM",dim=64, semantic=False,pos_embedding=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if semantic:
        tokenizer = BertTokenizer.from_pretrained("pretrain_model/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",local_files_only=True)
        bert_model = BertModel.from_pretrained("pretrain_model/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",local_files_only=True)
        bert_model.to(device)
        bert_model.eval() 
    icd9cm = InnerMap.load(vocab)

    # A `Data` object is returned
    G = icd9cm.graph
    nx.set_node_attributes(G,{n:{'code':n} for n in G.nodes()})

    data = from_networkx(G)
    
    y = [n[0] for n in data.code]
    # y 转换为unique label
    label_dict = {string: label for label, string in enumerate(set(y))}
    label = [label_dict[string] for string in y]

    data.y = torch.tensor(label)

    if semantic:
        x = []
        with torch.no_grad():
            for text in data['name']:
                inputs = tokenizer(text, return_tensors='pt').to(device)
                last_hidden_state = bert_model(**inputs).last_hidden_state.to('cpu')
                x.append(last_hidden_state[0, 0, :])
        data.x = torch.stack(x).to(device)
    else:
        data.x = torch.randn(data.num_nodes, dim)
        
        # ret = {}
        # for i, code in enumerate(data.code):
        #     code = code.replace(".","")
        #     ret[code] = data.x[i]
        # return ret,None,None


    train_loader = NeighborSampler(data.edge_index, sizes=[10, 10, 10], batch_size=256,#256
                                shuffle=True, num_nodes=data.num_nodes)

    model = SAGE(data.num_node_features, hidden_channels=dim, num_layers=3)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    x, edge_index = data.x.to(device), data.edge_index.to(device)

    def train():
        model.train()

        total_loss = 0
        for batch_size, n_id, adjs in train_loader:
            # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
            adjs = [adj.to(device) for adj in adjs]
            optimizer.zero_grad()

            out = model(x[n_id], adjs)
            out, pos_out, neg_out = out.split(out.size(0) // 3, dim=0)

            pos_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
            neg_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
            loss = -pos_loss - neg_loss
            loss.backward()
            optimizer.step()

            total_loss += float(loss) * out.size(0)

        return total_loss / data.num_nodes


    @torch.no_grad()
    def test():
        model.eval()
        out = model.full_forward(x, edge_index).cpu()

        clf = LogisticRegression()
        clf.fit(out[data.train_mask], data.y[data.train_mask])

        val_acc = clf.score(out[data.val_mask], data.y[data.val_mask])
        test_acc = clf.score(out[data.test_mask], data.y[data.test_mask])

        return val_acc, test_acc


    for epoch in range(1, 51):
        loss = train()
        print("epoch ", epoch, "loss", loss)
        #val_acc, test_acc = test()
        #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, '
        #      f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')
    with torch.no_grad():
        model.eval()
        out = model.full_forward(x, edge_index)#.cpu()
    
    ret = {}
    for i, code in enumerate(data.code):
        code = code.replace(".","")
        ret[code] = out[i]
    #return ret
    return ret,model,data



# 20240513 by liangyuliang
class PositionEmbedding(nn.Module):
    def __init__(self, 
                idx2token, 
                embedding_dim, 
                hierarchical=False):
        super(PositionEmbedding, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.idx2token = idx2token
        self.tokens = list(idx2token.values())
        self.hierarchical = hierarchical
        # remove padding and unknown token
        digit_embedding_dict, digit_embedding_dim, digit_encoder = get_digit_embedding_dict(self.tokens[2:])
        digit_count = int(sum(list(digit_embedding_dict.values())[0])) # 6 in icd9 code
        digit_category = digit_encoder.categories_
        digit_category_count = [len(c) for c in digit_category] #[2, 11, 10, 10, 11, 11]
        # recover padding and unknown token
        self.digit_embedding = [np.zeros(digit_embedding_dim)]*2 + list(digit_embedding_dict.values())
        self.digit_embedding = torch.tensor(self.digit_embedding, dtype= torch.float, device=device)
        self.embedding = nn.Embedding.from_pretrained(self.digit_embedding, freeze=True) # ,padding_idx=0 不设置padding 性能好一点

        # digit embedding -> dense embedding
        #self.pos_embedding = self.embedding
        self.dense = nn.Linear(digit_embedding_dim, embedding_dim)

        if hierarchical:
            # self.hierarchical_embedding = nn.Parameter(torch.randn(digit_embedding_dim, embedding_dim, device=device)) # [digit_embedding_dim, embedding_dim]
            # self.alpha = torch.arange(1, digit_count+1)
            # self.alpha = self.alpha/self.alpha.sum()
            # self.extend_alpha = torch.repeat_interleave(self.alpha, torch.tensor(digit_category_count)).to(device=device)

            # K linear layers
            self.up = nn.ParameterList([nn.Parameter(torch.randn(digit_embedding_dim)) for d in range(digit_count)]) 
            #nn.ModuleList([nn.Parameter(digit_embedding_dim, embedding_dim) for _ in range(digit_count)])
            self.down = nn.Linear(digit_embedding_dim * digit_count, embedding_dim)
        
    def forward(self, x):
        if self.hierarchical:
            x = self.embedding(x)
            # repect input liner layer
            outputs = []
            for i, w in enumerate(self.up):
                out = w * x
                #out = torch.exp(out)
                out = torch.relu(out)
                #out = F.sigmoid(out)
                outputs.append(out)
            outputs = torch.concat(outputs,-1)
            return outputs


        else:
            pos = self.embedding(x)
            pos = self.dense(pos)
            pos = F.relu(pos)
            return pos

# 20240513 by liangyuliang   
class GraphEmbedding(nn.Module):
    def __init__(self, vocab, embedding_dim, idx2token=None, semantic=False, pos_embedding=None):
        super(GraphEmbedding, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.semantic = semantic

        # icd graph embedding
        self.graph_embedding_dict, graph_model, graph_data = get_icd_graph_embedding_dict(
            vocab=vocab, #ICD9CM ICD9PROC
            dim=embedding_dim,
            semantic=semantic,
            pos_embedding=pos_embedding)
        
        # add padding and unknown token
        self.graph_embedding_dict.update(
            {
            '<pad>': torch.zeros(embedding_dim,device=self.device),
            '<unk>': torch.zeros(embedding_dim,device=self.device)
            }
        )

        self.graph_embedding_key = list(self.graph_embedding_dict.keys())
        self.num_graph_embedding = len(self.graph_embedding_key)
        padding_idx = self.graph_embedding_key.index('<pad>')
        self.graph_embedding = torch.stack(list(self.graph_embedding_dict.values())) #[17736+2, dim]
        self.graph_embedding = nn.Embedding.from_pretrained(self.graph_embedding, padding_idx=padding_idx)
        
        # idx2graphidx 
        self.idx2graphidx = []
        if idx2token is None:
            print("idx2token is None")
            return
        
        # transform idx to graph embedding idx
        for idx, code in idx2token.items():
            try:
                graph_idx = self.graph_embedding_key.index(code)
            except:
                print(f"find unknown code: {code} in {vocab}")
                graph_idx = self.graph_embedding_key.index('<unk>')
            self.idx2graphidx.append(graph_idx)

        self.idx2graphidx = torch.tensor(self.idx2graphidx,device=device)
        
        # idx embedding 不改变原来顺序，填充回去。 
        self.embedding = torch.index_select(self.graph_embedding.weight, 0, self.idx2graphidx)
        self.embedding = nn.Embedding.from_pretrained(self.embedding, padding_idx=0)

        self.text_projection = nn.Linear(768, embedding_dim)

    def forward(self, x):
        x = self.embedding(x)
        # if self.semantic:
        #     x = self.text_projection(x)
        return x


# 20240527 by liangyuliang
class HierarchicalEmbeddingModel(nn.Module):
    def __init__(self, vocab= "ICD9CM", embed_dim=128, alpha=0.5, idx2token=None, semantic=False):
        super(HierarchicalEmbeddingModel, self).__init__()


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embed_dim = embed_dim
        self.alpha = alpha
        # layer
        self.fc1 = nn.Linear(embed_dim * 2, embed_dim)
        self.fc2 = nn.Linear(embed_dim * 2, embed_dim)
        self.fc3 = nn.Linear(embed_dim * 2, embed_dim)
        self.fc4 = nn.Linear(embed_dim * 2, embed_dim)
        
        # icd embedding dict
        self.graph_embedding_dict, self.graph_data = self.get_icd_embedding(
            vocab=vocab, #ICD9CM/ICD9PROC
            embed_dim=embed_dim,
            semantic=semantic,)
        
        # add padding and unknown token
        self.graph_embedding_dict.update(
            {
            '<pad>': torch.zeros(embed_dim,device=self.device),
            '<unk>': torch.zeros(embed_dim,device=self.device)
            }
        )
        self.graph_embedding_key = list(self.graph_embedding_dict.keys())
        self.num_graph_embedding = len(self.graph_embedding_key)
        self.padding_idx = self.graph_embedding_key.index('<pad>')
        self.graph_embedding = torch.stack(list(self.graph_embedding_dict.values())) #[17736+2, dim]
        self.graph_embedding = nn.Embedding.from_pretrained(self.graph_embedding, padding_idx=self.padding_idx,freeze=True)


        # idx2graphidx 
        self.idx2graphidx = []
        if idx2token is None:
            print("idx2token is None")
            return
        
        # transform idx to graph embedding idx
        # 将key 去掉小数点，然后转换为graph_emb_key
        graph_emb_key = [k.replace(".","") for k in self.graph_embedding_key]
        for idx, code in idx2token.items():
            try:
                graph_idx = graph_emb_key.index(code)
            except:
                print(f"find unknown code: {code} in {vocab}")
                graph_idx = self.graph_embedding_key.index('<unk>')
            self.idx2graphidx.append(graph_idx)
        
        self.idx2graphidx = torch.tensor(self.idx2graphidx,device=self.device)
        
        # idx embedding 不改变原来顺序，填充回去。 
        selected_weights = torch.index_select(self.graph_embedding.weight, 0, self.idx2graphidx)
        self.embedding = nn.Embedding.from_pretrained(selected_weights, padding_idx=0)
        #self.embedding.weight = self.graph_embedding.weight

        print("HierarchicalEmbedding Done")

    def forward(self, x):
        x = self.embedding(x)
        return x

    """
    def finetune_forward(self, batch):
        
        # Get index
        node_idx = self.idx2graphidx[batch]
        ancestors = [self.graph_data.ancestors[i] if i<len(self.graph_data.ancestors) else '<pad>' for i in node_idx.flatten()]
        parents = [a[0] if len(a) > 0 and a != '<pad>' else '<pad>' for a in ancestors]
        parent_idx = [self.graph_embedding_key.index(p) for p in parents]
        parent_idx = torch.tensor(parent_idx,device=self.device).reshape(node_idx.shape)

        # Get embeddings
        node_embed = self.graph_embedding(torch.tensor(node_idx,device=self.device))
        parent_embed = self.graph_embedding(torch.tensor(parent_idx,device=self.device))


        #neighbor_emb = torch.mean(torch.stack([parent_embed, grand_parent_embed, great_grand_parent_embed]), dim=0)
        neighbor_emb = parent_embed
        
        node_hier_embed = F.relu(self.fc(torch.cat([node_embed, neighbor_emb], dim=-1)))
        return node_hier_embed
    """

    def train_forward(self, node, parent, grand_parent, great_grand_parent, sibling=None, is_test=False):
        
        # Get index
        node_idx = [self.graph_embedding_key.index(n) for n in node]
        parent_idx = [self.graph_embedding_key.index(p) for p in parent]
        grand_parent_idx = [self.graph_embedding_key.index(gp) for gp in grand_parent]
        great_grand_parent_idx = [self.graph_embedding_key.index(ggp) for ggp in great_grand_parent]
        sibling_idx = [self.graph_embedding_key.index(s) for s in sibling]

        # Get embeddings
        node_embed = self.graph_embedding(torch.tensor(node_idx,device=self.device))
        parent_embed = self.graph_embedding(torch.tensor(parent_idx,device=self.device))
        grand_parent_embed = self.graph_embedding(torch.tensor(grand_parent_idx,device=self.device))
        great_grand_parent_embed = self.graph_embedding(torch.tensor(great_grand_parent_idx,device=self.device))
        sibling_embed = self.graph_embedding(torch.tensor(sibling_idx,device=self.device))


        # method 1
        # neighbor_emb = torch.mean(torch.stack([parent_embed,
        #                                        grand_parent_embed, 
        #                                        great_grand_parent_embed
        #                                        ]), dim=0)
        #node_hier_embed = F.relu(self.fc(torch.cat([node_embed, neighbor_emb], dim=1)))
        
        # method 2

        agg = self.fc1(torch.cat([grand_parent_embed, great_grand_parent_embed], dim=1))
        agg = self.fc2(torch.cat([parent_embed, agg], dim=1))
        node_mean = torch.mean(torch.stack([node_embed, sibling_embed]), dim=0)
        node_hier_embed = F.relu(self.fc3(torch.cat([node_mean, agg], dim=1)))

        

        # node_embed = F.relu(self.fc(node_embed))
        # parent_embed = F.relu(self.fc(parent_embed))
        # grand_parent_embed = F.relu(self.fc(grand_parent_embed))
        # great_grand_parent_embed = F.relu(self.fc(great_grand_parent_embed))

        #sibling_embed = F.relu(self.fc(sibling))
        
        # Calculate hierarchical embeddings
        #self.alpha = [0.5, 0.3, 0.1, 0.1]
        #node_hier_embed = self.alpha * node_embed + (1 - self.alpha) * parent_embed
        # node_hier_embed = (self.alpha[0] * node_embed
        #         + self.alpha[1] * parent_embed
        #         + self.alpha[2] * grand_parent_embed
        #         + self.alpha[3] * great_grand_parent_embed
        # )

        ret = {
            'node': node_embed,
            'node_hier': node_hier_embed,
            'parent': parent_embed,
            'grand_parent': grand_parent_embed,
            'great_grand_parent': great_grand_parent_embed,
            'sibling': sibling_embed
        }
        
        return ret
    
    def compute_loss(self, result):

        node_embed = result['node']
        node_hier_embed = result['node_hier']
        parent_embed = result['parent']
        grand_parent_embed = result['grand_parent']
        great_grand_parent_embed = result['great_grand_parent']
        sibling_embed = result['sibling']

        neg_batch = torch.randint(0, node_embed.shape[0], (node_embed.shape[0],),dtype=torch.long)
            
        out = node_hier_embed #node_embed
        pos_out = parent_embed
        neg_out = node_embed[neg_batch]

        pos_parent_loss = F.logsigmoid((out * pos_out).sum(-1)).mean()
        neg_parent_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss_parent = -pos_parent_loss - neg_parent_loss

        pos_grandparent_loss = F.logsigmoid((out * grand_parent_embed).sum(-1)).mean()
        neg_grandparent_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss_grandparent = -pos_grandparent_loss - neg_grandparent_loss

        pos_greatgrandparent_loss = F.logsigmoid((out * great_grand_parent_embed).sum(-1)).mean()
        neg_greatgrandparent_loss = F.logsigmoid(-(out * neg_out).sum(-1)).mean()
        loss_greatgrandparent = -pos_greatgrandparent_loss - neg_greatgrandparent_loss

        sibling_loss = -F.cosine_similarity(node_embed, sibling_embed).mean()

        #loss = loss_parent + loss_grandparent + loss_greatgrandparent + sibling_loss
        loss =  loss_parent

        print(loss.item())
        return loss

        pos_sim_parent = F.cosine_similarity(node_embed, parent_embed)
        # pos_sim_grandparent = F.cosine_similarity(node_hier_embed, grand_parent_embed)
        # pos_sim_greatgrandparent = F.cosine_similarity(node_hier_embed, great_grand_parent_embed)

        neg_sim = F.cosine_similarity(node_hier_embed.unsqueeze(1), torch.randn_like(parent_embed).unsqueeze(0))
        
        pos_loss_parent = -torch.log(torch.exp(pos_sim_parent) / (torch.exp(pos_sim_parent) + torch.exp(neg_sim).sum(dim=1)))
        # pos_loss_grandparent = -torch.log(torch.exp(pos_sim_grandparent) / (torch.exp(pos_sim_grandparent) + torch.exp(neg_sim).sum(dim=1)))
        # pos_loss_greatgrandparent = -torch.log(torch.exp(pos_sim_greatgrandparent) / (torch.exp(pos_sim_greatgrandparent) + torch.exp(neg_sim).sum(dim=1)))
        print(pos_loss_parent.mean().item(), 
            #   pos_loss_grandparent.mean().item(), 
            #   pos_loss_greatgrandparent.mean().item()
              )
        loss = pos_loss_parent.mean() #+ pos_loss_grandparent.mean() # + pos_loss_greatgrandparent.mean()
    
        return loss


    # get node ancestors and siblings 
    # 20240528 by liangyuliang
    def get_icd_embedding(self, vocab, embed_dim=128, semantic=False):
        # prepare data
        icd9cm = InnerMap.load(vocab)
        # nxworks 
        G = icd9cm.graph 
        nx.set_node_attributes(G,{n:{'code':n} for n in G.nodes()})
        for node in G.nodes():
            # get ordered ancestors
            ancestors = nx.ancestors(G, node)
            ancestors_path_lenght = {anc:nx.shortest_path_length(G, source=anc, target=node) for anc in ancestors}
            sorted_ancestors_path_lenght = dict(sorted(ancestors_path_lenght.items(), key=lambda item: item[1]))
            sorted_ancestors = list(sorted_ancestors_path_lenght.keys())
            G.nodes[node]['ancestors'] = sorted_ancestors
            G.nodes[node]['ancestors_path_lenght'] = sorted_ancestors_path_lenght

            # get siblings
            if sorted_ancestors:
                parent = sorted_ancestors[0]
                siblings = list(nx.descendants(G, parent))
                siblings.remove(node)
                G.nodes[node]['siblings'] = siblings
            else:
                G.nodes[node]['siblings'] = []

        # to pyg
        data = from_networkx(G)
        
        if semantic:
            tokenizer = BertTokenizer.from_pretrained(
                "pretrain_model/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                local_files_only=True)
            bert_model = BertModel.from_pretrained(
                "pretrain_model/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                local_files_only=True)
            bert_model.to(self.device)
            bert_model.eval() 

            x = []
            with torch.no_grad():
                for text in data['name']:
                    inputs = tokenizer(text, return_tensors='pt').to(self.device)
                    last_hidden_state = bert_model(**inputs).last_hidden_state.to('cpu')
                    x.append(last_hidden_state[0, 0, :])
            data.x = torch.stack(x).to(self.device)
        else:
            data.x = torch.randn(data.num_nodes, self.embed_dim, device=self.device)
             
        # prepare emb dict
        emb_dict = {}
        for i, code in enumerate(data.code):
            #code = code.replace(".","")
            emb_dict[code] = data.x[i]
        return emb_dict, data

def Train_HierarchicalEmbeddingModel(model, graph_data, epoch=10, batch_size=32):
    all_code = graph_data.code
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    model.to(model.device)
    #graph_embedding_dict
    
    train_loader = DataLoader(all_code, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(all_code, batch_size=1, shuffle=False)
    #next(iter(train_loader))

    # train 
    for epoch in range(1, epoch):
        for batch in train_loader:
            # batch data
            node_idx = [graph_data.code.index(c) for c in batch]
            ancestors = [graph_data.ancestors[i] for i in node_idx]
            siblings = [graph_data.siblings[i] for i in node_idx]
            
            node = batch
            parent = []
            grand_parent = []
            great_grand_parent = []
            one_siblings = []
            
            for i, b in enumerate(batch):
                # ancestors
                if len(ancestors[i]) == 0:
                    parent.append('<pad>')
                    grand_parent.append('<pad>')
                    great_grand_parent.append('<pad>')
                elif len(ancestors[i]) == 1:
                    parent.append(ancestors[i][0])
                    grand_parent.append('<pad>')
                    great_grand_parent.append('<pad>')
                elif len(ancestors[i]) == 2:
                    parent.append(ancestors[i][0])
                    grand_parent.append(ancestors[i][1])
                    great_grand_parent.append('<pad>')
                else:
                    parent.append(ancestors[i][0])
                    grand_parent.append(ancestors[i][1])
                    great_grand_parent.append(ancestors[i][2])

            # siblings
            for i,s in enumerate(siblings):
                if len(s) == 0:
                    siblings[i] = ['<pad>']
                    one_siblings.append('<pad>')
                else:
                    np.random.shuffle(s)
                    one_siblings.append(s[0])
                    
           # emb_table = model.graph_embedding_dict

            optimizer.zero_grad()
            result = model.train_forward(node, parent, grand_parent, great_grand_parent, one_siblings)
            loss = model.compute_loss(result)
            loss.backward()
            optimizer.step()
        print(f"epoch {epoch} loss: {loss}")
    
    # inference
    emb_dict = {}
    model.eval()
    for batch in test_loader:
        node_idx = [graph_data.code.index(c) for c in batch]
        ancestors = [graph_data.ancestors[i] for i in node_idx]
        siblings = [graph_data.siblings[i] for i in node_idx]
        
        node = batch
        parent = []
        grand_parent = []
        great_grand_parent = []
        one_siblings = []
        all_sibling = []

        for i, b in enumerate(batch):
            if len(ancestors[i]) == 0:
                parent.append('<pad>')
                grand_parent.append('<pad>')
                great_grand_parent.append('<pad>')
            elif len(ancestors[i]) == 1:
                parent.append(ancestors[i][0])
                grand_parent.append('<pad>')
                great_grand_parent.append('<pad>')
            elif len(ancestors[i]) == 2:
                parent.append(ancestors[i][0])
                grand_parent.append(ancestors[i][1])
                great_grand_parent.append('<pad>')
            else:
                parent.append(ancestors[i][0])
                grand_parent.append(ancestors[i][1])
                great_grand_parent.append(ancestors[i][2])
        # siblings
        for i,s in enumerate(siblings):
            if len(s) == 0:
                siblings[i] = ['<pad>']
                one_siblings.append('<pad>')
            else:
                np.random.shuffle(s)
                one_siblings.append(s[0])
            
        result = model.train_forward(node, parent, grand_parent, great_grand_parent, one_siblings, is_test=True)
        emb_dict.update({n: e for n, e in zip(node, result['node_hier'])})

    selected_weights = torch.full_like(model.graph_embedding.weight, 0.0)
    # pretrained embedding as embedding layer
    for code, emb in emb_dict.items():
        selected_weights[model.graph_embedding_key.index(code)] = emb 
    model.embedding = nn.Embedding.from_pretrained(selected_weights, padding_idx=0)
    
    return emb_dict
    




# 使用icd数字特征->embedding
class SafeDrug_ICD_Unsup_Pos(BaseModel):
    """SafeDrug model.

    Paper: Chaoqi Yang et al. SafeDrug: Dual Molecular Graph Encoders for
    Recommending Effective and Safe Drug Combinations. IJCAI 2021.

    Note:
        This model is only for medication prediction which takes conditions
        and procedures as feature_keys, and drugs as label_key. It only operates
        on the visit level.

    Note:
        This model only accepts ATC level 3 as medication codes.

    Args:
        dataset: the dataset to train the model. It is used to query certain
            information such as the set of all tokens.
        embedding_dim: the embedding dimension. Default is 128.
        hidden_dim: the hidden dimension. Default is 128.
        num_layers: the number of layers used in RNN. Default is 1.
        dropout: the dropout rate. Default is 0.5.
        **kwargs: other parameters for the SafeDrug layer.
    """

    def __init__(
        self,
        dataset: SampleEHRDataset,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.5,
        **kwargs,
    ):
        super(SafeDrug_ICD_Unsup_Pos, self).__init__(
            dataset=dataset,
            feature_keys=["conditions", "procedures"],
            label_key="drugs",
            mode="multilabel",
        )
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        

        self.feat_tokenizers = self.get_feature_tokenizers()
        self.label_tokenizer = self.get_label_tokenizer()
        self.embeddings = self.get_embedding_layers(self.feat_tokenizers, embedding_dim)

        # --------------------------------------ours--------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        """
        # icd graph embedding

        self.cond_graph_embedding_dict, _, self.cond_graph_data = get_icd_graph_embedding_dict(vocab="ICD9CM",dim=embedding_dim//1)
        self.proc_graph_embedding_dict, _, self.proc_graph_data = get_icd_graph_embedding_dict(vocab="ICD9PROC",dim=embedding_dim//1)
        self.cond_graph_embedding_dict.update(
            {
            '<pad>': torch.zeros(embedding_dim//1,device=self.device),
            '<unk>': torch.zeros(embedding_dim//1,device=self.device)
            }
        )
        self.proc_graph_embedding_dict.update(
            {
            '<pad>': torch.zeros(embedding_dim//1,device=self.device),
            '<unk>': torch.zeros(embedding_dim//1,device=self.device)
            }
        )

        self.cond_graph_embedding = nn.Parameter(torch.stack(list(self.cond_graph_embedding_dict.values()))) #[17736+2, dim]
        self.proc_graph_embedding = nn.Parameter(torch.stack(list(self.proc_graph_embedding_dict.values()))) 
        self.cond_graph_embedding_key = list(self.cond_graph_embedding_dict.keys())
        self.proc_graph_embedding_key = list(self.proc_graph_embedding_dict.keys())

        self.cond_idx2graphidx = []
        self.proc_idx2graphidx = []

        #prepare index for graph embedding
        for idx,code in self.feat_tokenizers['conditions'].vocabulary.idx2token.items():
            graph_idx = self.cond_graph_embedding_key.index(code)
            self.cond_idx2graphidx.append(graph_idx)

        for idx,code in self.feat_tokenizers['procedures'].vocabulary.idx2token.items():
            try:
                graph_idx = self.proc_graph_embedding_key.index(code)
            except:
                print(f"find unknown code: {code}")
                graph_idx = self.proc_graph_embedding_key.index('<unk>')
            self.proc_idx2graphidx.append(graph_idx)
        
        self.cond_idx2graphidx = torch.tensor(self.cond_idx2graphidx,device=device)
        self.proc_idx2graphidx = torch.tensor(self.proc_idx2graphidx,device=device)
        """


        # discrete icd embedding
        self.cond_pos_embedding = PositionEmbedding(
            self.feat_tokenizers['conditions'].vocabulary.idx2token, embedding_dim, hierarchical=False
            )
        self.proc_pos_embedding = PositionEmbedding(
            self.feat_tokenizers['procedures'].vocabulary.idx2token, embedding_dim, hierarchical=False
            )
        
        # graph embedding
        self.cond_graph_embedding = GraphEmbedding("ICD9CM", 
                                                   embedding_dim, 
                                                   self.feat_tokenizers['conditions'].vocabulary.idx2token, 
                                                   semantic=False,
                                                   pos_embedding=self.cond_pos_embedding)
        self.proc_graph_embedding = GraphEmbedding("ICD9PROC",
                                                    embedding_dim,
                                                    self.feat_tokenizers['procedures'].vocabulary.idx2token,
                                                    semantic=False,
                                                    pos_embedding=self.proc_pos_embedding)

        # hierarchical embedding
        # self.cond_graph_embedding = HierarchicalEmbeddingModel(vocab= "ICD9CM", embed_dim=128, alpha=0.5, 
        #                            idx2token=self.feat_tokenizers['conditions'].vocabulary.idx2token, 
        #                            semantic=False)
        # self.proc_graph_embedding = HierarchicalEmbeddingModel(vocab= "ICD9PROC", embed_dim=128, alpha=0.5,
        #                             idx2token=self.feat_tokenizers['procedures'].vocabulary.idx2token,
        #                             semantic=False)
        # Train_HierarchicalEmbeddingModel(self.cond_graph_embedding, self.cond_graph_embedding.graph_data, epoch=10, batch_size=32)
        # Train_HierarchicalEmbeddingModel(self.proc_graph_embedding, self.proc_graph_embedding.graph_data, epoch=10, batch_size=32)
        
        # -------------------------------------- end ours --------------------------------------

        

        # drug space size
        self.label_size = self.label_tokenizer.get_vocabulary_size()

        self.all_smiles_list = self.generate_smiles_list()
        mask_H = self.generate_mask_H()
        (
            molecule_set,
            num_fingerprints,
            average_projection,
        ) = self.generate_molecule_info()
        ddi_adj = self.generate_ddi_adj()

        self.cond_rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.proc_rnn = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.query = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # validate kwargs for GAMENet layer
        if "hidden_size" in kwargs:
            raise ValueError("hidden_size is determined by hidden_dim")
        if "mask_H" in kwargs:
            raise ValueError("mask_H is determined by the dataset")
        if "ddi_adj" in kwargs:
            raise ValueError("ddi_adj is determined by the dataset")
        if "num_fingerprints" in kwargs:
            raise ValueError("num_fingerprints is determined by the dataset")
        if "molecule_set" in kwargs:
            raise ValueError("molecule_set is determined by the dataset")
        if "average_projection" in kwargs:
            raise ValueError("average_projection is determined by the dataset")
        self.safedrug = SafeDrugLayer(
            hidden_size=hidden_dim,
            mask_H=mask_H,
            ddi_adj=ddi_adj,
            num_fingerprints=num_fingerprints,
            molecule_set=molecule_set,
            average_projection=average_projection,
            **kwargs,
        )
        
        # save ddi adj
        ddi_adj = self.generate_ddi_adj()
        np.save(os.path.join(CACHE_PATH, "ddi_adj.npy"), ddi_adj.numpy())

    def generate_ddi_adj(self) -> torch.tensor:
        """Generates the DDI graph adjacency matrix."""
        atc = ATC()
        ddi = atc.get_ddi(gamenet_ddi=True)
        label_size = self.label_tokenizer.get_vocabulary_size()
        vocab_to_index = self.label_tokenizer.vocabulary
        ddi_adj = np.zeros((label_size, label_size))
        ddi_atc3 = [
            [ATC.convert(l[0], level=3), ATC.convert(l[1], level=3)] for l in ddi
        ]
        for atc_i, atc_j in ddi_atc3:
            if atc_i in vocab_to_index and atc_j in vocab_to_index:
                ddi_adj[vocab_to_index(atc_i), vocab_to_index(atc_j)] = 1
                ddi_adj[vocab_to_index(atc_j), vocab_to_index(atc_i)] = 1
        ddi_adj = torch.FloatTensor(ddi_adj)
        return ddi_adj

    def generate_smiles_list(self) -> List[List[str]]:
        """Generates the list of SMILES strings."""
        atc3_to_smiles = {}
        atc = ATC()
        for code in atc.graph.nodes:
            if len(code) != 7:
                continue
            code_atc3 = ATC.convert(code, level=3)
            smiles = atc.graph.nodes[code]["smiles"]
            if smiles != smiles:
                continue
            atc3_to_smiles[code_atc3] = atc3_to_smiles.get(code_atc3, []) + [smiles]
        # just take first one for computational efficiency
        atc3_to_smiles = {k: v[:1] for k, v in atc3_to_smiles.items()}
        all_smiles_list = [[] for _ in range(self.label_size)]
        vocab_to_index = self.label_tokenizer.vocabulary
        for atc3, smiles_list in atc3_to_smiles.items():
            if atc3 in vocab_to_index:
                index = vocab_to_index(atc3)
                all_smiles_list[index] += smiles_list
        return all_smiles_list

    def generate_mask_H(self) -> torch.tensor:
        """Generates the molecular segmentation mask H."""
        all_substructures_list = [[] for _ in range(self.label_size)]
        for index, smiles_list in enumerate(self.all_smiles_list):
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                substructures = BRICS.BRICSDecompose(mol)
                all_substructures_list[index] += substructures
        # all segment set
        substructures_set = list(set(sum(all_substructures_list, [])))
        # mask_H
        mask_H = np.zeros((self.label_size, len(substructures_set)))
        for index, substructures in enumerate(all_substructures_list):
            for s in substructures:
                mask_H[index, substructures_set.index(s)] = 1
        mask_H = torch.FloatTensor(mask_H)
        return mask_H

    def generate_molecule_info(self, radius: int = 1):
        """Generates the molecule information."""

        def create_atoms(mol, atom2idx):
            """Transform the atom types in a molecule (e.g., H, C, and O)
            into the indices (e.g., H=0, C=1, and O=2). Note that each atom
            index considers the aromaticity.
            """
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            for a in mol.GetAromaticAtoms():
                i = a.GetIdx()
                atoms[i] = (atoms[i], "aromatic")
            atoms = [atom2idx[a] for a in atoms]
            return np.array(atoms)

        def create_ijbonddict(mol, bond2idx):
            """Create a dictionary, in which each key is a node ID
            and each value is the tuples of its neighboring node
            and chemical bond (e.g., single and double) IDs.
            """
            i_jbond_dict = defaultdict(lambda: [])
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                bond = bond2idx[str(b.GetBondType())]
                i_jbond_dict[i].append((j, bond))
                i_jbond_dict[j].append((i, bond))
            return i_jbond_dict

        def extract_fingerprints(r, atoms, i_jbond_dict, fingerprint2idx, edge2idx):
            """Extract the fingerprints from a molecular graph
            based on Weisfeiler-Lehman algorithm.
            """
            nodes = [fingerprint2idx[a] for a in atoms]
            i_jedge_dict = i_jbond_dict

            for _ in range(r):

                """Update each node ID considering its neighboring nodes and edges.
                The updated node IDs are the fingerprint IDs.
                """
                nodes_ = deepcopy(nodes)
                for i, j_edge in i_jedge_dict.items():
                    neighbors = [(nodes[j], edge) for j, edge in j_edge]
                    fingerprint = (nodes[i], tuple(sorted(neighbors)))
                    nodes_[i] = fingerprint2idx[fingerprint]

                """Also update each edge ID considering
                its two nodes on both sides.
                """
                i_jedge_dict_ = defaultdict(list)
                for i, j_edge in i_jedge_dict.items():
                    for j, edge in j_edge:
                        both_side = tuple(sorted((nodes[i], nodes[j])))
                        edge = edge2idx[(both_side, edge)]
                        i_jedge_dict_[i].append((j, edge))

                nodes = deepcopy(nodes_)
                i_jedge_dict = deepcopy(i_jedge_dict_)
                del nodes_, i_jedge_dict_

            return np.array(nodes)

        atom2idx = defaultdict(lambda: len(atom2idx))
        bond2idx = defaultdict(lambda: len(bond2idx))
        fingerprint2idx = defaultdict(lambda: len(fingerprint2idx))
        edge2idx = defaultdict(lambda: len(edge2idx))
        molecule_set, average_index = [], []

        for smiles_list in self.all_smiles_list:
            """Create each data with the above defined functions."""
            counter = 0  # counter how many drugs are under that ATC-3
            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    continue
                mol = Chem.AddHs(mol)
                atoms = create_atoms(mol, atom2idx)
                molecular_size = len(atoms)
                i_jbond_dict = create_ijbonddict(mol, bond2idx)
                fingerprints = extract_fingerprints(
                    radius, atoms, i_jbond_dict, fingerprint2idx, edge2idx
                )
                adjacency = Chem.GetAdjacencyMatrix(mol)
                """Transform the above each data of numpy to pytorch tensor."""
                
                fingerprints = torch.LongTensor(fingerprints)
                adjacency = torch.FloatTensor(adjacency)
                molecule_set.append((fingerprints, adjacency, molecular_size))
                counter += 1
            average_index.append(counter)

        num_fingerprints = len(fingerprint2idx)
        # transform into projection matrix
        n_col = sum(average_index)
        n_row = len(average_index)
        average_projection = np.zeros((n_row, n_col))
        col_counter = 0
        for i, item in enumerate(average_index):
            if item > 0:
                average_projection[i, col_counter : col_counter + item] = 1 / item
            col_counter += item
        average_projection = torch.FloatTensor(average_projection)
        return molecule_set, num_fingerprints, average_projection

    def forward(
        self,
        conditions: List[List[List[str]]],
        procedures: List[List[List[str]]],
        drugs: List[List[str]],
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward propagation.

        Args:
            conditions: a nested list in three levels [patient, visit, condition].
            procedures: a nested list in three levels [patient, visit, procedure].
            drugs: a nested list in two levels [patient, drug].

        Returns:
            A dictionary with the following keys:
                loss: a scalar tensor representing the loss.
                y_prob: a tensor of shape [patient, visit, num_labels] representing
                    the probability of each drug.
                y_true: a tensor of shape [patient, visit, num_labels] representing
                    the ground truth of each drug.
        """


        # tokenizing and padding 
        conditions = self.feat_tokenizers["conditions"].batch_encode_3d(conditions)
        # (patient, visit, code)
        conditions = torch.tensor(conditions, dtype=torch.long, device=self.device)

        #conditions_random = self.embeddings["conditions"](conditions)

        # obtain position embedding
        conditions_pos = self.cond_pos_embedding(conditions)
        # obtain content embedding
        conditions_cont = self.cond_graph_embedding(conditions)
 
        conditions = conditions_cont # + conditions_pos 
        #conditions = conditions_random + conditions_pos 

        conditions = torch.sum(conditions, dim=2)
        #conditions = F.layer_norm(conditions, normalized_shape=(conditions.size(-1),))

        # 先sum 后concat 好一些
        #conditions = torch.sum(conditions_cont, dim=2)
        #conditions = torch.concat([conditions, conditions],dim=-1)

        # (batch, visit, hidden_size)
        conditions, _ = self.cond_rnn(conditions)

        #--------------------------------------------------------------------------------#

        procedures = self.feat_tokenizers["procedures"].batch_encode_3d(procedures)
        # (patient, visit, code)
        procedures = torch.tensor(procedures, dtype=torch.long, device=self.device)

        procedures_random = self.embeddings["procedures"](procedures)

        # obtain position embedding
        procedures_pos = self.proc_pos_embedding(procedures)
        # obtain content embedding
        procedures_cont = self.proc_graph_embedding(procedures)

        procedures = procedures_cont #+  procedures_pos 
        #procedures = procedures_random + procedures_pos
        #procedures =  procedures_pos
        procedures = torch.sum(procedures, dim=2)

        #procedures = F.layer_norm(procedures, normalized_shape=(procedures.size(-1),))

        #procedures = torch.sum(procedures_cont, dim=2)
        #procedures = torch.concat([procedures, procedures],dim=-1)

        #procedures = torch.sum(procedures, dim=2)
        # (batch, visit, hidden_size)
        procedures, _ = self.proc_rnn(procedures)

        # (batch, visit, 2 * hidden_size)
        patient_emb = torch.cat([conditions, procedures], dim=-1)
        # (batch, visit, hidden_size)
        patient_emb = self.query(patient_emb)

        # get mask
        mask = torch.sum(conditions, dim=2) != 0

        drugs = self.prepare_labels(drugs, self.label_tokenizer)
        
        loss, y_prob = self.safedrug(patient_emb, drugs, mask)


        return {
            #"loss": loss if self.step%2 == 0 else (cond_total_loss + proc_total_loss),
            "loss": loss,
            "y_prob": y_prob,
            "y_true": drugs,
        }




if __name__ == "__main__":

    print(get_icd_description("428.0"))
    #print(get_icd_description_embedding_dict(["428.0","6822"]))
    #ret = get_icd_graph_embedding_dict(vocab="ICD9CM",dim=64)
    pass