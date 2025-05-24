import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import random_walk
from sklearn.linear_model import LogisticRegression

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import NeighborSampler as RawNeighborSampler

import umap
import matplotlib.pyplot as plt
import seaborn as sns

# dataset = 'Cora'
# path = './data'
# dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())
# data = dataset[0]

from pyhealth.medcode import InnerMap
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx


from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import BertTokenizer, BertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("pretrain_model/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",local_files_only=True)
model = BertModel.from_pretrained("pretrain_model/models--microsoft--BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",local_files_only=True)
model.to(device)
model.eval() 
# inputs = tokenizer("hello world", return_tensors='pt', max_length=512, truncation=True, padding=True)['input_ids']
# last_hidden_state = model(inputs).last_hidden_state # [1, squence, hidden_dim]
# cls_token = last_hidden_state[:, 0, :]


icd9cm = InnerMap.load("ICD9CM")

# A `Data` object is returned
G = icd9cm.graph
nx.set_node_attributes(G,{n:{'code':n} for n in G.nodes()})

data = from_networkx(G)
#data.x = torch.randn(data.num_nodes, 64)
x = []
with torch.no_grad():
    for text in data['name']:
        inputs = tokenizer(text, return_tensors='pt').to(device)
        last_hidden_state = model(**inputs).last_hidden_state.to('cpu')
        x.append(last_hidden_state[0, 0, :])
data.x = torch.stack(x)


y = [n[0] for n in data.code]
# y 转换为unique label

label_dict = {string: label for label, string in enumerate(set(y))}
label = [label_dict[string] for string in y]

data.y = torch.tensor(label)


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


train_loader = NeighborSampler(data.edge_index, sizes=[25, 10], batch_size=256,
                               shuffle=True, num_nodes=data.num_nodes)


class SAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(SAGE, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = in_channels if i == 0 else hidden_channels
            self.convs.append(SAGEConv(in_channels, hidden_channels))

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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SAGE(data.num_node_features, hidden_channels=64, num_layers=2)
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
    out = model.full_forward(x, edge_index).cpu()
palette = {}

for n, y in enumerate(set(data.y.numpy())):
    palette[y] = f'C{n}'
embd = umap.UMAP().fit_transform(out.cpu().numpy())
plt.figure(figsize=(10, 10))
sns.scatterplot(x=embd.T[0], y=embd.T[1], hue=data.y.cpu().numpy(), palette=palette)
plt.legend(bbox_to_anchor=(1,1), loc='upper left')
plt.savefig("umap_embd_sage.png", dpi=120)

embd_x = umap.UMAP().fit_transform(data.x.numpy())
plt.figure(figsize=(10, 10))
sns.scatterplot(x=embd_x.T[0], y=embd_x.T[1], hue=data.y.cpu().numpy(), palette=palette)
plt.legend(bbox_to_anchor=(1,1), loc='upper left')
plt.savefig("umap_embd.png", dpi=120)

print("Done")