from pyhealth.medcode import InnerMap
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_networkx



if __name__ == "__main__":
    icd9cm = InnerMap.load("ICD9CM")
    print(icd9cm.stat())
    print("428.0" in icd9cm)
    print(icd9cm.lookup("4280"))
    print(icd9cm.get_ancestors("428.0"))
    print(icd9cm.get_descendants("428.0"))


    # A `Data` object is returned
    G = icd9cm.graph
    nx.set_node_attributes(G,{n:{'code':n} for n in G.nodes()})

    data = from_networkx(G)

    pass