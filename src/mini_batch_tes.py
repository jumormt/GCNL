from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader


dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    a = batch
    b = a.num_graphs