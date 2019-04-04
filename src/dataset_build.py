import torch
from torch_geometric.data import Data

#
# edge_index = torch.tensor([[0, 1],
#                            [2, 3],
#                            [4, 5],
#                            [6,7]
#                     ], dtype=torch.long)
# data = Data(edge_index=edge_index.t().contiguous())
from torch_geometric.data import DataLoader
# loader = DataLoader(data, batch_size=2, shuffle=True)
# for i in loader:
#     b = 1

import torch
from torch_geometric.data import InMemoryDataset


class MyOwnDatasetTest(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDatasetTest, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']
        # pass

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        edge_index = torch.tensor([[0, 1],
                                   [1,2]
                                   ], dtype=torch.long)
        y1 = torch.tensor([0], dtype=torch.long)
        y2 = torch.tensor([1], dtype=torch.long)

        data1 = Data(edge_index=edge_index.t().contiguous(), y=y1)
        data2 = Data(edge_index=edge_index.t().contiguous(), y=y2)

        data_list = [data1, data2, data1, data2]

        # if self.pre_filter is not None:
        #     data_list [data for data in data_list if self.pre_filter(data)]
        #
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
if __name__ == '__main__':
    dataset = MyOwnDatasetTest(root="/tmp/MyOwnDatasetTest")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    for i in loader:
        b = 1
    # edge_index = torch.tensor([[0, 1],
    #                            ], dtype=torch.long)
    # y = torch.tensor([0], dtype=torch.long)
    # data1 = Data(edge_index=edge_index.t().contiguous(), y=y)
    print("end")