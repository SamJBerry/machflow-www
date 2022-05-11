import meshio
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.nn as gnn
import torch_geometric.transforms as T
from torch_geometric.data import Data


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [
                gnn.GENConv(4, 8, learn_p=True, learn_t=True),
                gnn.GENConv(12, 16, learn_p=True, learn_t=True),
                gnn.GENConv(20, 32, learn_p=True, learn_t=True),
                gnn.GENConv(36, 64, learn_p=True, learn_t=True),
                gnn.GENConv(68, 32, learn_p=True, learn_t=True),
                gnn.GENConv(36, 16, learn_p=True, learn_t=True),
                gnn.GENConv(20, 8, learn_p=True, learn_t=True),
            ]
        )
        self.final_conv = gnn.SAGEConv(12, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.convs:
            x = conv(x, edge_index)
            x = torch.cat((x, data.x), dim=1)
            x = F.dropout(x, p=0.2)
        x = self.final_conv(x, edge_index)

        return x


def load_stl(stl_path, aoa=0):
    """

    :param stl_path:
    :param aoa:
    :return: meshio mesh, full data tensor
    """
    mesh = meshio.read(stl_path)
    vertexes = torch.from_numpy(mesh.points).to(torch.float)
    faces = torch.from_numpy(mesh.cells_dict['triangle']).t().contiguous()

    data = Data(pos=vertexes, face=faces)

    # Generate edges
    trans = T.Compose([T.FaceToEdge()])
    trans(data)

    aoa_tensor = torch.tensor(np.array([aoa] * data.pos.shape[0]))[:, None]

    data.x = torch.cat((data.pos, aoa_tensor), 1).float()
    return data


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN().to(device)
    model.load_state_dict(torch.load("model.pt", map_location=device))
    return model


def predict(model, input_data):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.autocast(device_type=device):
        prediction = model(input_data)
    return prediction
