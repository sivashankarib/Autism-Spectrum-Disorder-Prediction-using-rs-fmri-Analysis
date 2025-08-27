import torch
import torch.nn.functional as F


class GCNLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = self.linear(x)
        x = torch.matmul(adj, x)
        x = F.relu(x)
        return x


class GCN(torch.nn.Module):
    def __init__(self, n_features, n_classes, hidden_layers):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNLayer(n_features, hidden_layers[0]))
        for i in range(1, len(hidden_layers)):
            self.layers.append(GCNLayer(hidden_layers[i - 1], hidden_layers[i]))
        self.layers.append(GCNLayer(hidden_layers[-1], n_classes))

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return F.softmax(x, dim=1)


''' This implementation consists of two classes: GCNLayer and GCN.

The GCNLayer class defines a single GCN layer, which takes as input a tensor of node features and an adjacency matrix. It applies a linear transformation to the node features, multiplies the result by the adjacency matrix, and applies the ReLU activation function.

The GCN class defines the entire GCN model, which consists of a stack of GCN layers. The constructor takes as input the number of input features, the number of output classes, and a list of hidden layer sizes. The forward method applies each GCN layer to the input tensor in turn, and then applies a softmax function to the output to obtain class probabilities.'''
