import torch
from learning.utils import prepare_dom_data, compute_class_weight, load_batched_dom_data, train_nn, prepare_data
from learning.model import NeuralNet
from smt.surrogate_models import KRG, RBF, KPLS
import torch.nn as nn
import numpy as np

# This file implements the procedure to initiate the Pareto-Net or Theta-Net


def init_dom_nn_classifier(archive, rel_map, dom, device, input_size,
                           hidden_size, num_hidden_layers, epochs, batch_size=32,
                           activation='relu', lr=0.001, weight_decay=0.00001):
    
    # 准备帕累托优势关系数据
    data = prepare_dom_data(archive, rel_map, dom, data_kind='tensor', device=device)

    weight = compute_class_weight(data[:, -1])

    if weight is None:
        return None

    net = NeuralNet(input_size, hidden_size, num_hidden_layers,
                    activation=activation).to(device)

    weight = torch.tensor(weight, device=device).float()
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    train_nn(data, load_batched_dom_data, net, criterion, optimizer, batch_size, epochs)

    return net

def init_kriging_model(archive):

    # data = prepare_data(archive)
    # print(data)

    X = np.array([ind.normalized_var for ind in archive])
    y = np.array([ind.fitness.values for ind in archive])

    # print("X: ", X)
    # print("y: ", y)
    
    model = KRG(theta0=[1e-2] * len(X[0]), print_global=False)
    model.set_training_values(X, y)
    model.train()

    return model


def init_rbf_model(archive):
    return None

def init_kpls_model(archive):
    return None
