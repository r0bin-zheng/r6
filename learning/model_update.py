from learning.utils import *
from learning.prediction import nn_predict_dom
from learning.model_init import init_kpls_model, init_kriging_model, init_rbf_model
import math

# This file implements the procedure to update the Pareto-Net or Theta-Net


def update_dom_nn_classifier(net, archive, rel_map, dom, device, max_window_size,
                             max_adjust_epochs=20, batch_size=32, lr=0.001,
                             acc_thr=0.9, weight_decay=0.00001, timer=None):
    
    timer.start(desc="更新Pareto-Net：初始化数据")

    n = len(archive)
    start = get_start_pos(n, max_window_size)

    new_data = prepare_new_dom_data(archive, rel_map, dom, n - 1, start=start, data_kind='tensor', device=device)
    labels, _ = nn_predict_dom(new_data[:, :-1], net)

    timer.next(desc="更新Pareto-Net：计算准确度")

    # 注意，当该类别的样本数量为0时，准确度也为1
    acc0, acc1, acc2 = get_accuracy(new_data[:, -1], labels)
    min_acc = min(acc0, acc1, acc2)

    print("Estimated accuracy for each class: ", acc0, acc1, acc2)

    # 当准确度大于阈值（默认0.9）时，不需要更新
    if min_acc >= acc_thr:
        return
    
    timer.next(desc="更新Pareto-Net：准备训练数据")

    data = prepare_dom_data(archive, rel_map, dom, start=start, data_kind='tensor', device=device)

    weight = compute_class_weight(data[:, -1])
    if weight is None:
        return

    weight = torch.tensor(weight, device=device).float()
    criterion = nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    adjust_epochs = max_adjust_epochs * ((acc_thr - min_acc) / acc_thr)
    adjust_epochs = math.ceil(adjust_epochs)

    timer.next(desc="更新Pareto-Net：训练")

    train_nn(data, load_batched_dom_data, net, criterion, optimizer, batch_size, adjust_epochs)

    timer.end()


def update_kriging_model(archive):
    return init_kriging_model(archive)







