import torch
import torch.nn as nn
import numpy as np
import itertools

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def conf_matrix_part(model, test):
    # confusion matrix for part
    conf_matrix = np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]])

    permutations = list(itertools.permutations([0, 1, 2, 3, 4], 2))
    permutations.extend([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)])

    model.eval()
    with torch.no_grad():
        for x, y in test:
            # loss on valid set
            raw_predict = model(x.to(device))
            probs = nn.Softmax(dim=1)(raw_predict)

            predict_labels = [torch.argmax(label).item() for label in probs]
            true_labels = y.reshape(-1).tolist()

            # conf_matrix filling
            for true_label, predict_label in zip(true_labels, predict_labels):
                for idx_conf_matrix_true, idx_conf_matrix_predict in permutations:
                    if (true_label == idx_conf_matrix_true) and (predict_label == idx_conf_matrix_predict):
                        conf_matrix[idx_conf_matrix_true, idx_conf_matrix_predict] += 1
                    else:
                        pass

        return conf_matrix
