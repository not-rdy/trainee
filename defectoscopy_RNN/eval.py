import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def eval_model(model, test, loss_func):
    val_loss_batch = []

    class_recognition = {'class_0_hit': [], 'class_0_true': [],
                         'class_2_hit': [], 'class_2_true': [],
                         'class_3_hit': [], 'class_3_true': [],
                         'class_5_hit': [], 'class_5_true': [],
                         'class_6_hit': [], 'class_6_true': []}

    eval_metrics = {'val_loss_part': None,
                    'class_0': None, 'class_2': None,
                    'class_3': None, 'class_5': None, 'class_6': None}

    model.eval()
    with torch.no_grad():
        for x, y in test:
            # loss on valid set
            raw_predict = model(x.to(device))
            loss = loss_func(raw_predict, y.reshape(-1).to(device))
            val_loss_batch.append(loss.item())

            # class recognition
            probs = nn.Softmax(dim=1)(raw_predict)
            predict_labels = [torch.argmax(label).item() for label in probs]
            true_labels = y.reshape(-1).tolist()

            for label_predict, label_true in zip(predict_labels, true_labels):
                if (label_true == 0) and (label_true == label_predict):
                    class_recognition['class_0_true'].append(1)
                    class_recognition['class_0_hit'].append(1)
                elif (label_true == 0) and (label_true != label_predict):
                    class_recognition['class_0_true'].append(1)
                    class_recognition['class_0_hit'].append(0)
                elif (label_true == 1) and (label_true == label_predict):
                    class_recognition['class_2_true'].append(1)
                    class_recognition['class_2_hit'].append(1)
                elif (label_true == 1) and (label_true != label_predict):
                    class_recognition['class_2_true'].append(1)
                    class_recognition['class_2_hit'].append(0)
                elif (label_true == 2) and (label_true == label_predict):
                    class_recognition['class_3_true'].append(1)
                    class_recognition['class_3_hit'].append(1)
                elif (label_true == 2) and (label_true != label_predict):
                    class_recognition['class_3_true'].append(1)
                    class_recognition['class_3_hit'].append(0)
                elif (label_true == 3) and (label_true == label_predict):
                    class_recognition['class_5_true'].append(1)
                    class_recognition['class_5_hit'].append(1)
                elif (label_true == 3) and (label_true != label_predict):
                    class_recognition['class_5_true'].append(1)
                    class_recognition['class_5_hit'].append(0)
                elif (label_true == 4) and (label_true == label_predict):
                    class_recognition['class_6_true'].append(1)
                    class_recognition['class_6_hit'].append(1)
                elif (label_true == 4) and (label_true != label_predict):
                    class_recognition['class_6_true'].append(1)
                    class_recognition['class_6_hit'].append(0)

        try:
            eval_metrics['class_0'] = round(sum(class_recognition['class_0_hit']) /
                                            sum(class_recognition['class_0_true']), 2)
        except ZeroDivisionError:
            eval_metrics['class_0'] = 0

        try:
            eval_metrics['class_2'] = round(sum(class_recognition['class_2_hit']) /
                                            sum(class_recognition['class_2_true']), 2)
        except ZeroDivisionError:
            eval_metrics['class_2'] = 0

        try:
            eval_metrics['class_3'] = round(sum(class_recognition['class_3_hit']) /
                                            sum(class_recognition['class_3_true']), 2)
        except ZeroDivisionError:
            eval_metrics['class_3'] = 0

        try:
            eval_metrics['class_5'] = round(sum(class_recognition['class_5_hit']) /
                                            sum(class_recognition['class_5_true']), 2)
        except ZeroDivisionError:
            eval_metrics['class_5'] = 0

        try:
            eval_metrics['class_6'] = round(sum(class_recognition['class_6_hit']) /
                                            sum(class_recognition['class_6_true']), 2)
        except ZeroDivisionError:
            eval_metrics['class_6'] = 0

        eval_metrics['val_loss_part'] = (sum(val_loss_batch) / len(val_loss_batch))

        return eval_metrics
