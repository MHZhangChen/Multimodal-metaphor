import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from tensorboardX import SummaryWriter
from datetime import timedelta

loss_func = torch.nn.CrossEntropyLoss()
learning_rate = 1e-5  # 学习率
num_epochs = 20  # epoch数
require_improvement = 5000  # 若超过1000batch效果还没提升，则提前结束训练
num_classed = 2
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def train(model, train_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    for epoch in range(num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            text = trains[0]
            image = trains[1]
            objects = trains[2]
            ocr = trains[3]
            # labels = torch.sparse.torch.eye(2).index_select(dim=0, index=labels.cpu().data)
            labels = labels.to(DEVICE)

            outputs = model(text, image, objects, ocr)
            model.zero_grad()
            loss = loss_func(outputs, labels.long())
            loss.backward()
            # for p in model.parameters():
            #     print(p.grad.norm())
            nn.utils.clip_grad_norm_(model.parameters(), 20)
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                # true = labels.data.cpu()
                # predic = torch.max(outputs.data, 1)[1].cpu()
                labels = torch.sparse.torch.eye(num_classed).index_select(dim=0, index=labels.cpu().data)
                _, true = labels.data.cpu().max(1)
                _, predic = outputs.cpu().max(1)
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(model, test_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1
            if total_batch - last_improve > require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
        test(model, test_iter)
    test(model, test_iter)


def test(model, test_iter):
    # test
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    with open('results.txt', 'w', encoding='utf-8') as f:
        f.write(msg.format(test_loss, test_acc))
        f.write("\nPrecision, Recall and F1-Score...\n")
        f.write(test_report)
        f.write("\nConfusion Matrix...\n")
        f.write(str(test_confusion))


def evaluate(model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for (tests, labels) in data_iter:
            text = tests[0]
            image = tests[1]
            objects = tests[2]
            ocr = tests[3]
            # labels = torch.sparse.torch.eye(2).index_select(dim=0, index=labels.cpu().data)
            labels = labels.to(DEVICE)

            outputs = model(text, image, objects, ocr)
            # print(test, texts.shape, labels.shape, outputs.shape)
            loss = loss_func(outputs, labels.long())
            loss_total += loss

            labels = torch.sparse.torch.eye(num_classed).index_select(dim=0, index=labels.cpu().data)
            _, true = labels.data.cpu().max(1)
            _, predic = outputs.cpu().max(1)

            labels_all = np.append(labels_all, true)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=['literature', 'metaphor'],
                                               digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
