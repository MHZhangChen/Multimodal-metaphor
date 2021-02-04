import time, os
import numpy as np
from load_data import build_dataset, build_iterator
from model import mergeNet
from train_eval import *

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device_ids = [2]

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

if __name__ == '__main__':
    print(DEVICE)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # 获取谷歌词向量
    print("Loading data...")
    start_time = time.time()
    train_data, test_data = build_dataset(new=True, text_pad=35, ocr_pad=20)
    train_iter = build_iterator(train_data, batch_size=48, device=DEVICE)
    test_iter = build_iterator(test_data, batch_size=48, device=DEVICE)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    model = mergeNet()
    # model = nn.DataParallel(model, device_ids=device_ids)
    model.to(DEVICE)
    # model.cuda(device=device_ids[0])

    train(model, train_iter, test_iter)
