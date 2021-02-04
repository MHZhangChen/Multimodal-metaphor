import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from TCN_model import *
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from pytorch_transformers import BertModel
from mcb import CompactBilinearPooling

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.res_model = models.resnet50(pretrained=True)
        self.new_model = torch.nn.Sequential(*list(self.res_model.children())[:-1])

    def forward(self, x):
        image_feature_map = self.new_model(x).data
        return image_feature_map #(n, 2048, 7, 7)

class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, x):
        text_feature_map = self.model(x)[0]
        return text_feature_map #(n, seq_len, 768)

class mergeNet(nn.Module):
    def __init__(self):
        super(mergeNet, self).__init__()
        self.textEncoder = TextEncoder()
        self.imageEncoder = ImageEncoder()
        for p in self.parameters():
            p.requires_grad = False
        self.filter_sizes = [2, 3, 4]
        self.num_filters = 256
        self.dropout = 0
        self.tcn = TCN(self.filter_sizes, self.num_filters, dropout=self.dropout)
        self.mcb1 = CompactBilinearPooling(2048, 2048, 8000, sum_pool=False)
        self.cov = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=1, stride=1, padding=0)
        self.pool = nn.AvgPool2d((7, 7), stride=(1, 1))
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 300)
        self.mcb2 = CompactBilinearPooling(300, 300, 1024, sum_pool=True)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(256, 2)

    def forward(self, text, img, obj, ocr):
        img = self.imageEncoder(img).squeeze()
        text = self.textEncoder(text)
        ocr = self.textEncoder(ocr)
        it_vec = F.relu(self.tcn(ocr))
        tt_vec = F.relu(self.tcn(text)) # 1024
        tx_vec = torch.cat([img, it_vec, tt_vec], dim=1)
        # tx_vec = torch.cat([it_vec, tt_vec], dim=1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 7, 7) # 2048
        # mcb_feature = self.mcb1(img, tx_vec).permute(0, 3, 1, 2)
        # merge_map = self.cov(img)
        # merge_map = self.pool(merge_map).squeeze()
        merge_map = F.relu(tx_vec)
        merge_vec = F.relu(self.fc1(merge_map))
        # merge_vec = F.relu(self.fc2(merge_map))
        # print(merge_vec.shape)
        # print(obj.shape)
        # merge_vec = self.mcb2(merge_vec.unsqueeze(-1).unsqueeze(-1), obj.unsqueeze(-1).unsqueeze(-1))
        merge_vec = F.relu(self.fc3(merge_vec))
        out = F.softmax(self.fc4(merge_vec), dim=1)
        return out

if __name__ == '__main__':
    img = torch.rand(16, 2048, 7, 7)
    text = torch.rand(16, 35, 768)
    obj = torch.rand(16, 300)
    ocr = torch.rand(16, 20, 768)

    model = mergeNet()

    out = model(img, text, obj, ocr)

    print(out.shape)
