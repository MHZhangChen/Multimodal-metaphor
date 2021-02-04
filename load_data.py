import csv, pickle, re
import torch
import numpy as np
from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM
from PIL import Image
from torchvision import transforms

def load_dataset_new(path, text_pad = 35, ocr_pad = 20):
    contents = []
    results = re.compile(r'[http|https|pic]*[://|.][a-zA-Z0-9.?/&=:]*', re.S)

    with open('id_imgpath1.pkl', 'rb') as f:
        id_imgpath = pickle.load(f)
    with open('id_objfeature1.pkl', 'rb') as f:
        id_objfeature = pickle.load(f)
    with open('id_OCR1.pkl', 'rb') as f:
        id_OCR = pickle.load(f)
    with open('id_text1.pkl', 'rb') as f:
        id_text = pickle.load(f)

    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    with open(path, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            id = line[0]
            label = line[1]
            if id not in id_imgpath.keys() or id not in id_text.keys():
                continue

            text = id_text[id]
            text = text.replace('\n', '')
            text = text.replace('\t', '')
            text = text.lower()
            text = text.replace('#', '')
            text = text.replace('metaphor', '')
            text = text.replace('@', 'user ')
            text = text.replace('-', ' ')
            text = re.sub(results, '', text)

            imgpaths = id_imgpath[id]

            ocr = id_OCR.get(id, "")
            ocr = ocr.replace('\n', '')
            ocr = ocr.replace('\t', '')
            ocr = ocr.lower()
            ocr = ocr.replace('#', '')
            ocr = ocr.replace('metaphor', '')
            ocr = ocr.replace('@', 'user ')
            ocr = ocr.replace('-', ' ')
            ocr = re.sub(results, '', ocr)

            obj = id_objfeature.get(id, np.zeros(300))

            text_tokens = tokenizer.encode(text=text)
            if len(text_tokens) < text_pad:
                text_tokens.extend([tokenizer.pad_token_id] * (text_pad - len(text_tokens)))
            else:
                text_tokens = text_tokens[:text_pad]

            ocr_tokens = tokenizer.encode(text=ocr)
            if len(ocr_tokens) < ocr_pad:
                ocr_tokens.extend([tokenizer.pad_token_id] * (ocr_pad - len(ocr_tokens)))
            else:
                ocr_tokens = ocr_tokens[:ocr_pad]

            for path in imgpaths:
                pa = path.split('\\')
                path = "."
                for p in pa:
                    path += '/' + p
                try:
                    img = Image.open(path)
                    img = img.convert('RGB')
                    img = trans(img)
                # img.unsqueeze_(dim=0)
                    contents.append([text_tokens, img, obj, ocr_tokens, int(label)])
                except OSError:
                    pass

    return contents  # [([...], 0), ([...], 1), ...]

def load_dataset(path):
    contents = []
    with open('id_imgfeature.pkl', 'rb') as f:
        id_imgfeature = pickle.load(f)
    with open('id_objfeature.pkl', 'rb') as f:
        id_objfeature = pickle.load(f)
    with open('id_ocrfeature.pkl', 'rb') as f:
        id_OCR = pickle.load(f)
    with open('id_textfeature.pkl', 'rb') as f:
        id_text = pickle.load(f)
    with open(path, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        next(reader)
        for line in reader:
            id = line[0]
            label = line[1]
            if id not in id_imgfeature.keys() or id not in id_text.keys():
                continue
            text = id_text[id]
            img = id_imgfeature[id]
            ocr = id_OCR.get(id, torch.zeros((20, 768)))
            obj = id_objfeature.get(id, torch.zeros(300))

            contents.append([text, img, obj, ocr, int(label)])

    return contents  # [([...], 0), ([...], 1), ...]

def build_dataset(new = True, text_pad = 35, ocr_pad = 20):
    train_path = 'train_withFB.csv'
    test_path = 'test_withFB.csv'
    if new:
        train = load_dataset_new(train_path, text_pad = text_pad, ocr_pad = ocr_pad)
        test = load_dataset_new(test_path, text_pad = text_pad, ocr_pad = ocr_pad)
    else:
        train = load_dataset(train_path)
        test = load_dataset(test_path)
    return train, test

class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        text = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        image = torch.FloatTensor([_[1].cpu().numpy() for _ in datas]).to(self.device)
        objects = torch.FloatTensor([_[2] for _ in datas]).to(self.device)
        ocr = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        # print(text.shape)
        # print(objects.shape)
        # print(ocr.shape)
        # print(image.shape)
        # y = y.unsqueeze(-1)

        # pad前的长度(超过pad_size的设为pad_size)
        # seq_len = torch.LongTensor([_[-1] for _ in datas]).to(self.device)
        return [text, image, objects, ocr], y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches

def build_iterator(dataset, batch_size, device):
    iter = DatasetIterater(dataset, batch_size, device)
    return iter

if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab, train_data, test_data = build_dataset(new=True, text_pad=35, ocr_pad=20)
    train_iter = build_iterator(train_data, batch_size=8, device=DEVICE)
    test_iter = build_iterator(train_data, batch_size=8, device=DEVICE)
