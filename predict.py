import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl

# 复制自 src/models/textCNN.py 并根据权重 shape 调整
class Config(object):
    def __init__(self):
        self.model_name = "textCNN"
        self.class_list = [x.strip() for x in open("data/data/class.txt", encoding="utf-8").readlines()]
        self.vocab_path = "data/data/vocab.pkl"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = 0.5
        self.num_classes = len(self.class_list)
        self.n_vocab = 4762  # 根据权重 shape 确定
        self.pad_size = 32
        self.embed = 300
        self.filter_sizes = (2, 3, 4, 5)
        self.num_filters = 1024 # 根据权重 shape 确定

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes]
        )
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

class Predictor:
    def __init__(self):
        self.config = Config()
        self.vocab = pkl.load(open(self.config.vocab_path, 'rb'))
        self.model = Model(self.config).to(self.config.device)
        model_path = "data/textCNN_9125.pt"
        self.model.load_state_dict(torch.load(model_path, map_location=self.config.device))
        self.model.eval()
        self.tokenizer = lambda x: [y for y in x]

    def predict(self, text):
        token = self.tokenizer(text)
        seq_len = len(token)
        pad_size = self.config.pad_size
        
        if seq_len < pad_size:
            token.extend(['[PAD]'] * (pad_size - seq_len))
        else:
            token = token[:pad_size]
        
        # 将词转换为id
        ids = [self.vocab.get(word, self.vocab.get('[UNK]')) for word in token]
        ids_tensor = torch.LongTensor([ids]).to(self.config.device)
        
        with torch.no_grad():
            outputs = self.model(ids_tensor)
            pred = torch.max(outputs.data, 1)[1].cpu().numpy()[0]
        
        return self.config.class_list[pred]
