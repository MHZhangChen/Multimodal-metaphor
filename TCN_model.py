import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        # 表示对继承自父类属性进行初始化
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        tensor.contiguous()会返回有连续内存的相同张量
        有些tensor并不是占用一整块内存，而是由不同的数据块组成
        tensor的view()操作依赖于内存是整块的，这时只需要执行
        contiguous()函数，就是把tensor变成在内存中连续分布的形式
        本函数主要是增加padding方式对卷积后的张量做切边而实现因果卷积
        """
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数,输入层通道为1，隐含层是25。
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)   # *作用是将输入迭代器拆成一个个元素

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)

class TCN(nn.Module):
    def __init__(self,  num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(768, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], 1024)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)"""
        # out = self.embedding(inputs)
        # emb = self.embedding(x)
        # out = out.unsqueeze(1)  # (N,Ci,W,D)
        y1 = F.relu(self.tcn(x.transpose(1, 2)))  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])# 增加一个维度,1D·FCN
        # return F.log_softmax(o, dim=1)
        return o

if __name__ == '__main__':
    filter_sizes = [2, 3, 4]
    num_filters = 256
    dropout = 0.1
    tcn = TCN(filter_sizes, num_filters, dropout=dropout)

    text = torch.rand(8, 35, 768)

    out = tcn(text)
    print(out.shape)
