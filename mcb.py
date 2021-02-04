import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class CompactBilinearPooling(torch.nn.Module):
    def __init__(self, input_dim1, input_dim2, output_dim,
                 sum_pool=True, device=DEVICE):
        super(CompactBilinearPooling, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool

        rand_h_1 = np.random.randint(output_dim, size=self.input_dim1)
        rand_s_1 = 2 * np.random.randint(2, size=self.input_dim1) - 1
        self.sparse_sketch_matrix1 = Variable(self.generate_sketch_matrix(
            rand_h_1, rand_s_1, self.output_dim)).to(device)

        rand_h_2 = np.random.randint(output_dim, size=self.input_dim2)
        rand_s_2 = 2 * np.random.randint(2, size=self.input_dim2) - 1
        self.sparse_sketch_matrix2 = Variable(self.generate_sketch_matrix(
            rand_h_2, rand_s_2, self.output_dim)).to(device)

    def generate_sketch_matrix(self, rand_h, rand_s, output_dim):
        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert (rand_h.ndim == 1 and rand_s.ndim ==
                1 and len(rand_h) == len(rand_s))
        assert (np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[:, np.newaxis],
                                  rand_h[:, np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        rand_s = torch.from_numpy(rand_s)
        sparse_sketch_matrix = torch.sparse.FloatTensor(
            indices.t(), rand_s, torch.Size([input_dim, output_dim]))
        return sparse_sketch_matrix.to_dense()

    def forward(self, bottom1, bottom2):
        assert bottom1.size(1) == self.input_dim1 and \
               bottom2.size(1) == self.input_dim2

        batch_size, _, height, width = bottom1.size()

        bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1)
        bottom2_flat = bottom2.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2)

        sketch_1 = bottom1_flat.mm(self.sparse_sketch_matrix1)
        sketch_2 = bottom2_flat.mm(self.sparse_sketch_matrix2)
        sketch_1 = torch.stack((sketch_1, torch.zeros_like(sketch_1)), 2)
        sketch_2 = torch.stack((sketch_2, torch.zeros_like(sketch_2)), 2)

        fft1 = torch.fft(sketch_1, 1).split(1, dim=-1)
        fft1_real = fft1[0].squeeze()
        fft1_imag = fft1[1].squeeze()

        fft2 = torch.fft(sketch_2, 1).split(1, dim=-1)
        fft2_real = fft2[0].squeeze()
        fft2_imag = fft2[1].squeeze()

        fft_product_real = fft1_real.mul(fft2_real) - fft1_imag.mul(fft2_imag)
        fft_product_imag = fft1_real.mul(fft2_imag) + fft1_imag.mul(fft2_real)
        fft_product = torch.stack((fft_product_real, fft_product_imag), dim=2)

        cbp_flat = torch.ifft(fft_product, 1).split(1, dim=-1)[0].squeeze()
        cbp = cbp_flat.view(batch_size, height, width, self.output_dim)

        if self.sum_pool:
            cbp = cbp.sum(dim=1).sum(dim=1)

        return cbp


if __name__ == '__main__':
    # bottom1 = torch.rand((32, 512, 7, 7))
    # bottom2 = torch.rand((32, 512, 7, 7))

    bottom1 = Variable(torch.randn(32, 512, 14, 14))
    bottom2 = Variable(torch.randn(32, 512, 14, 14))

    layer = CompactBilinearPooling(512, 512, 8000, sum_pool=False)
    pool = nn.AvgPool2d((14, 14), stride=(1, 1))
    out = layer(bottom1, bottom2).permute(0, 3, 1, 2)
    print(out.shape)
    out = pool(out).squeeze()
    print(out.shape)
    # bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, 512)
    # print(bottom1_flat.shape)
    #
    # rand_h_1 = np.random.randint(16000, size=512)
    # print(rand_h_1.shape)
    # print(rand_h_1[:, np.newaxis])
    # print(rand_h_1[..., np.newaxis])
    # rand_s_1 = 2 * np.random.randint(2, size=512) - 1
    # print(rand_h_1.shape)
    # print(rand_s_1.shape)
    #
    # sparse_sketch_matrix1 = Variable(generate_sketch_matrix(rand_h_1, rand_s_1, 16000))
    # print(sparse_sketch_matrix1.shape)
    # sketch_1 = bottom1_flat.mm(sparse_sketch_matrix1)
    # print(sketch_1.shape)
    # sketch_1 = torch.stack((sketch_1, torch.zeros_like(sketch_1)), 2)
    # print(sketch_1.shape)
    #
    # fft1 = torch.fft(sketch_1, 1)
    # fft1 = fft1.split(1, dim=-1)
    # fft1_real = fft1[0].squeeze()
    # fft1_imag = fft1[1].squeeze()
    #
    # print(fft1_real.shape)
    # print(fft1_imag.shape)
    #
    # fft1 = torch.stack((fft1_real, fft1_imag), dim=2)
    # print(fft1.shape)
    # cbp_flat = torch.ifft(fft1, 1).split(1, dim=-1)[0].squeeze()
    # print(cbp_flat.shape)
    #
    # cbp = cbp_flat.view(32, 7, 7, 16000)
    #
    # print(cbp.shape)
    # cbp = cbp.sum(dim=1).sum(dim=1)
    # print(cbp.shape)
