import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        # c: graph summary vector [H]
        # h_pl / h_mi: positive / negative node embeddings
        if isinstance(h_pl, dict):
            h_pl = torch.cat(list(h_pl.values()), dim=0)
        if isinstance(h_mi, dict):
            h_mi = torch.cat(list(h_mi.values()), dim=0)

        c_x = c.unsqueeze(0).expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x).squeeze(-1)
        sc_2 = self.f_k(h_mi, c_x).squeeze(-1)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), dim=0)
        return logits