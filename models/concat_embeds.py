import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class int_model(nn.Module):
    def __init__(self, d_args):

        super(int_model, self).__init__()
        self.d_args = d_args
        self.enh_DNN = self._make_layers(d_args['code_dim'], d_args['dnn_l_nodes'])
        self.fc_out = nn.Linear(d_args['dnn_l_nodes'][-1], 2, bias = False)

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm):    

        asv_enr = torch.squeeze(embd_asv_enr, 1) # 32, 192
        asv_tst = torch.squeeze(embd_asv_tst, 1) # 32, 192
        cm_tst = torch.squeeze(embd_cm, 1) # 32, 160

        x = self.enh_DNN(torch.cat([asv_enr, asv_tst, cm_tst], dim = 1)) # 32, 32
        x = self.fc_out(x)  # 32, 2

        return x

    def _make_layers(self, in_dim, l_nodes):
        l_fc = []
        for idx in range(len(l_nodes)):
            if idx == 0:
                l_fc.append(nn.Linear(in_features = in_dim,
                    out_features = l_nodes[idx]))
            else:
                l_fc.append(nn.Linear(in_features = l_nodes[idx-1],
                    out_features = l_nodes[idx]))
            l_fc.append(nn.LeakyReLU(negative_slope = 0.3))
        return nn.Sequential(*l_fc)