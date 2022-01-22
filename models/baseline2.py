import math
import torch

class Model(torch.nn.Module):
    def __init__(self, model_config):

        super().__init__()
        self.enh_DNN = self._make_layers(model_config['code_dim'], model_config['dnn_l_nodes'])
        self.fc_out = torch.nn.Linear(model_config['dnn_l_nodes'][-1], 2, bias = False)

    def forward(self, embd_asv_enr, embd_asv_tst, embd_cm):    

        asv_enr = torch.squeeze(embd_asv_enr, 1) # shape: (bs, 192)
        asv_tst = torch.squeeze(embd_asv_tst, 1) # shape: (bs, 192)
        cm_tst = torch.squeeze(embd_cm, 1) # shape: (bs, 160)

        x = self.enh_DNN(torch.cat([asv_enr, asv_tst, cm_tst], dim = 1)) # shape: (bs, 32)
        x = self.fc_out(x)  # (bs, 2)

        return x

    def _make_layers(self, in_dim, l_nodes):
        l_fc = []
        for idx in range(len(l_nodes)):
            if idx == 0:
                l_fc.append(torch.nn.Linear(in_features = in_dim,
                    out_features = l_nodes[idx]))
            else:
                l_fc.append(torch.nn.Linear(in_features = l_nodes[idx-1],
                    out_features = l_nodes[idx]))
            l_fc.append(torch.nn.LeakyReLU(negative_slope = 0.3))
        return torch.nn.Sequential(*l_fc)