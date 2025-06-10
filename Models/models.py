import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SimpleMLP4(nn.Module):
    """Shared initial"""

    def __init__(
        self,
        gene_sizes: dict,
        input_dim=2560,
        emb_dim=256,
        out_dim=3,
        ann_dim=65,
        use_ann: bool = False,
        **kwargs,
    ):
        super(SimpleMLP4, self).__init__()

        pad_size = kwargs.get("pad_size", 20)
        self.use_ann = use_ann
        ann_emb_dim = kwargs.get("ann_emb_dim", 256)
        squeeze_dim = kwargs.get("squeeze_dim", 16)  # 8

        if use_ann:
            self.l_ann = nn.Sequential(
                nn.Linear(ann_dim, ann_emb_dim),
                nn.LayerNorm(ann_emb_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.25),
                nn.Linear(ann_emb_dim, 32),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.25),
            )
            self.l_out = nn.Linear(64, out_dim)
        else:
            self.l_out = nn.Linear(32, out_dim)

        # Multi col
        self.l_1 = nn.Linear(input_dim, emb_dim)
        self.l_2 = nn.Linear(emb_dim, squeeze_dim)
        self.l_3 = nn.Linear(squeeze_dim * pad_size * 2 + squeeze_dim, 32)

    def forward(self, x, ann, **kwargs):
        # x :: (bs, n_res, 1280)

        x = self.l_1(x)
        x = F.dropout(F.leaky_relu(x, 0.2), 0.25)
        x = self.l_2(x)
        x = F.dropout(F.leaky_relu(x, 0.2), 0.25)
        x = x.reshape(x.shape[0], -1)
        x = self.l_3(x)
        x = F.dropout(F.leaky_relu(x, 0.2), 0.25)

        if self.use_ann:
            a = self.l_ann(ann)
            x = torch.cat((x, a), dim=1)
        return self.l_out(x)
