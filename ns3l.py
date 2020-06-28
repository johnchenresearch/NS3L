# NS3L Loss which can be easily added to the VAT Loss or
# the MixMatch loss.

import torch
import torch.nn as nn
import torch.nn.functional as F
  
class NS3L(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.th = threshold

    def forward(self, x, y, model, mask):
        y_probs = y.softmax(1)
        gt_mask = (y_probs < self.th).float()
        gt_mask = gt_mask.max(1)[0]
        lt_mask = 1 - gt_mask
        lt_target = lt_mask[:,None] * y_probs
        gt_mask = (y_probs < self.th).float()

        model.update_batch_stats(False)

        output = model(x)
        output_softmax = output.softmax(1)
        n_tot = (1 - (gt_mask.detach() * output_softmax).sum(1) ).clamp(1e-8,1).log()
        loss = (-( n_tot  )*mask).mean()
        model.update_batch_stats(True)

        return loss

    def __make_one_hot(self, y, n_classes=10):
        return torch.eye(n_classes)[y].to(y.device)
ssl_obj = NS3L(threshold=0.04)