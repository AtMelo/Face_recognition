import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self,alpha=0.2):
        super(TripletLoss, self).__init__()
        self.alpha = alpha

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha
        pos_dist = torch.pow(anchor - positive,2).sum(dim=1)
        neg_dist = torch.pow(anchor - negative,2).sum(dim=1)
        triplet_loss = torch.relu(pos_dist-neg_dist+alpha)
        return triplet_loss.mean()


def get_triplet_loss(anchor,positive, negative,alpha=0.2):
    return TripletLoss(alpha)(anchor,positive,negative)
