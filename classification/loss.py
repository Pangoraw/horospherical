import math
import torch
from torch import nn


class PeBusePenalty(nn.Module):
    def __init__(self, dimension, penalty_option='dim', mult=1.0):
        super(PeBusePenalty, self).__init__()
        self.dimension = dimension
        if penalty_option == 'non':
            self.penalty_constant = 1.0
        elif penalty_option == 'dim':
            self.penalty_constant = mult * self.dimension
        else:
            print('~~~~~~~~!Your option is not available, I am choosing!~~~~~~~~')
            self.penalty_constant = 1.0

    def forward(self, p, g):
        # first part of loss
        prediction_difference = g - p
        difference_norm = torch.norm(prediction_difference, dim=1)
        difference_log = 2 * torch.log(difference_norm)

        # second part of loss
        data_norm = torch.norm(p, dim=1)
        proto_difference = (1 - data_norm.pow(2) + 1e-6)
        proto_log = (1 + self.penalty_constant) * torch.log(proto_difference)

        # second part of loss
        constant_loss = self.penalty_constant * math.log(2)

        one_loss = difference_log - proto_log + constant_loss
        total_loss = torch.mean(one_loss)

        return total_loss

