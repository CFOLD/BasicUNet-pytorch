import torch
import torch.nn.functional as F

from torch import nn
from torch import Tensor


class DiceLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def dice_coeff(self, inputs: Tensor, targets: Tensor, epsilon=1e-10, eval=False):
        inputs = F.softmax(inputs, dim=1).float()
        targets = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()
        assert inputs.size() == targets.size()

        dice, dice_batch = 0, 0

        # Separate the inputs by class for Multi-classes
        class_range = range(inputs.shape[1])
        if eval and (len(class_range) > 1):
            class_range = class_range[1:] # ignoring the background during evaluation
        for _class in class_range:
            inputs_class = inputs[:, _class, ...]
            targets_class = targets[:, _class, ...]

            dice_batch = 0
            for i in range(inputs.shape[0]):
                input = inputs_class[i].reshape(-1)
                target = targets_class[i].reshape(-1)
    
                intersection = torch.dot(input, target)
                _sum = input.sum() + target.sum()

                dice_batch += (2.*intersection + epsilon) / (_sum + epsilon)
            dice += dice_batch / inputs.shape[0]

        return dice / inputs.shape[1]

    def forward(self, inputs: Tensor, targets: Tensor, epsilon=1e-10):
        dice = self.dice_coeff(inputs, targets, epsilon)
        return 1 - dice