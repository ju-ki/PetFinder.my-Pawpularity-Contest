import torch
import torch.nn as nn


class my_round_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class ScaleLayer(nn.Module):
    def __init__(self, max_value, min_value, step_value):
        """[summary]
        The following custom layer will rescale the output to fit the discrete steps in values to be found in the target. 
        In such a way, you will force your network to learn how to provide outputs that do not need further post processing.

        Args:
            max_value ([int or float?])
            min_value ([int or float])
            step_value(diff discrete value?)
        """
        "今回のtargetは一定の間隔があったため他のコンペでは扱いにくい可能性大"
        "VPPではtargetの間隔が0.07とほぼ一定だったためこの手法は有効だったかも"
        "https://www.kaggle.com/lucamassaron/rescaling-layer-for-discrete-output-in-tensorflow"
        super(ScaleLayer, self).__init__()
        self.min = max_value
        self.max = min_value
        self.step = step_value
        self.my_round_func = my_round_func()

    def forward(self, inputs):
        steps = inputs.add(-self.min).divide(self.step)
        int_steps = self.my_round_func.apply(steps)
        rescaled_steps = int_steps.multiply(self.step).add(self.min)
        clipped = torch.clamp(rescaled_steps, self.min, self.max)
        return clipped
