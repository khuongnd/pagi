import torch
import torchvision.models as models

from pagi.loss import dcl

resnet18 = models.resnet18()
random_input = torch.rand((10, 3, 244, 244))
output = resnet18(random_input)

# for DCL
loss_fn = dcl.DCL(temperature=0.5)
loss = loss_fn(output, output)  # loss = tensor(-0.2726, grad_fn=<AddBackward0>

# for DCLW
loss_fn = dcl.DCLW(temperature=0.5, sigma=0.5)
loss = loss_fn(output, output)  # loss = tensor(38.8402, grad_fn=<AddBackward0>)

print(loss)