import torch

import torch.nn as nn

# Example of target with class indices
loss = nn.CrossEntropyLoss()
input = torch.tensor([[[0, 100., 0, 0, 0],
                       [100., 0, 0, 0, 0],
                       [0, 0, 0, 0, 100.]],

                      [[0, 100., 0, 0, 0],
                       [100., 0, 0, 0, 0],
                       [0, 0, 0, 0, 100.]]], requires_grad=True)
print(input)
print(input.transpose(1, 2))
print(input.view(-1, 5, 3))
print(torch.argmax(input, dim=-1))
print(loss(input.view(-1, 5, 3), torch.argmax(input, dim=-1)).item())
print(loss(input.transpose(1, 2), torch.argmax(input, dim=-1)).item())
