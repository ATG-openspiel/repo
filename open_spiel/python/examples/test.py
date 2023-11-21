import torch

reward=[[1,2],[3,4],[4,5]]
l = torch.tensor(reward)
new_tensor = l[:, 0].view(-1, 1)

print(new_tensor)
