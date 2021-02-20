from inference import cal_dis
import torch
q = torch.tensor([[0, 0, 0]])
g = torch.tensor([[1, 2, 3]])
result = cal_dis(q, g)
print(result)