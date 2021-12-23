import torch

from model import CRNN

device = torch.device("cpu")

model = CRNN(device=device, num_channels=1, num_class=(37), num_units=256)

model.load_state_dict(torch.load("../crnn_jittor/result/ep14_bs256_v1/e2_i1999"))
model.to(device)
model.eval()

eg = torch.rand(1, 1, 32, 100).to(device)
trace=torch.jit.trace(model,eg)
trace.save("crnn_j.pt")
