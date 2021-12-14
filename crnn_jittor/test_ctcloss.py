import jittor as jt
jt.flags.use_cuda = 1

from model import CRNN

import numpy as np
np.random.seed(2021)


class FirstModel(jt.Module):

    def __init__(self):
        super().__init__()

        # 随机初始化参数 a 和 b
        self.a = jt.rand(1)
        self.b = jt.rand(1) 

    def execute(self, x):
        # 模型通过输入的 x 值，进行与参数 a 和参数 b 的计算，得到预测的 y 值，并返回计算结果
        y_pred = self.a + self.b * x
        return y_pred

model = CRNN(num_channels=1, num_class=37, num_units=64, num_layers=1)
# model = FirstModel()


optimizer = jt.nn.SGD(model.parameters(), 0.1)
# criterion = jt.nn.MSELoss()
criterion = jt.CTCLoss(blank=0)

T = 24
C = 37
N = 256
S = 17
S_min = 5

img = jt.rand(256, 1, 32, 100)
preds = model(img)
# preds = jt.randn(24, 256, 37).log_softmax(2)
target = jt.randint(low=1, high=37, shape=(256, 17), dtype=jt.int)

preds_length = jt.full((256,), 24, dtype=jt.int)
target_length = jt.randint(low=5, high=18, shape=(256,), dtype=jt.int)

print("preds.dtype ", preds.dtype)
print("preds.shape ", preds.shape)
print("target.dtype ", target.dtype)
print("target.shape", target.shape)
print("preds_length.dtype ", preds_length.dtype)
print("preds_length.shape", preds_length.shape)
print("target_length.dtype ", target_length.dtype)
print("target_length.shape", target_length.shape)
loss = criterion(preds, target, preds_length, target_length)
optimizer.step(loss)
