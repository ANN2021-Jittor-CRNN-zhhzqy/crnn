import jittor.nn as nn
import jittor.init as init
from transformer import Transformer


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.gauss_(m.weight, mean=0.0, std=0.02)
        init.zero_(m.bias)
    if classname.find('Conv') != -1:
        init.gauss_(m.weight, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        init.gauss_(m.weight, mean=1.0, std=0.02)
        init.zero_(m.bias)



class CRNN(nn.Module):
    def __init__(self,
                 num_channels,
                 num_class,
                 num_units,
                 n_rnn=2,
                 leakyRelu=False):
        super(CRNN, self).__init__()
        cnn = nn.Sequential(
            nn.Conv2d(num_channels,
                      64,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2)),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 1), padding=(0, 1)),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 1), padding=(0, 1)),
            nn.Conv2d(512, 512, kernel_size=(2, 2), stride=1, padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
        )

        self.cnn = cnn
        self.rnn = Transformer(embed_dim=512)

    def forward(self, x):
        # conv features
        x = self.cnn(x)
        x = x.squeeze(2)
        x = x.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        x = self.rnn(x)
        x = nn.log_softmax(x, dim=2)

        return x
