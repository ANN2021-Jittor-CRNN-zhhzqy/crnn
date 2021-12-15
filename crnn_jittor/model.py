import jittor as jt
import jittor.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        jt.init.gauss_(m.weight, mean=0.0, std=0.02)
        jt.init.zero_(m.bias)
    if classname.find('Conv') != -1:
        jt.init.gauss_(m.weight, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        jt.init.gauss_(m.weight, mean=1.0, std=0.02)
        jt.init.zero_(m.bias)


def show_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        print("Linear ", m.weight)
    if classname.find('Conv') != -1:
        print("Conv ", m.weight)
    elif classname.find('BatchNorm') != -1:
        print("BatchNorm ", m.weight)


class BidirectionalLSTM(jt.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.embedding = nn.Linear(hidden_dim * 2, output_dim)

    def execute(self, x):
        x, _ = self.rnn(x)
        T, b, h = x.shape
        x = self.embedding(x.view(T * b, h)).view(T, b,
                                                  -1)  # (24, 256, num_class)
        return x


class CRNN(jt.Module):
    def __init__(self, num_channels, num_class, num_units=256, num_layers=2):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, (3, 3), stride=1,
                      padding=1),  # (256, 64, 32, 100)
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2)),  # (256, 64, 16, 50)
            nn.Conv2d(64, 128, (3, 3), stride=1,
                      padding=1),  # (256, 128, 16, 50)
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2)),  # (256, 128, 8, 25)
            nn.Conv2d(128, 256, (3, 3), stride=1,
                      padding=1),  # (256, 256, 8, 25)
            nn.BatchNorm2d(256),  # !!
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, (3, 3), stride=1,
                      padding=1),  # (256, 256, 8, 25)
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 1),
                         padding=(0, 1)),  # (256, 256, 4, 25)
            nn.Conv2d(256, 512, (3, 3), stride=1,
                      padding=1),  # (256, 512, 4, 25)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(512, 512, (3, 3), stride=1,
                      padding=1),  # (256, 512, 4, 25)
            # nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d((2, 2), stride=(2, 1),
                         padding=(0, 1)),  # (256, 512, 2, 25) 
            nn.Conv2d(512, 512, (2, 2), stride=1,
                      padding=0),  # (256, 512, 1, 24)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
        )
        #self.rnn = nn.LSTM(512, num_units, num_layers=num_layers, bidirectional=True)
        #self.linear = nn.Linear(2 * num_units, num_class)
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, num_units, num_units),
            BidirectionalLSTM(num_units, num_units, num_class),
        )

    def execute(self, x):  # (256, 1, 32, 100)
        x = self.cnn(x)  # (256, 512, 1, 24)
        x = x.squeeze(2).permute((2, 0, 1))
        # (24, 256, 512) --- seq(width), batch, feature(channel)
        # x, _ = self.rnn(x)  # (24, 256, 1024) ---- h_0, c_0 default to zero if not provided; output dim double because rnn is bidirectional
        # T, b, h = x.shape
        # x = self.linear(x.view(T * b, h)).view(T, b, -1)  # (24, 256, num_class)
        x = self.rnn(x)
        x = x.log_softmax(dim=2)
        return x
