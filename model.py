import chainer.links as L
import chainer.functions as F
import cupy as xp
import chainer

class conv_unit(chainer.Chain):
    def __init__(self, out_ch, stride=1):
        super(conv_unit, self).__init__()
        with self.init_scope():
            self.c1 = L.Convolution2D(None, out_ch, 3, pad=1, stride=stride)
            self.bn1 = L.BatchNormalization(out_ch)
    def forward(self, x):
        y = F.leaky_relu(self.bn1(self.c1(x)), slope=0.1)
        return y

class conv_block(chainer.Chain):
    def __init__(self, out_ch, make_last=True):
        super(conv_block, self).__init__()
        with self.init_scope():
            self.make_last = make_last
            for i in range(1,4):
                setattr(self, "c"+str(i), conv_unit(out_ch))
            if self.make_last:
                setattr(self, "c4", conv_unit(out_ch, 2))
    def forward(self, x):
        for i in range(1, 4):
            x = getattr(self, "c"+str(i))(x)
        if self.make_last:
            x = getattr(self, "c4")(x)
        return x


class CNN(chainer.Chain):
    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            for i in range(1,4):
                setattr(self, "c"+str(i), conv_block(8*(2**(i-1))))
            setattr(self, "c4", conv_block(64, False))
            self.l1 = L.Linear(None, 2)

    def forward(self, x):
        for i in range(1,4):
            x = getattr(self, "c"+str(i))(x)
        x = getattr(self, "c4")(x)
        x = F.mean(x, axis=(2,3))
        x = self.l1(x)

        return x
