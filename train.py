from functools import partial
import chainer
import chainer.links as L

from chainer import optimizers, training
from chainer.training import extensions
from chainer.datasets import LabeledImageDataset, TransformDataset
from dataset import *
from model import CNN

def train_and_validate(model, optimizer, train, validation, n_epoch, batchsize, device):

    # 1. deviceがgpuであれば、gpuにモデルのデータを転送する
    if device >= 0:
        model.to_gpu(device)

    # 2. Optimizerを設定する
    optimizer.setup(model)

    # 3. DatasetからIteratorを作成する
    train_iter = chainer.iterators.MultiprocessIterator(train, batchsize)
    validation_iter = chainer.iterators.MultiprocessIterator(
        validation, batchsize, repeat=False, shuffle=False)

    # 4. Updater・Trainerを作成する
    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'), out='out')

    # 5. Trainerの機能を拡張する
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.Evaluator(validation_iter, model, device=device), name='val')
    trainer.extend(extensions.ExponentialShift('alpha', 0.1), trigger=(20, 'epoch'))
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'val/main/loss'],x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.dump_graph('main/loss'))

    # 6. 訓練を開始する
    trainer.run()

if __name__ == '__main__':
    # Enable autotuner of cuDNN
    chainer.config.autotune = True
    device = 0
    n_epoch = 60 # Only 5 epochs
    batchsize = 128
    model = CNN() # CNN model
    classifier_model = L.Classifier(model)
    optimizer = optimizers.Adam(1e-3)

    train_path = get_image_filepath_list("cats_and_dogs_small/train/")
    val_path = get_image_filepath_list("cats_and_dogs_small/validation/")

    train_data = LabeledImageDataset(train_path)
    val_data = LabeledImageDataset(val_path)

    train_transform = partial(transform, random_angle=15., pca_sigma=255., train=True)
    val_transform = partial(transform, random_angle=15., pca_sigma=255.,train=False)

    train = TransformDataset(train_data, train_transform)
    val = TransformDataset(val_data, val_transform)

    train_and_validate(classifier_model, optimizer, train, val, n_epoch, batchsize, device)

    # optimizer = optimizers.NesterovAG(1e-4)
    # train_and_validate(classifier_model, optimizer, train, val, n_epoch, batchsize, device)
