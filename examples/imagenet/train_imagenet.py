#!/usr/bin/env python

from __future__ import print_function
import argparse
import multiprocessing
import random
import sys

import numpy as np

import chainer
import chainer.cuda
from chainer import dataset
from chainer import training
from chainer.training import extensions

import chainermn


import models.alex as alex
import models.googlenet as googlenet
import models.googlenetbn as googlenetbn
import models.nin as nin
import models.resnet50 as resnet50

try:
    from nvidia import dali
    from nvidia.dali import ops
    from nvidia.dali import pipeline
    _dali_available = True
except ImportError:
    class pipeline(object):
        Pipeline = object
        pass
    _dali_available = False

from chainer.backends import cuda
import ctypes


def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return x, x


class DaliPipelineTrain(pipeline.Pipeline):

    def __init__(self, file_list, file_root, crop_size,
                 batch_size, num_threads, device_id,
                 random_shuffle=True, seed=-1, mean=None, std=None,
                 num_samples=None):
        super(DaliPipelineTrain, self).__init__(batch_size, num_threads,
                                                device_id, seed=seed)
        crop_size = _pair(crop_size)
        if mean is None:
            mean = (0.485 * 255, 0.456 * 255, 0.406 * 255)
        if std is None:
            std = (0.229 * 255, 0.224 * 255, 0.225 * 255)
        if num_samples is None:
            initial_fill = 4096
        else:
            initial_fill = min(4096, num_samples)
        self.loader = ops.FileReader(file_root=file_root, file_list=file_list,
                                     random_shuffle=random_shuffle,
                                     initial_fill=initial_fill)
        self.decode = ops.HostDecoder()
        self.resize = ops.Resize(device="gpu", resize_a=256, resize_b=256,
                                 warp_resize=True)
        # self.hue = ops.Hue(device="gpu")
        # self.bright = ops.Brightness(device="gpu")
        # self.cntrst = ops.Contrast(device="gpu")
        # self.rotate = ops.Rotate(device="gpu")
        # self.jitter = ops.Jitter(device="gpu")
        random_area = (crop_size[0] / 256) * (crop_size[1] / 256)
        random_area = _pair(random_area)
        random_aspect_ratio = _pair(1.0)
        self.rrcrop = ops.RandomResizedCrop(
            device="gpu", size=crop_size, random_area=random_area,
            random_aspect_ratio=random_aspect_ratio)
        self.cmnorm = ops.CropMirrorNormalize(
            device="gpu", crop=crop_size, mean=mean, std=std)
        self.coin = ops.CoinFlip(probability=0.5)

    def define_graph(self):
        jpegs, labels = self.loader()
        images = self.decode(jpegs)
        images = self.resize(images.gpu())
        # images = self.hue(images, hue=ops.Uniform(range=(-3.0, 3.0))())
        # images = self.bright(images,
        #                      brightness=ops.Uniform(range=(0.9, 1.1))())
        # images = self.cntrst(images,
        #                      contrast=ops.Uniform(range=(0.9, 1.1))())
        # images = self.rotate(images,
        #                      angle=ops.Uniform(range=(-5.0, 5.0))())
        # images = self.jitter(images)
        images = self.rrcrop(images)
        images = self.cmnorm(images, mirror=self.coin())
        return [images, labels]


class DaliPipelineVal(pipeline.Pipeline):

    def __init__(self, file_list, file_root, crop_size,
                 batch_size, num_threads, device_id,
                 random_shuffle=False, seed=-1, mean=None, std=None,
                 num_samples=None):
        super(DaliPipelineVal, self).__init__(batch_size, num_threads,
                                              device_id, seed=seed)
        crop_size = _pair(crop_size)
        if mean is None:
            mean = (0.485 * 255, 0.456 * 255, 0.406 * 255)
        if std is None:
            std = (0.229 * 255, 0.224 * 255, 0.225 * 255)
        if num_samples is None:
            initial_fill = 512
        else:
            initial_fill = min(512, num_samples)
        self.loader = ops.FileReader(file_root=file_root, file_list=file_list,
                                     random_shuffle=random_shuffle,
                                     initial_fill=initial_fill)
        self.decode = ops.HostDecoder()
        self.resize = ops.Resize(device="gpu", resize_a=256, resize_b=256,
                                 warp_resize=True)
        self.cmnorm = ops.CropMirrorNormalize(
            device="gpu", crop=crop_size, mean=mean, std=std)

    def define_graph(self):
        jpegs, labels = self.loader()
        images = self.decode(jpegs)
        images = self.resize(images.gpu())
        images = self.cmnorm(images)
        return [images, labels]


class DaliConverter(object):

    def __init__(self, mean, crop_size):
        self.mean = mean
        self.crop_size = crop_size

        ch_mean = np.average(mean, axis=(1, 2))
        perturbation = (mean - ch_mean.reshape(3, 1, 1)) / 255.0
        perturbation = perturbation[:3, :crop_size, :crop_size].astype(
            np.float32)
        self.perturbation = perturbation.reshape(1, 3, crop_size, crop_size)

    def __call__(self, inputs, device=None):
        """Convert DALI arrays to Numpy/CuPy arrays"""

        xp = cuda.get_array_module(self.perturbation)
        if xp is not cuda.cupy:
            self.perturbation = cuda.to_gpu(self.perturbation, device)

        outputs = []
        for i in range(len(inputs)):
            x = inputs[i].as_tensor()
            if (isinstance(x, dali.backend_impl.TensorCPU)):
                x = np.array(x)
                if x.ndim == 2 and x.shape[1] == 1:
                    x = x.squeeze(axis=1)
                if device is not None and device >= 0:
                    x = cuda.to_gpu(x, device)
            elif (isinstance(x, dali.backend_impl.TensorGPU)):
                x_cupy = cuda.cupy.empty(shape=x.shape(), dtype=x.dtype())
                # Synchronization is necessary here to avoid data corruption
                # because DALI and CuPy will use different CUDA streams.
                cuda.cupy.cuda.runtime.deviceSynchronize()
                # copy data from DALI array to CuPy array
                x.copy_to_external(ctypes.c_void_p(x_cupy.data.ptr))
                cuda.cupy.cuda.runtime.deviceSynchronize()
                x = x_cupy
                if self.perturbation is not None:
                    x = x - self.perturbation
                if device is not None and device < 0:
                    x = cuda.to_cpu(x)
            else:
                raise ValueError('Unexpected object')
            outputs.append(x)
        return tuple(outputs)


# Check Python version if it supports multiprocessing.set_start_method,
# which was introduced in Python 3.4
major, minor, _, _, _ = sys.version_info
if major <= 2 or (major == 3 and minor < 4):
    sys.stderr.write("Error: ImageNet example uses "
                     "chainer.iterators.MultiprocessIterator, "
                     "which works only with Python >= 3.4. \n"
                     "For more details, see "
                     "http://chainermn.readthedocs.io/en/master/"
                     "tutorial/tips_faqs.html#using-multiprocessiterator\n")
    exit(-1)


class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root, mean, crop_size, random=True):
        self.base = chainer.datasets.LabeledImageDataset(path, root)
        self.mean = mean.astype('f')
        self.crop_size = crop_size
        self.random = random

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
        # It reads the i-th image/label pair and return a preprocessed image.
        # It applies following preprocesses:
        #     - Cropping (random or center rectangular)
        #     - Random flip
        #     - Scaling to [0, 1] value
        crop_size = self.crop_size

        image, label = self.base[i]
        _, h, w = image.shape

        if self.random:
            # Randomly crop a region and flip the image
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
            if random.randint(0, 1):
                image = image[:, :, ::-1]
        else:
            # Crop the center
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size

        image = image[:, top:bottom, left:right]
        image -= self.mean[:, top:bottom, left:right]
        image *= (1.0 / 255.0)  # Scale to [0, 1]
        return image, label


# chainermn.create_multi_node_evaluator can be also used with user customized
# evaluator classes that inherit chainer.training.extensions.Evaluator.
class TestModeEvaluator(extensions.Evaluator):

    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret


def main():
    # Check if GPU is available
    # (ImageNet example does not support CPU execution)
    if not chainer.cuda.available:
        raise RuntimeError("ImageNet requires GPU support.")

    archs = {
        'alex': alex.Alex,
        'googlenet': googlenet.GoogLeNet,
        'googlenetbn': googlenetbn.GoogLeNetBN,
        'nin': nin.NIN,
        'resnet50': resnet50.ResNet50,
    }

    parser = argparse.ArgumentParser(
        description='Learning convnet from ILSVRC2012 dataset')
    parser.add_argument('train', help='Path to training image-label list file')
    parser.add_argument('val', help='Path to validation image-label list file')
    parser.add_argument('--arch', '-a', choices=archs.keys(), default='nin',
                        help='Convnet architecture')
    parser.add_argument('--batchsize', '-B', type=int, default=32,
                        help='Learning minibatch size')
    parser.add_argument('--epoch', '-E', type=int, default=10,
                        help='Number of epochs to train')
    parser.add_argument('--initmodel',
                        help='Initialize the model from given file')
    parser.add_argument('--loaderjob', '-j', type=int,
                        help='Number of parallel data loading processes')
    parser.add_argument('--mean', '-m', default='mean.npy',
                        help='Mean file (computed by compute_mean.py)')
    parser.add_argument('--resume', '-r', default='',
                        help='Initialize the trainer from given file')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--root', '-R', default='.',
                        help='Root directory path of image files')
    parser.add_argument('--val_batchsize', '-b', type=int, default=250,
                        help='Validation minibatch size')
    parser.add_argument('--val_root', default='.')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--communicator', default='hierarchical')
    parser.set_defaults(test=False)
    parser.add_argument('--dali', action='store_true')
    parser.set_defaults(dali=False)
    parser.add_argument('--tempdir', default='temp')
    args = parser.parse_args()

    # Prepare ChainerMN communicator.
    comm = chainermn.create_communicator(args.communicator)
    device = comm.intra_rank

    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        print('Using {} communicator'.format(args.communicator))
        print('Using {} arch'.format(args.arch))
        print('Num Minibatch-size: {}'.format(args.batchsize))
        print('Num epoch: {}'.format(args.epoch))
        print('==========================================')

    model = archs[args.arch]()
    if args.initmodel:
        print('Load model from', args.initmodel)
        chainer.serializers.load_npz(args.initmodel, model)

    chainer.cuda.get_device_from_id(device).use()  # Make the GPU current
    model.to_gpu()

    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.
    mean = np.load(args.mean)
    if comm.rank == 0:
        if args.dali:
            def read_pairs(path):
                pairs = []
                with open(path) as f:
                    for line in f:
                        path, label = line.split()
                        label = int(label)
                        pairs.append((path, label))
                return pairs

            train = read_pairs(args.train)
            val = read_pairs(args.val)
        else:
            train = PreprocessedDataset(args.train, args.root, mean,
                                        model.insize)
            val = PreprocessedDataset(
                args.val, args.val_root, mean, model.insize, False)
    else:
        train = None
        val = None
    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    val = chainermn.scatter_dataset(val, comm)

    if args.dali:
        if not _dali_available:
            raise RuntimeError('DALI seems not available on your system.')
        num_threads = args.loaderjob
        if num_threads is None or num_threads <= 0:
            num_threads = 1

        def write_pairs(pairs, name):
            dataset = args.tempdir + "/" + name + "." + str(comm.size) + "_" + str(comm.rank)
            with open(dataset, mode='w') as f:
                for i in range(len(pairs)):
                    path, label = pairs[i]
                    f.write("%s %d\n" % (path, label))
            return dataset

        my_train = write_pairs(train, "train")
        my_val = write_pairs(val, "val")

        # Setup DALI pipelines
        ch_mean = np.average(mean, axis=(1, 2))
        ch_std = (255.0, 255.0, 255.0)
        train_pipe = DaliPipelineTrain(
            my_train, args.root, model.insize, args.batchsize,
            num_threads, device, random_shuffle=True,
            mean=ch_mean, std=ch_std, num_samples=len(train))
        val_pipe = DaliPipelineVal(
            my_val, args.val_root, model.insize, args.val_batchsize,
            num_threads, device, random_shuffle=False,
            mean=ch_mean, std=ch_std)
        train_iter = chainer.iterators.DaliIterator(train_pipe)
        val_iter = chainer.iterators.DaliIterator(val_pipe, repeat=False)
        # converter = dali_converter
        converter = DaliConverter(mean=mean, crop_size=model.insize)
    else:
        # We need to change the start method of multiprocessing module if we are
        # using InfiniBand and MultiprocessIterator. This is because processes
        # often crash when calling fork if they are using Infiniband.
        # (c.f., https://www.open-mpi.org/faq/?category=tuning#fork-warning )
        multiprocessing.set_start_method('forkserver')
        train_iter = chainer.iterators.MultiprocessIterator(
            train, args.batchsize, n_processes=args.loaderjob)
        val_iter = chainer.iterators.MultiprocessIterator(
            val, args.val_batchsize, repeat=False, n_processes=args.loaderjob)
        converter = dataset.concat_examples

    # Create a multi node optimizer from a standard Chainer optimizer.
    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9), comm)
    optimizer.setup(model)

    # Set up a trainer
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=converter, device=device)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), args.out)

    checkpoint_interval = (10, 'iteration') if args.test else (1, 'epoch')
    val_interval = (10, 'iteration') if args.test else (1, 'epoch')
    log_interval = (10, 'iteration') if args.test else (1, 'epoch')

    checkpointer = chainermn.create_multi_node_checkpointer(
        name='imagenet-example', comm=comm)
    checkpointer.maybe_load(trainer, optimizer)
    trainer.extend(checkpointer, trigger=checkpoint_interval)

    # Create a multi node evaluator from an evaluator.
    evaluator = TestModeEvaluator(
        val_iter, model, converter=converter, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=val_interval)

    # Some display and output extensions are necessary only for one worker.
    # (Otherwise, there would just be repeated outputs.)
    if comm.rank == 0:
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        trainer.extend(extensions.PrintReport([
            'epoch', 'iteration', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy', 'lr'
        ]), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
