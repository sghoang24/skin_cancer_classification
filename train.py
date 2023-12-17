"""Train CIFAR-10 with TensorFlow2.0."""
import tensorflow as tf
from tensorflow.keras import layers

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm

from models import *
from utils import *

strategy = tf.distribute.MirroredStrategy()
print('DEVICES AVAILABLE: {}'.format(strategy.num_replicas_in_sync))
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

parser = argparse.ArgumentParser(description='TensorFlow2.0 CIFAR-10 Training')
parser.add_argument('--model', required=True, type=str, help='model type')
parser.add_argument('--lr', default=1e-1, type=float, help='learning rate')
parser.add_argument('--batch_size', default=BATCH_SIZE, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='number of training epoch')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--gpu', default=0, type=int, help='specify which gpu to be used')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
args.model = args.model.lower()
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

class Model():
    def __init__(self, model_type, decay_steps, num_classes=9):
        if 'lenet' in model_type:
            self.model = LeNet(num_classes)
        elif 'alexnet' in model_type:
            self.model = AlexNet(num_classes)
        elif 'vgg' in model_type:
            self.model = VGG(model_type, num_classes)
        elif 'resnet' in model_type:
            if 'se' in model_type:
                if 'preact' in model_type:
                    self.model = SEPreActResNet(model_type, num_classes)
                else:
                    self.model = SEResNet(model_type, num_classes)
            else:
                if 'preact' in model_type:
                    self.model = PreActResNet(model_type, num_classes)
                else:
                    self.model = ResNet(model_type, num_classes)
        elif 'densenet' in model_type:
            self.model = DenseNet(model_type, num_classes)
        elif 'mobilenet' in model_type:
            if 'v2' not in model_type:
                self.model = MobileNet(num_classes)
            else:
                self.model = MobileNetV2(num_classes)
        else:
            sys.exit(ValueError("{:s} is currently not supported.".format(model_type)))
        
        self.loss_object = tf.keras.losses.CategoricalCrossentropy()
        learning_rate_fn = tf.keras.experimental.CosineDecay(args.lr, decay_steps=decay_steps)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)
        self.weight_decay = 5e-4
        
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
        
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(images, training=True)
            # Cross-entropy loss
            ce_loss = self.loss_object(labels, predictions)
            # L2 loss(weight decay)
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.model.trainable_variables])
            loss = ce_loss + l2_loss*self.weight_decay
            
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)
        
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
        
    def train(self, train_ds, test_ds, epoch):
        best_acc = tf.Variable(0.0)
        curr_epoch = tf.Variable(0)  # start from epoch 0 or last checkpoint epoch
        ckpt_path = './checkpoints/{:s}/'.format(args.model)
        ckpt = tf.train.Checkpoint(curr_epoch=curr_epoch, best_acc=best_acc,
                                   optimizer=self.optimizer, model=self.model)
        manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)
        
        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint...')
            assert os.path.isdir(ckpt_path), 'Error: no checkpoint directory found!'

            # Restore the weights
            ckpt.restore(manager.latest_checkpoint)
        
        for e in tqdm(range(int(curr_epoch), epoch)):
            # Reset the metrics at the start of the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            for images, labels in train_ds:
                self.train_step(images, labels)
                
            for images, labels in test_ds:
                self.test_step(images, labels)

            template = 'Epoch {:0}, Loss: {:.4f}, Accuracy: {:.2f}%, Test Loss: {:.4f}, Test Accuracy: {:.2f}%'
            print (template.format(e+1,
                                   self.train_loss.result(),
                                   self.train_accuracy.result()*100,
                                   self.test_loss.result(),
                                   self.test_accuracy.result()*100))
            
            # Save checkpoint
            if self.test_accuracy.result() > best_acc:
                print('Saving...')
                if not os.path.isdir('./checkpoints/'):
                    os.mkdir('./checkpoints/')
                if not os.path.isdir(ckpt_path):
                    os.mkdir(ckpt_path)
                best_acc.assign(self.test_accuracy.result())
                curr_epoch.assign(e+1)
                manager.save()
    
    def predict(self, pred_ds, best):
        if best:
            ckpt_path = './checkpoints/{:s}/'.format(args.model)
            ckpt = tf.train.Checkpoint(model=self.model)
            manager = tf.train.CheckpointManager(ckpt, ckpt_path, max_to_keep=1)
            
            # Load checkpoint
            print('==> Resuming from checkpoint...')
            assert os.path.isdir(ckpt_path), 'Error: no checkpoint directory found!'
            ckpt.restore(manager.latest_checkpoint)
        
        self.test_accuracy.reset_states()
        for images, labels in pred_ds:
            self.test_step(images, labels)
        print ('Prediction Accuracy: {:.2f}%'.format(self.test_accuracy.result()*100))

def main():
    # Data
    print('==> Preparing data...')
    df = read_data()
    df = loading_and_resize(df)
    df = resize_image(df)
    df = data_augmentation(df)
    x_train, x_validate, x_test, y_train, y_validate, y_test = prepare_dataset(df)

    train_ds = dataset_generator(x_train, y_train, args.batch_size)
    validation_ds = tf.data.Dataset.from_tensor_slices((x_validate, y_validate)).\
            batch(args.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).\
            batch(args.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    class_names = ['pigmented benign keratosis', 'melanoma', 'vascular lesion',
                   'actinic keratosis', 'squamous cell carcinoma', 'basal cell carcinoma',
                   'seborrheic keratosis', 'dermatofibroma', 'nevus']
    decay_steps = int(args.epoch*len(x_train)/args.batch_size)
    
    # Train
    print('==> Building model...')
    model = Model(args.model, decay_steps)
    model.train(train_ds, validation_ds, args.epoch)
    
    # Evaluate
    model.predict(test_ds, best=True)

if __name__ == "__main__":
    main()
