import tensorflow as tf
import numpy as np
import collections
import math
import re
import json
import os
import abc

from settings import FLAGS
from models.choose_model import choose_model
from config.choose_builder import choose_builder
from config.choose_dataset import choose_dataset

class Builder(object):

    def __init__(self):
        self.EPOCHS = FLAGS.epochs
        self.strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))
        self.optimizer = self.builder_optimizer()
        with self.strategy.scope():
            self.model = self.model_fn()
        self.metric = self.metric_def()
        self.loss = self.loss_def()
        self.checkpoint, self.manager = self.build_checkpoint(self.optimizer, self.model)

    def get_assignment_map_from_checkpoint(self):
        pass

    def build_checkpoint(self, optimizer, model):
        checkpoint = tf.train.Checkpoint(epoch=tf.Variable(1), step=tf.Variable(1), optimizer=optimizer, model=model)
        manager = tf.train.CheckpointManager(checkpoint, FLAGS.logdir, max_to_keep=FLAGS.max_ckpt_files_to_keep)
        return checkpoint, manager

    def builder_optimizer(self):
        optimizer = tf.keras.optimizers.Adam()
        return optimizer

    def build_learning_rate(self):
        pass

    def features_and_labels_construct(self, features, labels, label_type):
        pass

    @abc.abstractmethod
    def loss_fn(self, features, logits, labels, endpoints):
        pass

    @abc.abstractmethod
    def metric_fn(self, features, logits, labels, loss, endpoints, weights=None):
        pass
        
    @abc.abstractmethod
    def export(self, features, logits, labels, endpoints):
        pass

    @abc.abstractmethod
    def head(self, features, logits, labels, endpoints, training):
        pass

    def model_fn(self):
        model = choose_model(FLAGS.model_name)
        return model

    @abc.abstractmethod
    @tf.function
    def train_step(self, features, labels):
        # forward 
        with tf.GradientTape() as tape:
            logits = self.model(features, training=True)
            loss_per_batch = self.loss_fn(features, logits, labels, dict())

        # backward
        gradients = tape.gradient(loss_per_batch, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # metrics
        self.metric_fn(features, logits, labels, loss_per_batch, dict())

        # checkpoint
        self.checkpoint.step.assign_add(1)

    @abc.abstractmethod
    @tf.function
    def test_step(self, features, labels):
        # forward
        logits = self.model(features, training=False)
        loss_per_batch = self.loss_fn(features, logits, labels, dict())

        # metrics
        self.metric_fn(features, logits, labels, loss_per_batch, dict())


def main(mode):
    # Define builder and dataset
    builder = choose_builder(FLAGS.builder_name)()
    train_ds = choose_dataset(FLAGS.data_name)().input_fn('train')
    test_ds = choose_dataset(FLAGS.data_name)().input_fn('eval')

    # Restore checkpoint
    builder.checkpoint.restore(builder.manager.latest_checkpoint)
    if builder.manager.latest_checkpoint:
        print("Restored from {}".format(builder.manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")


    if mode == 'train':
        # Training Loop #int(builder.checkpoint.epoch.numpy()), 
        for epoch in range(builder.EPOCHS):
            # Reset the metrics at the start of the next epoch
            builder.metric.reset_states()

            builder.checkpoint.epoch.assign_add(1)
            for images, labels in train_ds:
                builder.train_step(images, labels)
                if int(builder.checkpoint.step) % FLAGS.steps_saving_ckpt == 0:
                    save_path = builder.manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(builder.checkpoint.step), save_path))
            print('Epoch {} result {}'.format(int(builder.checkpoint.epoch), builder.metric.result()))


    if mode == 'eval':
        builder.metric.reset_states()
        for images, labels in test_ds:
            builder.test_step(images, labels)
        #print('result: {}'.format(builder.metric.result()))
        print(builder.metric.result())




if __name__ == "__main__":
    main(FLAGS.mode)
