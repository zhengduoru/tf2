import tensorflow as tf
from main import Builder
from loss import Loss
from metrics import Metrics_tf2
from loss import Loss_tf2

class Builder_detail(Builder):

    def features_and_labels_construct(self, features, labels, label_type):
        return features, labels

    def head(self, features, logits, labels, endpoints, training):
        return logits, endpoints


    def loss_def(self):
        loss = Loss_tf2()
        loss.SparseCategoricalCrossentropy = loss._SparseCategoricalCrossentropy()
        return loss
    def loss_fn(self, features, logits, labels, endpoints):
        return self.loss.SparseCategoricalCrossentropy(labels, logits)


    def metric_def(self):
        metric = Metrics_tf2()
        metric.dic['acc'] = metric._acc()
        metric.dic['mean'] = metric._mean()
        return metric
    def metric_fn(self, features, logits, labels, loss, endpoints, weights=None):
        self.metric.dic['acc'](labels, logits)
        self.metric.dic['mean'](loss)


    def export(self, features, logits, labels, endpoints):
        values, indices = tf.nn.top_k(logits, self.num_label_classes)
        predictions = {'classes': indices, 'probabilities': tf.nn.softmax(values)}
        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs={'classify': tf.estimator.export.PredictOutput(predictions)})

