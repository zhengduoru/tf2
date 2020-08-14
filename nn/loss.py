import tensorflow as tf

class Loss_tf2(object):

    def __init__(self):
        pass

    def _SparseCategoricalCrossentropy(self):
        return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)




class Loss(object):

    def __init__(self):
        pass

    def softmax_cross_entropy(self, labels, logits, label_smoothing):
        return tf.losses.softmax_cross_entropy(
            logits=logits,
            onehot_labels=labels,
            label_smoothing=label_smoothing)

    def earth_mover_loss(self, labels, logits):
        cdf_ytrue = tf.cumsum(labels, axis=-1)
        y_pred = tf.nn.softmax(logits, dim=-1)
        cdf_ypred = tf.cumsum(y_pred, axis=-1)
        samplewise_emd = tf.sqrt(tf.reduce_mean(tf.square(tf.abs(cdf_ytrue - cdf_ypred)), axis=-1))
        return tf.reduce_mean(samplewise_emd)

    def mse(self, labels, logits):
        return tf.losses.mean_squared_error(labels, logits)

    def tripled_loss(self, fake_labels, logits, margin):
        return tf.contrib.losses.metric_learning.triplet_semihard_loss(fake_labels, logits, margin)

    def ls_regularization(self, exclude_variables=['BatchNorm', 'batchnorm', 'batch_normalization']):
        return tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if all(exclude_variable not in v.name for exclude_variable in exclude_variables)])
