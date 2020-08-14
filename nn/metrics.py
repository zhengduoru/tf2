from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import confusion_matrix
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
import tensorflow as tf 


class Metrics_tf2(object):

    def __init__(self):
        self.dic = dict()

    def _acc(self):
        return tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')

    def _mean(self):
        return tf.keras.metrics.Mean(name='mean')

    def reset_states(self):
        for i in self.dic.values():
            i.reset_states()

    def result(self):
        return {key: value.result().numpy() for key, value in self.dic.items()}




class Metrics(object):

    def __init__(self):
        pass

    def _create_local(self, name, shape, collections=None, validate_shape=True,
                      dtype=dtypes.float32):
        """Creates a new local variable.
    
        Args:
          name: The name of the new or existing variable.
          shape: Shape of the new or existing variable.
          collections: A list of collection names to which the Variable will be added.
          validate_shape: Whether to validate the shape of the variable.
          dtype: Data type of the variables.
    
        Returns:
          The created variable.
        """
        # Make sure local variables are added to tf.GraphKeys.LOCAL_VARIABLES
        collections = list(collections or [])
        collections += [ops.GraphKeys.LOCAL_VARIABLES]
        return variable_scope.variable(
            lambda: array_ops.zeros(shape, dtype=dtype),
            name=name,
            trainable=False,
            collections=collections,
            validate_shape=validate_shape)

    def streaming_confusion_matrix_single_label(self, labels, logits, num_label_classes, weights=None):
        """Calculate a streaming confusion matrix.
    
        Calculates a confusion matrix. For estimation over a stream of data,
        the function creates an  `update_op` operation.
    
        Args:
          labels: A `Tensor` of ground truth labels with shape [batch size] and of
            type `int32` or `int64`. The tensor will be flattened if its rank > 1.
          logits: A `Tensor` of prediction results for semantic labels, whose
            shape is [batch size] and type `int32` or `int64`. The tensor will be
            flattened if its rank > 1.
          num_label_classes: The possible number of labels the prediction task can
            have. This value must be provided, since a confusion matrix of
            dimension = [num_label_classes, num_label_classes] will be allocated.
          weights: Optional `Tensor` whose rank is either 0, or the same rank as
            `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
            be either `1`, or the same as the corresponding `labels` dimension).
    
        Returns:
          total_cm: A `Tensor` representing the confusion matrix.
          update_op: An operation that increments the confusion matrix.
        """
        # Local variable to accumulate the logits in the confusion matrix.
        total_cm = _create_local(
            'total_confusion_matrix',
            shape=[num_label_classes, num_label_classes],
            dtype=dtypes.float64)
    
        # Cast the type to int64 required by confusion_matrix_ops.
        logits = math_ops.to_int64(logits)
        labels = math_ops.to_int64(labels)
        num_label_classes = math_ops.to_int64(num_label_classes)
    
        # Flatten the input if its rank > 1.
        if logits.get_shape().ndims > 1:
            logits = array_ops.reshape(logits, [-1])
    
        if labels.get_shape().ndims > 1:
            labels = array_ops.reshape(labels, [-1])
    
        if (weights is not None) and (weights.get_shape().ndims > 1):
            weights = array_ops.reshape(weights, [-1])
    
        # Accumulate the prediction to current confusion matrix.
        current_cm = confusion_matrix.confusion_matrix(
            labels, logits, num_label_classes, weights=weights, dtype=dtypes.float64)
        update_op = state_ops.assign_add(total_cm, current_cm)
        return total_cm, update_op

    def streaming_confusion_matrix_multi_label(self, labels, logits, num_label_classes, weights=None):
        """Calculate a streaming confusion matrix.
    
        Calculates a confusion matrix. For estimation over a stream of data,
        the function creates an  `update_op` operation.
    
        Args:
          labels: A `Tensor` of ground truth labels with shape [batch size] and of
            type `int32` or `int64`. The tensor will be flattened if its rank > 1.
          logits: A `Tensor` of prediction results for semantic labels, whose
            shape is [batch size] and type `int32` or `int64`. The tensor will be
            flattened if its rank > 1.
          num_label_classes: The possible number of labels the prediction task can
            have. This value must be provided, since a confusion matrix of
            dimension = [num_label_classes, num_label_classes] will be allocated.
          weights: Optional `Tensor` whose rank is either 0, or the same rank as
            `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
            be either `1`, or the same as the corresponding `labels` dimension).
    
        Returns:
          total_cm: A `Tensor` representing the confusion matrix.
          update_op: An operation that increments the confusion matrix.
        """
        # Local variable to accumulate the logits in the confusion matrix.
        total_cm = self._create_local(
            'total_confusion_matrix',
            shape=[num_label_classes, 2, 2],
            dtype=dtypes.float64)
    
        # Cast the type to int64 required by confusion_matrix_ops.
        logits = math_ops.to_int64(logits)
        labels = math_ops.to_int64(labels)
        num_label_classes = math_ops.to_int64(num_label_classes)
    
        # Flatten the input if its rank > 1.
        if logits.get_shape().ndims > 1:
            logits = array_ops.reshape(logits, [-1])
    
        if labels.get_shape().ndims > 1:
            labels = array_ops.reshape(labels, [-1])
    
        if (weights is not None) and (weights.get_shape().ndims > 1):
            weights = array_ops.reshape(weights, [-1])
    
        # Accumulate the prediction to current confusion matrix.
        current_cm = tf.stack(
                [confusion_matrix.confusion_matrix(labels[:, i], logits[:, i], 2, weights=weights, dtype=dtypes.float64)
                for i in range(num_label_classes)],
                axis=0)
        #current_cm = confusion_matrix.confusion_matrix(
        #    labels, logits, num_label_classes, weights=weights, dtype=dtypes.float64)
        update_op = state_ops.assign_add(total_cm, current_cm)
        return total_cm, update_op

    def precisions_multi_label(self, labels, logits, num_label_classes, weights=None, 
            metrics_collections=None, updates_collections=None, name=None):
        """Calculates the precisions of the per-class.
    
            For estimation of the metric over a stream of data, the function creates an
            `update_op` operation that updates these variables and returns the
            `per_class_precision`.
    
            If `weights` is `None`, weights default to 1. Use weights of 0 to mask values.
    
            Args:
              labels: A `Tensor` of ground truth labels with shape [batch size] and of
                type `int32` or `int64`. The tensor will be flattened if its rank > 1.
              logits: A `Tensor` of prediction results for semantic labels, whose
                shape is [batch size] and type `int32` or `int64`. The tensor will be
                flattened if its rank > 1.
              num_label_classes: The possible number of labels the prediction task can
                have. This value must be provided, since a confusion matrix of
                dimension = [num_label_classes, num_label_classes] will be allocated.
              weights: Optional `Tensor` whose rank is either 0, or the same rank as
                `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
                be either `1`, or the same as the corresponding `labels` dimension).
              metrics_collections: An optional list of collections that
                `mean_per_class_accuracy'
                should be added to.
              updates_collections: An optional list of collections `update_op` should be
                added to.
              name: An optional variable_scope name.
    
            Returns:
              per_class_precision: A `Tensor` representing the precisions of the per-class.
              update_op: An operation that increments the confusion matrix.
    
            Raises:
              ValueError: If `logits` and `labels` have mismatched shapes, or if
                `weights` is not `None` and its shape doesn't match `logits`, or if
                either `metrics_collections` or `updates_collections` are not a list or
                tuple.
            """
        with variable_scope.variable_scope(name, 'per_class_precision',
                                           (logits, labels, weights)):
            # Check if shape is compatible.
            logits.get_shape().assert_is_compatible_with(labels.get_shape())
    
            total_cm, update_op = self.streaming_confusion_matrix_multi_label(
                labels, logits, num_label_classes, weights=weights)
    
            def compute_per_class_precision(name):
                """Compute the mean per class accuracy via the confusion matrix."""
                tp = total_cm[:,1,1]
                tp_and_fp = total_cm[:,1,1] + total_cm[:,0,1]
                return math_ops.div(tp, tp_and_fp+1e-8, name=name)
            per_class_precision_v = compute_per_class_precision('per_class_precision')
    
            if metrics_collections:
                ops.add_to_collections(metrics_collections, per_class_precision_v)
    
            if updates_collections:
                ops.add_to_collections(updates_collections, update_op)
    
            return per_class_precision_v, update_op
    
    def earth_mover_loss(self, labels, logits, metrics_collections=None, updates_collections=None, name=None):
        total_emd_loss = self._create_local('total_emd_loss',shape=[2],dtype=dtypes.float32)
        cdf_ytrue = math_ops.cumsum(labels, axis=-1)
        y_pred = tf.nn.softmax(logits, dim=-1)
        cdf_ypred = math_ops.cumsum(y_pred, axis=-1)
        current_emd_loss = math_ops.sqrt(math_ops.reduce_mean(math_ops.square(math_ops.abs(cdf_ytrue - cdf_ypred)), axis=-1))
        
        update_op = state_ops.assign_add(total_emd_loss, [math_ops.reduce_sum(current_emd_loss),labels.shape[0].value])
        emd_loss = math_ops.div(total_emd_loss[0],total_emd_loss[1])
        
        if metrics_collections:
            ops.add_to_collections(metrics_collections, emd_loss)
    
        if updates_collections:
            ops.add_to_collections(updates_collections, update_op)
        
        return emd_loss, update_op
    
    def pairwise_test(self, labels, logits, weights=None):
        total_cm = self._create_local('pairwise_test_numerator_denominator', shape=[2], dtype=dtypes.float32)
        if labels.get_shape().ndims == 1:
            labels_weighted_average = labels
            logits_weighted_average = logits
        if labels.get_shape().ndims == 2:
            batch = tf.shape(labels)[0]
            #logits = tf.nn.softmax(logits, dim=-1)
            labels_weighted_average = tf.reduce_sum(labels * weights, axis=1)
            logits_weighted_average = tf.reduce_sum(logits * weights, axis=1)
        broadcast_labels_weighted_average = tf.reshape(tf.tile(labels_weighted_average, [batch]), [batch,batch])
        broadcast_logits_weighted_average = tf.reshape(tf.tile(logits_weighted_average, [batch]), [batch,batch])
        self_constrast_labels = tf.subtract(broadcast_labels_weighted_average, tf.transpose(broadcast_labels_weighted_average))
        self_constrast_logits = tf.subtract(broadcast_logits_weighted_average, tf.transpose(broadcast_logits_weighted_average))
        one_self_constrast_labels = tf.cast(tf.greater_equal(self_constrast_labels, 0), tf.int32)
        one_self_constrast_logits = tf.cast(tf.greater_equal(self_constrast_logits, 0), tf.int32)
        band_one_self_constrast_labels = tf.linalg.band_part(one_self_constrast_labels, batch, 0)
        band_one_self_constrast_logits = tf.linalg.band_part(one_self_constrast_logits, batch, 0)
        numerator = tf.cast(tf.reduce_sum(tf.cast(tf.equal(band_one_self_constrast_labels, band_one_self_constrast_logits), tf.float32)), tf.float32) - tf.cast(tf.divide(tf.multiply(batch, tf.add(batch, 1)), 2), tf.float32)
        denominator = tf.cast(tf.divide(tf.multiply(batch, tf.subtract(batch, 1)), 2), tf.float32)  # C(x, 2)
        current_cm = tf.stack([numerator, denominator])
        update_op = state_ops.assign_add(total_cm, current_cm)
        #return tf.stack([total_cm[0], total_cm[1]]), update_op
        return tf.divide(total_cm[0], total_cm[1]), update_op

    def accuracy(self, labels, logits, weights=None):
        return tf.metrics.accuracy(labels, logits)

