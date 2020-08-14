import os
import tensorflow as tf
import abc


class Dataset(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def dataset_parser(self, serialized_example):
        pass

    @abc.abstractmethod
    def input_fn(self):
        pass


if __name__=='__main__':
    import time
    def timeit(batches=100):
        init_start = time.time()
        tf.logging.info('start')
        num_workers=1
        training=True
        image_size=224
        frame_size=20
        data_dir=''
        file_pattern='part-r-00001'
        batch_size=2
        task_index=0
        DATASET = Dataset(num_workers, training, image_size, frame_size, data_dir, file_pattern, batch_size, task_index)
        dataset = DATASET.input_fn()
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        sess = tf.Session()
    
        start = time.time()
        for i in range(batches):
            value = sess.run(next_element)
            #print(value)
            print(value[0].shape, value[1].shape)
        end = time.time()
        duration = end-start
        print("{} batches: {} s".format(batches, duration))
        print("{:0.5f} Images/s".format(batch_size*batches/duration))
        print("Total time: {}s".format(end-init_start))

    timeit(1)
