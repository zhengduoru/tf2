import glob
import imp

class Choose_dataset(object):
    def __init__(self):
        self.module = {}

    def __call__(self, dataset_name):
        for filename in filter(lambda x:'dataset' in x, glob.glob('./config/tasks/*/%s.py'%dataset_name)):
            #self.module[filename.split('/')[-1].split('.')[0]] = imp.load_source(filename.split('/')[-1].split('.')[0], filename)
            self.module[dataset_name] = imp.load_source(dataset_name, filename)
        assert dataset_name in self.module.keys()
        return self.module[dataset_name].Dataset#, self.module[dataset_name].build_image_serving_input_fn

choose_dataset = Choose_dataset()
