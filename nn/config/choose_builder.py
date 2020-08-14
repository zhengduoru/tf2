import glob
import imp

class Choose_builder(object):
    def __init__(self):
        self.module = {}

    def __call__(self, builder_name):
        for filename in filter(lambda x:'builder' in x, glob.glob('./config/tasks/*/%s.py'%builder_name)):
            #self.module[filename.split('/')[-1].split('.')[0]] = imp.load_source(filename.split('/')[-1].split('.')[0], filename)
            self.module[builder_name] = imp.load_source(builder_name, filename)
        assert builder_name in self.module.keys()
        return self.module[builder_name].Builder_detail

choose_builder = Choose_builder()
