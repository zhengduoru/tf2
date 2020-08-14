#try:
#    import efficientnet
#except:
#    pass
#try:
#    import s3dg
#except:
#    pass
#try:
#    import tcc
#except:
#    pass
#try:
#    import tcn
#except:
#    pass
#try:
#    from mobilenet import mobilenet_v3
#except:
#    pass
#try:
from models.mobilenet import mobilenet_v2
#except:
#    pass
try:
    from models import mnist
except:
    pass

class Choose_model(object):
    def __call__(self, model_name):
        if model_name == 'efficientnet-b3':
            return efficientnet.Model(logging=False, summary=False)
        if model_name == 's3dg':
            return s3dg.Model(logging=False, summary=False)
        if model_name == 'tcc':
            return tcc.Model(logging=False, summary=False)
        if model_name == 'tcn':
            return tcn.Model(logging=False, summary=False)
        if model_name == 'mobilenet-v3-large':
            return  mobilenet_v3.Model(model='large')
        if model_name == 'mobilenet-v3-small':
            return  mobilenet_v3.Model(model='small')
        if model_name == 'mobilenet-v2-140':
            return  mobilenet_v2.Model(model='mobilenet_v2_140')
        if model_name == 'mobilenet-v2-050':
            return  mobilenet_v2.Model(model='mobilenet_v2_050')
        if model_name == 'mnist':
            return  mnist.MyModel()

choose_model = Choose_model()

