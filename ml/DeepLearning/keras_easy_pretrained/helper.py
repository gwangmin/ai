'''
에러 날 수가 있는데 nomkl 설치하면 해결됩니다.
conda install -c anaconda nomkl
'''

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
import numpy as np


class PreTrainedModelHelper:
    '''
    Easy use pre-trained model
    
    Available models(13):
        vgg16
        vgg19
        resnet50
        xception
        mobilenet
        mobilenet_v2
        nasnet_mobile
        nasnet_large
        inception_v3
        inception_resnet_v2
        densenet121
        densenet169
        densenet201
    '''
    def __init__(self, model_name, include_last_fc=True, input_shape=None):
        '''
        Load pre-trained model.
        
        model_name: one of available models.
        include_last_fc: whether to include the fully-connected layer at the top of the network. default True
        input_shape: : Transfer learning input shape. Default None. https://keras.io/applications/
        '''
        self.name = model_name
        # vgg16
        if model_name == 'vgg16':
            import keras.applications.vgg16 as vgg16
            self.lib = vgg16
            if input_shape == None:
                self.input_size = (224, 224)
                self.model = vgg16.VGG16(include_top=include_last_fc)
            else:
                self.input_size = (input_shape[0], input_shape[1])
                self.model = vgg16.VGG16(include_top=include_last_fc, input_shape=input_shape)
        # vgg 19
        elif model_name == 'vgg19':
            import keras.applications.vgg19 as vgg19
            self.lib = vgg19
            if input_shape == None:
                self.input_size = (224, 224)
                self.model = vgg19.VGG19(include_top=include_last_fc)
            else:
                self.input_size = (input_shape[0], input_shape[1])
                self.model = vgg19.VGG19(include_top=include_last_fc, input_shape=input_shape)
        # resnet50
        elif model_name == 'resnet50':
            import keras.applications.resnet50 as resnet50
            self.lib = resnet50
            if input_shape == None:
                self.input_size = (224, 224)
                self.model = resnet50.ResNet50(include_top=include_last_fc)
            else:
                self.input_size = (input_shape[0], input_shape[1])
                self.model = resnet50.ResNet50(include_top=include_last_fc, input_shape=input_shape)
        # xception
        elif model_name == 'xception':
            import keras.applications.xception as xception
            self.lib = xception
            if input_shape == None:
                self.input_size = (299, 299)
                self.model = xception.Xception(include_top=include_last_fc)
            else:
                self.input_size = (input_shape[0], input_shape[1])
                self.model = xception.Xception(include_top=include_last_fc, input_shape=input_shape)
        # densenet121
        elif model_name == 'densenet121':
            import keras.applications.densenet as densenet
            self.lib = densenet
            if input_shape == None:
                self.input_size = (224, 224)
                self.model = densenet.DenseNet121(include_top=include_last_fc)
            else:
                self.input_size = (input_shape[0], input_shape[1])
                self.model = densenet.DenseNet121(include_top=include_last_fc, input_shape=input_shape)
        # densenet169
        elif model_name == 'densenet169':
            import keras.applications.densenet as densenet
            self.lib = densenet
            if input_shape == None:
                self.input_size = (224, 224)
                self.model = densenet.DenseNet169(include_top=include_last_fc)
            else:
                self.input_size = (input_shape[0], input_shape[1])
                self.model = densenet.DenseNet169(include_top=include_last_fc, input_shape=input_shape)
        # densenet201
        elif model_name == 'densenet201':
            import keras.applications.densenet as densenet
            self.lib = densenet
            if input_shape == None:
                self.input_size = (224, 224)
                self.model = densenet.DenseNet201(include_top=include_last_fc)
            else:
                self.input_size = (input_shape[0], input_shape[1])
                self.model = densenet.DenseNet201(include_top=include_last_fc, input_shape=input_shape)
        # inceptionResnetV2
        elif model_name == 'inception_resnet_v2':
            import keras.applications.inception_resnet_v2 as inception_resnet_v2
            self.lib = inception_resnet_v2
            if input_shape == None:
                self.input_size = (299, 299)
                self.model = self.lib.InceptionResNetV2(include_top=include_last_fc)
            else:
                self.input_size = (input_shape[0], input_shape[1])
                self.model = self.lib.InceptionResNetV2(include_top=include_last_fc, input_shape=input_shape)
        # inceptionV3
        elif model_name == 'inception_v3':
            import keras.applications.inception_v3 as inception_v3
            self.lib = inception_v3
            if input_shape == None:
                self.input_size = (299, 299)
                self.model = self.lib.InceptionV3(include_top=include_last_fc)
            else:
                self.input_size = (input_shape[0], input_shape[1])
                self.model = self.lib.InceptionV3(include_top=include_last_fc, input_shape=input_shape)
        # nasnet mobile
        elif model_name == 'nasnet_mobile':
            import keras.applications.nasnet as nasnet
            self.lib = nasnet
            if input_shape == None:
                self.input_size = (224, 224)
                self.model = self.lib.NASNetMobile(include_top=include_last_fc)
            else:
                self.input_size = (input_shape[0], input_shape[1])
                self.model = self.lib.NASNetMobile(include_top=include_last_fc, input_shape=input_shape)
        # nasnet large
        elif model_name == 'nasnet_large':
            import keras.applications.nasnet as nasnet
            self.lib = nasnet
            if input_shape == None:
                self.input_size = (331, 331)
                self.model = self.lib.NASNetLarge(include_top=include_last_fc)
            else:
                self.input_size = (input_shape[0], input_shape[1])
                self.model = self.lib.NASNetLarge(include_top=include_last_fc, input_shape=input_shape)
        # mobilenet
        elif model_name == 'mobilenet':
            import keras.applications.mobilenet as mobilenet
            self.lib = mobilenet
            if input_shape == None:
                self.input_size = (224, 224)
                self.model = self.lib.MobileNet(include_top=include_last_fc)
            else:
                self.input_size = (input_shape[0], input_shape[1])
                self.model = self.lib.MobileNet(include_top=include_last_fc, input_shape=input_shape)
        # mobilenet v2
        elif model_name == 'mobilenet_v2':
            import keras.applications.mobilenet_v2 as mobilenet_v2
            self.lib = mobilenet_v2
            if input_shape == None:
                self.input_size = (224, 224)
                self.model = self.lib.MobileNetV2(include_top=include_last_fc)
            else:
                self.input_size = (input_shape[0], input_shape[1])
                self.model = self.lib.MobileNetV2(include_top=include_last_fc, input_shape=input_shape)
    
    def add_layers(self, layers):
        '''
        Add layers for transfer learning.
        
        layers: list of layes to added.
        '''
        last = self.model.output
        for layer in layers:
            last = layer(last)
        self.model = Model(self.model.input, last)
    
    def freeze(self, n):
        '''
        Freeze some layers
        
        n: number of frozen layers.
        '''
        self.model.trainable = True
        for layer in self.model.layers[:n]:
            layer.trainable = False
    
    def img_to_np(self, path, size):
        '''
        Load image and convert to numpy.ndarray
        
        path: image path
        size: image size (height, width)
        '''
        arr = img_to_array(load_img(path, target_size=size))
        return arr
    
    def img_to_input(self, img):
        '''
        Add batch dim
        
        img: image
        '''
        x = img.reshape((1,img.shape[0],img.shape[1],img.shape[2]))
        return x
    
    def predict(self, path, transfered = False):
        '''
        Inference one image using pre-trained model.
        
        path: input image path
        transfered: whether transfer learning. default False.
        '''
        # Load the image
        img = self.img_to_np(path, self.input_size)
        # Reshape to batch
        img = self.img_to_input(img)
        # Preprocessing and predict
        pred = self.model.predict(self.lib.preprocess_input(img))
        if transfered == False:
            # Decode predictions
            result = self.lib.decode_predictions(pred)
        else:
            result = pred
        return result
    
    def summary(self):
        '''
        Show model summary
        '''
        return self.model.summary()


# test(example)
def load_pretrained():
    m = PreTrainedModelHelper('mobilenet_v2')
    print(m.predict('./mug.jpg'))
    print(m.summary())


def transfer():
    # create transfer model
    from keras.layers import GlobalAveragePooling2D, Dense
    m = PreTrainedModelHelper('mobilenet_v2', False, (224,224,3))
    m.add_layers([GlobalAveragePooling2D(), Dense(3, activation='softmax')])
    m.freeze(3)
    m.model.compile(loss='categorical_crossentropy', optimizer='adam')
    # training
    img = m.lib.preprocess_input(m.img_to_input(m.img_to_np('./mug.jpg', m.input_size)))
    y = np.array([[1,0,0]])
    m.model.fit(img, y)
    # using
    print(m.predict('./mug.jpg', True))
    print(m.summary())


if __name__ == '__main__':
#    load_pretrained()
    transfer()
