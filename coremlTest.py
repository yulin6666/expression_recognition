from PIL import Image
import coremltools
import numpy as np
mlmodel = coremltools.models.MLModel('FER2013_mobilenetv2_privateTest_sim_13.mlmodel')
pil_img = Image.open('images/1_gray_32.jpg')

def softmax(x):
    orig_shape = x.shape
    if len(x.shape) > 1:
        tmp = np.max(x,axis=1)
        x -= tmp.reshape((x.shape[0],1))
        x = np.exp(x)
        tmp = np.sum(x, axis = 1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
#forward
out = mlmodel.predict({'data': pil_img})

outTensor = out['outTensor'];
print(outTensor.shape)
result =outTensor[0,:];
print(result)
print(softmax(result))

max = np.argmax(softmax(result))
print(max)

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
print("The Expression is %s" %str(class_names[max]))

#visualize the model output
