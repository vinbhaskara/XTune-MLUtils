import bcolz
def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]



# Criag Glastonbury 23rd on PB LB (0.5 logloss) did a preprocessing of normalizing the RGB histogram with
# Obvly it gave him great results than us (~1 logloss) :/
import cv2

def normalized(rgb):
    norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)
    b=rgb[:,:,0]
    g=rgb[:,:,1]
    r=rgb[:,:,2]
    norm[:,:,0]=cv2.equalizeHist(b)
    norm[:,:,1]=cv2.equalizeHist(g)
    norm[:,:,2]=cv2.equalizeHist(r)
    return norm

def get_im_cv2(path):
    img = cv2.imread(path,1)
    
    # For color historgram
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # Sharpen
    output_3 = cv2.filter2D(img_output, -1, kernel_sharpen_3)
    
    # Reduce to manageable size
    resized = cv2.resize(output_3, (224, 224), interpolation = cv2.INTER_LINEAR)
    return resized


# Whenever using pretrained models, please be sure to do the necessary preprocessing on the inputs
# A lot of tutorials online for transfer learning miss this very important step. 
# Example for VGG 16 in Keras (it's got a readymade function, use it!):
from keras.applications.vgg16 import preprocess_input # Input preprocessing for VGG16 pretrained model