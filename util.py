import numpy as np
import cv2
import glob 
from PIL import Image

"""
data_dir = "puddle_test/data"
label_dir = "puddle_test/label"
data_list = glob.glob(data_dir + "/*.png")
label_list = glob.glob(label_dir + "/*.png")
sigma = 150
size = (500,300)

img = cv2.imread(data_list[0], 1)
label = Image.open(label_list[0])
label = np.array(label)
img = cv2.resize(img, size)
label = cv2.resize(label, size)
"""

#all label data should be binarized

def gauss_only_puddle(img, label, sigma=1):
    noise = np.random.normal(0, sigma, np.shape(img))
    noise[label==0] = 0
    img = img +  noise
    img[img > 255] = 255
    img[img < 0] = 0
    return img.astype(np.uint8)

def gauss_around_puddle(img, label, sigma=1, k=(3,3)):
    noise = np.random.normal(0, sigma, np.shape(img))
    edges = cv2.Canny(label, 130, 285, L2gradient=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, k)
    dst = cv2.dilate(edges, kernel)
    noise[dst==0] = 0
    img = img +  noise
    img[img > 255] = 255
    img[img < 0] = 0
    
    return img.astype(np.uint8)

def gauss_all_img(img, label, sigma=1, ):
    noise = np.random.normal(0, sigma, np.shape(img))
    img = img +  noise
    img[img > 255] = 255
    img[img < 0] = 0
    
    return img.astype(np.uint8)

def eval(pred, label, thresh=0.9, eps=1e-9):
    tp = len(np.where((pred>=255*thresh) & (label==255))[0])
    tn = len(np.where((pred<=255*thresh) & (label==0))[0])
    fp = len(np.where((pred<=255*thresh) & (label==255))[0])
    fn = len(np.where((pred>=255*thresh) & (label==0))[0])
        
    acc = (tp + tn) / (tp + fp + tn + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    f1 = (2 * prec * rec) / (prec + rec + eps)
    iou = tp / (tp + fp + fn + eps)
    return np.array(acc, prec, rec, iou, f1)
    

