"""
 @Time    : 9/29/19 17:14
 @Author  : TaylorMei
 @Email   : mhy666@mail.dlut.edu.cn
 
 @Project : ICCV2019_MirrorNet
 @File    : infer.py
 @Function: predict mirror map.
 
"""
import numpy as np
import os
import time

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from config import msd_testing_root
from misc import check_mkdir, crf_refine
from mirrornet import MirrorNet
import glob
import matplotlib.pyplot as plt
from util import gauss_all_img, gauss_around_puddle, gauss_only_puddle, eval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch.set_device(device)

ckpt_path = './ckpt'
exp_name = 'MirrorNet'
args = {
    'snapshot': '160',
    'scale': 384,
    'crf': True
}

img_transform = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_test = {'MSD': msd_testing_root}

to_pil = transforms.ToPILImage()

data_dir = "/content/ICCV2019_MirrorNet/MSD/train/image"
label_dir = "/content/ICCV2019_MirrorNet/MSD/train/mask"   
    
            
def evaluate(net, img_path_list, label_path_list, sigma, thresh=0.8):
    net.eval()
    op = np.zeros(5)
    ap = np.zeros(5)
    an = np.zeros(5)
    
    op_dir = "/content/drive/MyDrive/noise_only_mirror/sigma_{}".format(sigma)
    ap_dir = "/content/drive/MyDrive/noise_around_mirror/sigma_{}".format(sigma)
    an_dir = "/content/drive/MyDrive/noise_only_mirror/sigma_{}".format(sigma)
    os.makedirs(op_dir, exist_ok=True)
    os.makedirs(ap_dir, exist_ok=True)
    os.makedirs(an_dir, exist_ok=True)
    
    
    with torch.no_grad():
        for idx,path in enumerate(img_path_list):
            #print("="*30 + "iteration" + str(idx+1) + "="*30)
            img = Image.open(path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            w, h = img.size
            img = np.array(img)            
            label = Image.open(label_path_list[idx])
            label = np.array(label)
            label[label==1] = 255
            

            only_puddle = gauss_only_puddle(img, label, sigma)
            around_puddle = gauss_around_puddle(img, label, sigma, k=(15,15))
            all_noise = gauss_all_img(img, label, sigma)
            
            only_puddle = Image.fromarray(only_puddle)
            around_puddle = Image.fromarray(around_puddle)
            all_noise = Image.fromarray(all_noise)
           
            
            img_var_op = Variable(img_transform(only_puddle).unsqueeze(0)).to(device)
            img_var_ap = Variable(img_transform(around_puddle).unsqueeze(0)).to(device)
            img_var_an = Variable(img_transform(all_noise).unsqueeze(0)).to(device)
            
            _, _, _, f_1_op = net(img_var_op)
            _, _, _, f_1_ap = net(img_var_ap)
            _, _, _, f_1_an = net(img_var_an)
            
            f_1_op = f_1_op.data.squeeze(0).to(device)
            f_1_ap = f_1_ap.data.squeeze(0).to(device)
            f_1_an = f_1_an.data.squeeze(0).to(device)
            
            
            f_1_op = np.array(transforms.Resize((h, w))(to_pil(f_1_op)))
            f_1_ap = np.array(transforms.Resize((h, w))(to_pil(f_1_ap)))
            f_1_an = np.array(transforms.Resize((h, w))(to_pil(f_1_an)))
            img = Image.fromarray(img)
            if args['crf']:
                f_1_op = crf_refine(np.array(img.convert('RGB')), f_1_op)
                f_1_ap = crf_refine(np.array(img.convert('RGB')), f_1_ap) 
                f_1_an = crf_refine(np.array(img.convert('RGB')), f_1_an)
            
            op_path = op_dir + "/" + os.path.basename(path) 
            ap_path = ap_dir + "/" + os.path.basename(path) 
            an_path = an_dir + "/" + os.path.basename(path) 
            
            only_puddle.save(op_path)
            around_puddle.save(ap_path)
            all_noise.save(an_path)
            
            op += eval(f_1_op, label, thresh)
            ap += eval(f_1_ap, label, thresh)
            an += eval(f_1_an, label, thresh)
    
    op = op/len(img_path_list)
    ap = ap/len(img_path_list)
    an = an/len(img_path_list)
    
    print("-------- Threshold Value = {} --------".format(thresh))
    print("------------ Sigma Value = {} --------".format(sigma))
    print("[only_puddle]   || accuracy : {}, precision : {}, recall : {}, iou : {}, f1 : {}".format(*op.tolist()))
    print("[around_puddle] || accuracy : {}, precision : {}, recall : {}, iou : {}, f1 : {}".format(*ap.tolist()))
    print("[all_image]     || accuracy : {}, precision : {}, recall : {}, iou : {}, f1 : {}".format(*an.tolist()))
    

def main():
    net = MirrorNet().to(device)
    net.load_state_dict(torch.load("/content/ICCV2019_MirrorNet/MirrorNet.pth"))
    print('Load trained model succeed!')
    print("="*100)
    
    img_path_list = glob.glob(data_dir + "/*.jpg")
    label_path_list = glob.glob(label_dir + "/*.png")
    img_path_list = sorted(img_path_list)
    label_path_list = sorted(label_path_list)
    
    thresh_list = [0.7, 0.8, 0.9]
    sigma_list = [x for x in range(30, 160, 30)]
    
        
    for thresh in thresh_list:
        for sigma in sigma_list:
            evaluate(net, img_path_list, label_path_list, sigma, thresh)

if __name__ == '__main__':
    main()
