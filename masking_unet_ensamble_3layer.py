import torch
from torch import jit
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import cv2, csv
import torchvision.transforms.functional as F
import torchvision.transforms as T
import numpy as np
import os,copy,math
import random
import sys, argparse
from torchvision import models
from scipy import stats as st
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import custom_dset, reconstruct_img

# see run.bat and test.bat for examples
parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-d","--directory",type = str, default = 'C:\\projects\\combine_msks\\combined\\')
parser.add_argument("-c","--class_name",type = str, default = 'new')
parser.add_argument("-n","--run_name",type = str, default = '768-unet-9-16il')
parser.add_argument("-l","--load_weight_name",type = str, default = "768-unet-9-16il.pt")
parser.add_argument("-t","--test_only",type = int, default = 0)
parser.add_argument("-tr","--training",type = int, default = 0)
parser.add_argument("-ev","--evaluate",type = int, default = 1)
parser.add_argument("-m","--model_only",type = int, default = 0)
parser.add_argument("-ts","--train_size",type = int, default= 125)
parser.add_argument("-lr","--learning_rate", type = float, default = 0.001)
parser.add_argument("-ap","--augment_prob", type = float, default = 0.25)
parser.add_argument("-mp","--mosiac_prob", type = float, default = 0.0)
parser.add_argument("-dr","--dropout_prob", type = float, default = 0.00)
parser.add_argument("-il","--initial_layer", type = int, default = 8)
parser.add_argument("-s","--input_shape", type = int, default = 640)
parser.add_argument("-eol","--eol_loss", type = int, default = 0)
parser.add_argument("-v","--augment_validation", type = bool, default = False)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-b","--batch_size",type=int, default = 8)
parser.add_argument("-e","--epoch",type=int, default=1750)
parser.add_argument("-dk","--decay",type=int, default=20)
parser.add_argument("-ch","--channels",type=int, default=3)
parser.add_argument("-w","--weak_boost",type=float, default=0.00)

args = parser.parse_args()
shape = args.input_shape

clas = args.class_name
workingdir = args.directory

pore_folder = 'pore_'+clas
nonpore_folder = 'non-pore_'+clas
clas = args.run_name
name="weight\\weight_"+clas+".pt"
tempna='./'

load_name = "weight\\weight_"+args.load_weight_name 
test_only=args.test_only
evaluate=args.evaluate
model_only = args.model_only
training = args.training
train_size = args.train_size

patiance = 1500

N=args.batch_size

class Cnn(nn.Module):
    def __init__(self,layers,outlayer=1,il=args.initial_layer):
        super().__init__()
        self.layers = layers
        self.blur = False
        l =[il*(2**i) for i in range(9)]
        #l = [4,8,16,32,64, 128, 256, 512,1024,2048, 4096]
        # input channel encoder
        self.maxpl7 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpl6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpl5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpl4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpl3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpl2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpl1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpl0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.d1 =  self.down_sample(self.layers, l[0], 3)
        self.d2 =  self.down_sample(l[0], l[1], 3)
        self.d3 =  self.down_sample(l[1], l[2], 3)
        self.d4 =  self.down_sample(l[2], l[3], 3)
        self.d5 =  self.down_sample(l[3], l[4], 3)
        #self.d6 =  self.down_sample(l[4], l[5], 3)
        #self.d7 =  self.down_sample(l[5], l[6], 3)
        #self.d8 =  self.down_sample(l[6], l[7], 3)
        #self.d9 =  self.down_sample(l[7], l[8], 3)

        #self.u0 =  self.up_sample(  l[8], l[7], 2)
        #self.ud0 = self.down_sample(l[7], l[7], 3)
        #self.u1 =  self.up_sample(  l[7], l[6], 2)
        #self.ud1 = self.down_sample(l[6], l[6], 3)
        #self.u2 =  self.up_sample(  l[5], l[5], 2)
        #self.ud2 = self.down_sample(l[5], l[4], 3)
        self.u3 =  self.up_sample(  l[4], l[4], 2)
        self.ud3 = self.down_sample(l[4], l[3], 3)
        self.u4 =  self.up_sample(  l[3], l[3], 2)
        self.ud4 = self.down_sample(l[3], l[2], 3)
        self.u5 =  self.up_sample(  l[2], l[2], 2)
        self.ud5 = self.down_sample(l[2], l[1], 3)
        self.u6 =  self.up_sample(  l[1], l[1], 2)
        self.ud6 = self.down_sample(l[1], l[0], 3)
        self.u7 =  self.up_sample(  l[0], l[0], 2)
        self.ud7 = self.down_sample(l[0], l[0], 3)
        self.out = nn.Conv2d(l[0], outlayer, kernel_size=1)
        self.cl = nn.Sigmoid()
        self.test = False
        self.print = False
        self.reduce = F.center_crop
        #self.reduce = nn.functional.interpolate
        


    def down_sample(self,ci, co, k, s=1, d=1, p=1):
        if self.blur:
            block = nn.Sequential(nn.Conv2d(ci, co, kernel_size=k, stride=s, dilation=d, padding=p),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
            T.GaussianBlur(3),
            nn.Conv2d(co, co, kernel_size=k, stride=s, dilation=d, padding=p),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
            T.GaussianBlur(3))
        else:
            block = nn.Sequential(nn.Conv2d(ci, co, kernel_size=k, stride=s, dilation=d, padding=p),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
            nn.Conv2d(co, co, kernel_size=k, stride=s, dilation=d, padding=p),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True))
        return block

    def up_sample(self,ci, co, k, s=2, d=1, p=0):
        if self.blur:
            block = nn.Sequential(nn.ConvTranspose2d(ci, co, kernel_size=k, stride=s, dilation=d, padding=p),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),
            T.GaussianBlur(3))
        else:
            block = nn.Sequential(nn.ConvTranspose2d(ci, co, kernel_size=k, stride=s, dilation=d, padding=p),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True))
        return block

    def unet(self,x,t="u"):
        x1 = self.maxpl0(self.d1(x))
        print('x1',x1.shape) if self.print else None

        x2 = self.maxpl1(self.d2(x1))
        print('x2',x2.shape) if self.print else None
        x3 = self.maxpl2(self.d3(x2))
        print('x3',x3.shape) if self.print else None
        x4 = self.maxpl3(self.d4(x3))
        print('x4',x4.shape) if self.print else None
        x5 = self.maxpl4(self.d5(x4))
        #print('x5',x5.shape) if self.print else None
        #x6 = self.maxpl5(self.d6(x5))
        #print('x6',x6.shape) if self.print else None
        #x7 = self.d7(x6)
        #print('x7',x7.shape) if self.print else None
        #x8 = self.maxpl5(self.d8(x7))
        #print('x8',x8.shape) if self.print else None
        #x9 = self.d9(x8)
        #print('x9',x9.shape) if self.print else None


        #y0 = self.u0(x9)
        #print('y0',y0.shape) if self.print else None
        #crop0 = self.reduce(x8,(y0.shape[-2],y0.shape[-1]))
        #y0b = self.ud0(crop0+y0)
        #y1 = self.u1(y0b)
        #
        #print('y1',y1.shape) if self.print else None
        #crop1 = self.reduce(x7,(y1.shape[-2],y1.shape[-1]))
        #y1b = self.ud1(crop1+y1)
        #y2 = self.u2(x6)
        #
        #print('y2',y2.shape) if self.print else None
        ##crop2 = self.reduce(x6,(y2.shape[-2],y2.shape[-1]))
        #y2b = self.ud2(y2)
        y3 = self.u3(x5)
        
        print('y3',y3.shape) if self.print else None
        #crop3 = self.reduce(x5,(y3.shape[-2],y3.shape[-1]))
        y3b = self.ud3(y3)
        y4 = self.u4(x4+y3b)
        
        print('y4',y4.shape) if self.print else None
        #crop4 = self.reduce(x3,(y4.shape[-2],y4.shape[-1]))
        y4b = self.ud4(y4)
        y5 = self.u5(x3+y4b)
        
        print('y5',y5.shape) if self.print else None
        y5b = self.ud5(y5)
        print('y5b',y5b.shape) if self.print else None
        #crop5 = self.reduce(x3,(y5.shape[-2],y5.shape[-1]))
        
        y6 = self.u6(x2+y5b)
        
        print('y6',y6.shape) if self.print else None
        #crop6 = self.reduce(x2,(y6.shape[-2],y6.shape[-1]))

        y6b = self.ud6(y6)
        y7 = self.u7(x1+y6b)

        print('y7',y7.shape) if self.print else None
        #crop7 = self.reduce(x1,(y7.shape[-2],y7.shape[-1]))
        y7b = self.ud7(y7)
        
        out = self.out(y7b)
        return out


    def forward(self, x):
        x1 = self.unet(x)
        return x1

class UPyramid(nn.Module):
    def __init__(self,inlayer,outlayer,il=args.initial_layer): #4,3,8
        super().__init__()
        
        self.inlayer = inlayer
        #self.P1 = Pyramid(inlayer,outlayer,il)
        self.P1 = Cnn(inlayer,outlayer,il)
        #self.P2 = Pyramid(3)
        #self.P3 = Pyramid(3)
        self.cl = nn.Sigmoid()

    def forward(self,x):
        s = x.shape
        #print(s)
        img_size = [s[-2],s[-1]]
        x = x.view(-1,self.inlayer, img_size[0], img_size[1])
        xo = self.P1(x)
        xo = self.cl(xo)
        #x2 = self.P2(x1)+x1
        #xo = self.P3(x2)+x2

        return(xo)



class Pyramid(nn.Module):
    def __init__(self,inchannels,outchannels,il=args.initial_layer):
        super().__init__()
        self.layer_1  = Cnn(inchannels,outchannels,il)
        self.layer_2  = Cnn(inchannels,outchannels,il)
        self.layer_3  = Cnn(inchannels,outchannels,il)

        #self.layer_out  = Cnn(4*3,outchannels)


        self.cl = nn.Sigmoid()

    def down_sample(self,x,scale):
        if scale>1:
            x = nn.functional.interpolate(x,((int(x.shape[-2]/scale),int(x.shape[-1]/scale))))
        x = F.gaussian_blur(x,3)
        return x

    def learn_up(self,channelsIn,channelsOut):
        block = nn.Sequential(nn.ConvTranspose2d(channelsIn, channelsIn, 3, stride=2,output_padding=1),
        nn.ReLU(),
        nn.Conv2d(channelsIn, channelsOut, 4,padding=1),
        nn.ReLU(),
        nn.Conv2d(channelsIn, channelsOut, 4,padding=1),
        nn.ReLU())
        return block

    def forward(self,x):
        x2 = self.down_sample(x,2)  #256
        x3 = self.down_sample(x,4) #128

        y3 = self.layer_3(x3)     #128
        y3 = nn.functional.interpolate(y3,(x.shape[-2],x.shape[-1]),mode='bilinear')
        y2 = self.layer_2(x2)     #256
        y2 = nn.functional.interpolate(y2,(x.shape[-2],x.shape[-1]),mode='bilinear')
        y1 = self.layer_1(x)      #512
        #o1 = self.layer_out(torch.cat((y1,y2,y3,y4),1))

        out =  y1+y2+y3
        return out


def save_model(name,model,model2,model3,model4,m5,m6,m7,m8,m9,file_dict,optimizer):
    save_dict = {'state_dict1':model.state_dict(),
                'state_dict2':model2.state_dict(),
                'state_dict3':model3.state_dict(),
                'state_dict4':model4.state_dict(),
                'state_dict5':m5.state_dict(),
                'state_dict6':m6.state_dict(),
                'state_dict7':m7.state_dict(),
                'state_dict8':m8.state_dict(),
                'state_dict9':m9.state_dict(),

                 'data_dict':file_dict,
                 'optimizer' : optimizer.state_dict()}
    x = torch.ones(1,512,512,1).to('cuda')
    model.test = True
    #net_trace = jit.trace(model,x)
    #jit.save(net_trace, name)
    torch.save(save_dict, name.replace('.','_torch.'))
    model.test = False

def load_model(name):
    save_dict = torch.load(name.replace('.','_torch.'), map_location="cuda")
    optimizer = save_dict['optimizer']
    net_dic1 = save_dict['state_dict1']
    net_dic2 = save_dict['state_dict2']
    net_dic3 = save_dict['state_dict3']
    net_dic4 = save_dict['state_dict4']
    net_dic5 = save_dict['state_dict5']
    net_dic6 = save_dict['state_dict6']
    net_dic7 = save_dict['state_dict7']
    net_dic8 = save_dict['state_dict8']
    net_dic9 = save_dict['state_dict9']

    file_dict = save_dict['data_dict']
    return net_dic1,net_dic2,net_dic3,net_dic4,net_dic5,net_dic6,net_dic7,net_dic8,net_dic9,file_dict, optimizer

def minmaxtransform(data,mins,maxs):
    return (data-mins)/(maxs-mins)

def generate_sets(path):
    # Creates classes from folders
    dirs = os.listdir( path )
    #data = []
    #with open(datafile, newline='') as f:
    #    reader = csv.reader(f)
    #    for row in reader:
    #        data.append(row)
    random.seed(0)
    random.shuffle(dirs)
   
    inp = []
    scale = []
    out = []
    weight = []
    i=0
    for d in dirs:
        if "mask" in d:
            try:
                fpath = path+d
                img = cv2.imread(fpath)
                #img1 = cv2.imread(fpath.replace("_mask",""))
                if (np.sum(img)>0):
                    #print(fpath,np.sum(img))
                    out.append(fpath)
                    s=img.shape
                    scale.append([s[1],s[0]])

                    if os.path.isfile(fpath.replace("_mask","").replace('png','jpg')):
                        inp.append(fpath.replace("_mask","").replace('png','jpg'))
                    else:
                        inp.append(fpath.replace("_mask","").replace('png','bmp'))
            except:
                print("Check %s"%(fpath))
        i+=1
        if i>2000:
            break
        
    print("%s images found and loaded"%(len(inp)))
    
    #out=np.array(out).astype(float)
    scale=np.array(scale).astype(float)

    size = len(inp)
    train = int(size*0.9)
    tst_val = int(size*0.1)
    index = np.argsort(scale[:,0])
    index = index.astype(int).tolist()


    inp = [inp[i] for i in index]
    scale = [scale[i] for i in index]
    out = [out[i] for i in index]
    #print(index)
    print(len(inp))

    file_dict = {}
    file_dict['train']={}

    file_dict['train']['inp']=inp[0:train]
    file_dict['train']['scale']=scale[0:train]
    #transf_norm = StandardScaler().fit(out[0:train])
    #transf_mnmx = MinMaxScaler().fit(transf_norm.transform(out[0:train]))
    #transf_mnmx = MinMaxScaler().fit(out[0:train])
    #file_dict['minmax']=transf_mnmx
    #file_dict['norm' ] = transf_norm
    file_dict['train']['out']= out[0:train]

    #file_dict['train']['out']= transf_norm.transform(out[0:train])
    #file_dict['train']['out']= transf_mnmx.transform(out[0:train])
    file_dict['val'  ]={}
    file_dict['val'  ]['inp']=inp[train:train+tst_val]
    file_dict['val'  ]['scale']=scale[train:train+tst_val]
    file_dict['val'  ]['out']=out[train:train+tst_val]

    #file_dict['val'  ]['out']=transf_norm.transform(out[train:train+tst_val])
    #file_dict['val'  ]['out']=transf_mnmx.transform(out[train:train+tst_val])
    file_dict['test' ]={}
    file_dict['test' ]['inp']=inp[train+tst_val:train+tst_val+tst_val]
    file_dict['test' ]['scale']=scale[train+tst_val:train+tst_val+tst_val]
    file_dict['test' ]['out']=out[train+tst_val:train+tst_val+tst_val]

    #file_dict['test' ]['out']=transf_norm.transform(out[train+tst_val:train+tst_val+tst_val])
    #file_dict['test' ]['out']=transf_mnmx.transform(out[train+tst_val:train+tst_val+tst_val])
    return file_dict
 
def eval_set(datadir,file_dict={}):
    imgs = os.listdir(datadir)

    dir_list = [datadir+i for i in imgs if "mask" not in i]
    file_dict['eval'] = {}
    file_dict['eval']['inp']=dir_list

    return file_dict

def eol_mask(img1):
    sh = img1.shape
    img1 = torch.round(img1).to(torch.uint8)
    #print(sh)
    #img1 = img1.view(-1,sh[2])
    num_cases = 100
    i=0
    while num_cases>0:
        w9 = img1[0:-2,0:-2]
        w2 = img1[1:-1,0:-2]
        w3 = img1[2:,0:-2]
        w8 = img1[0:-2,1:-1]
        w1 = img1[1:-1,1:-1]
        w4 = img1[2:,1:-1]
        w7 = img1[0:-2,2:]
        w6 = img1[1:-1,2:]
        w5 = img1[2:,2:]
        index = w1==1
        C = (torch.logical_not(w2[index]) & (w3[index] | w4[index])) + \
            (torch.logical_not(w4[index]) & (w5[index] | w6[index])) + \
            (torch.logical_not(w6[index]) & (w7[index] | w8[index])) + \
            (torch.logical_not(w8[index]) & (w9[index] | w2[index]))
        N1 = (w9[index] | w2[index]) + (w3[index] | w4[index]) + (w5[index] | w6[index]) + (w7[index] | w8[index])
        N2 = (w2[index] | w3[index]) + (w4[index] | w5[index]) + (w6[index] | w7[index]) + (w8[index] | w9[index])
        N  = torch.minimum(N1,N2)
        if i%2==0:
            m  = ((w2[index] | w3[index] | torch.logical_not(w5[index])) & w4[index])
        else:
            m  = ((w6[index] | w7[index] | torch.logical_not(w9[index])) & w8[index])
        cse = (C==1)*(N>=2)*(N<=3)*(m==0)
        num_cases = torch.sum(cse)
        img1[1:-1,1:-1][index]=~cse*img1[1:-1,1:-1][index]
        i+=1
    sh = img1.shape
    #print(sh)
    index = img1==1
    mask_overlap = torch.zeros(img1.shape,device="cuda")
    
    mask_overlap[0:-1,:] += img1[1:,:]
    mask_overlap[0:-1,0:-1] += img1[1:,1:]
    mask_overlap[0:,0:-1] += img1[:,1:]
    mask_overlap[1:,0:-1] += img1[:-1,1:]
    mask_overlap[1:,:] += img1[:-1,:]
    mask_overlap[1:,1:] += img1[:-1,:-1]
    mask_overlap[:,1:] += img1[:,:-1]
    mask_overlap[0:-1,1:] += img1[1:,:-1]
    
    mask_overlap[mask_overlap>1] = 0
    mask_overlap[mask_overlap==1] = 1
    mask_overlap[0,:] = 0
    mask_overlap[:,0] = 0
    mask_overlap[sh[0]-1,:]=0
    mask_overlap[:,sh[1]-1]=0
    index = img1==0
    mask_overlap[index]=0
    return mask_overlap

if __name__ == '__main__': 
    torch.cuda.empty_cache ()
    
    file_dict=generate_sets(workingdir)

    net =UPyramid(1,3,16)
    net2=UPyramid(1,3,16)
    net3=UPyramid(1,3,16)
    net4=UPyramid(4,3,16)
    net5=UPyramid(4,3,16)
    net6=UPyramid(4,3,16)
    net7=UPyramid(4,3,16) 
    net8=UPyramid(4,3,16)
    net9=UPyramid(4,3,16)


    #net2.blur=True
    print(torch.cuda.is_available())
    lr = args.learning_rate
    num_epoches = args.epoch
    optimizer = torch.optim.Adam(net.parameters(), lr)
    feature_encoder_scheduler = StepLR(optimizer,step_size=10,gamma=0.995)
    optimizer2 = torch.optim.Adam(net2.parameters(), lr)
    feature_encoder_scheduler2 = StepLR(optimizer2,step_size=10,gamma=0.995)
    optimizer3 = torch.optim.Adam(net3.parameters(), lr)
    feature_encoder_scheduler3 = StepLR(optimizer3,step_size=10,gamma=0.995)
    optimizer4 = torch.optim.Adam(net4.parameters(), lr)
    feature_encoder_scheduler4 = StepLR(optimizer4,step_size=10,gamma=0.995)
    optimizer5 = torch.optim.Adam(net5.parameters(), lr)
    feature_encoder_scheduler5 = StepLR(optimizer5,step_size=10,gamma=0.995)
    optimizer6 = torch.optim.Adam(net6.parameters(), lr)
    feature_encoder_scheduler6 = StepLR(optimizer6,step_size=10,gamma=0.995)
    optimizer7 = torch.optim.Adam(net7.parameters(), lr)
    feature_encoder_scheduler7 = StepLR(optimizer7,step_size=10,gamma=0.995)
    optimizer8 = torch.optim.Adam(net8.parameters(), lr)
    feature_encoder_scheduler8 = StepLR(optimizer8,step_size=10,gamma=0.995)
    optimizer9 = torch.optim.Adam(net9.parameters(), lr)
    feature_encoder_scheduler9 = StepLR(optimizer9,step_size=10,gamma=0.995)


    if test_only:
        print("loading")
        # load saved sets from model
        net_dic1,net_dic2,net_dic3,net_dic4,file_dict,_ = load_model(load_name)

        net.load_state_dict(net_dic1)
        net2.load_state_dict(net_dic2)
        net3.load_state_dict(net_dic3)
        net4.load_state_dict(net_dic4)
        net5.load_state_dict(net_dic5)
        net6.load_state_dict(net_dic6)
        net7.load_state_dict(net_dic7)

    elif model_only:
        print("loading")
        net_dic1,net_dic2,net_dic3,net_dic4,net_dic5,net_dic6,net_dic7,net_dic8,net_dic9,file_dict,optimizer = load_model(load_name)

        net.load_state_dict(net_dic1)
        net2.load_state_dict(net_dic2)
        net3.load_state_dict(net_dic3)
        #net4.load_state_dict(net_dic4)
        #net5.load_state_dict(net_dic5)
        #net6.load_state_dict(net_dic6)
        #net7.load_state_dict(net_dic7)
        #net8.load_state_dict(net_dic8)
        #net9.load_state_dict(net_dic9)

        optimizer = torch.optim.Adam(net.parameters(), lr)
        feature_encoder_scheduler = StepLR(optimizer,step_size=10,gamma=0.99)
        optimizer2 = torch.optim.Adam(net2.parameters(), lr)
        feature_encoder_scheduler2 = StepLR(optimizer2,step_size=10,gamma=0.99)
        optimizer3 = torch.optim.Adam(net3.parameters(), lr)
        feature_encoder_scheduler3 = StepLR(optimizer3,step_size=10,gamma=0.99)
        optimizer4 = torch.optim.Adam(net4.parameters(), lr)
        feature_encoder_scheduler4 = StepLR(optimizer4,step_size=10,gamma=0.99)
        optimizer5 = torch.optim.Adam(net5.parameters(), lr)
        feature_encoder_scheduler5 = StepLR(optimizer5,step_size=10,gamma=0.99)
        optimizer6 = torch.optim.Adam(net6.parameters(), lr)
        feature_encoder_scheduler6 = StepLR(optimizer6,step_size=10,gamma=0.99)
        optimizer7 = torch.optim.Adam(net7.parameters(), lr)
        feature_encoder_scheduler7 = StepLR(optimizer7,step_size=10,gamma=0.99)
        optimizer8 = torch.optim.Adam(net8.parameters(), lr)
        feature_encoder_scheduler8 = StepLR(optimizer8,step_size=10,gamma=0.99)
        optimizer9 = torch.optim.Adam(net9.parameters(), lr)
        feature_encoder_scheduler9 = StepLR(optimizer9,step_size=10,gamma=0.99)


    if torch.cuda.is_available() :
        net = net.cuda()  
        net2=net2.cuda()
        net3=net3.cuda()
        net4=net4.cuda()
        net5=net5.cuda()
        net6=net6.cuda()
        net7=net7.cuda()
        net8=net8.cuda()
        net9=net9.cuda()

    
    train_set = custom_dset(file_dict,'train',args)
    #train_set.aug_indx = [3,7,8,13,15]
    train_set.aug_indx = [3,8,9,14]
    train_loader = DataLoader(train_set, batch_size=N, shuffle=True, num_workers=8,pin_memory=True,persistent_workers=True)
    val_set = custom_dset(file_dict,'val',args)
    val_set.aug_indx =   []
    val_loader = DataLoader(val_set, batch_size=N, shuffle=True, num_workers=4,pin_memory=True,persistent_workers=True)  


    np.set_printoptions(threshold=sys.maxsize)
    loss_func = nn.MSELoss()
    l_his=[]
    acc_hist = []
    iou_hist = []
    elastic_transformer = T.ElasticTransform(alpha=25.0)
    if training==1:
        acc = 1000
        for epoch in range(num_epoches):
            net.train()
            print('Epoch:', epoch + 1, 'Training...')
            running_loss = 0.0 

            for i,data in enumerate(train_loader, 0):
                image1s,img_size,outputs,aug_img,aug_img2,_=data

                if torch.cuda.is_available():
                    image1s = image1s.cuda(0)
                    aug_img = aug_img.cuda(1)
                    aug_img2 = aug_img2.cuda(2)
                    outputs1 = outputs.cuda(0)
                    outputs2 = outputs.cuda(1)
                    outputs3 = outputs.cuda(2)

                image1s,  outputs1 = Variable(image1s.float()), Variable(outputs1.float())
                aug_img,  outputs2 = Variable(aug_img.float()), Variable(outputs2.float())
                aug_img2, outputs3 = Variable(aug_img2.float()), Variable(outputs3.float())

                aug_img = nn.functional.interpolate(aug_img.unsqueeze(1),(int(aug_img2.shape[-2]/2),int(aug_img2.shape[-1]/2)),mode='bilinear')
                aug_img = F.gaussian_blur(aug_img,3)
                aug_img = nn.functional.interpolate(aug_img,(image1s.shape[-2],image1s.shape[-1]),mode='bilinear')

                aug_img2 = nn.functional.interpolate(aug_img2.unsqueeze(1),(int(aug_img2.shape[-2]/4),int(aug_img2.shape[-1]/4)),mode='bilinear')
                aug_img2 = F.gaussian_blur(aug_img2,3)
                aug_img2 = nn.functional.interpolate(aug_img2,(image1s.shape[-2],image1s.shape[-1]),mode='bilinear')

                aug_img = aug_img.squeeze(1)
                aug_img2 = aug_img2.squeeze(1)

                optimizer.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                optimizer4.zero_grad()
                optimizer5.zero_grad()
                optimizer6.zero_grad()
                optimizer7.zero_grad()
                optimizer8.zero_grad()
                optimizer9.zero_grad()

                outputs = outputs.view(-1,3,shape,shape)
                with torch.no_grad():
                    f1=net(image1s)
                    f2=net2(aug_img)
                    f3=net3(aug_img2)
                    loss1 =loss_func(f1,outputs1)
                    loss2 =loss_func(f2,outputs2)
                    loss3 =loss_func(f3,outputs3)
                
                f4=net4(torch.cat((image1s.unsqueeze(1),f1),1))
                f5=net5(torch.cat((aug_img.unsqueeze(1), f2),1))
                f6=net6(torch.cat((aug_img2.unsqueeze(1), f3),1))

                f7=net7(torch.cat((aug_img2.unsqueeze(1),f3),1))
                f8=net8(torch.cat((aug_img.unsqueeze(1), f2),1))
                f9=net9(torch.cat((image1s.unsqueeze(1), f1),1))
            
            
                lossres1m =loss_func(f4,(outputs1 - f1).clip(0,1))
                lossres2m =loss_func(f5,(outputs2 - f2).clip(0,1))
                lossres3m =loss_func(f6,(outputs3 - f3).clip(0,1))

                lossres3 = loss_func(f7,(f3 - outputs3).clip(0,1))
                lossres2 = loss_func(f8,(f2 - outputs2).clip(0,1))
                lossres1 = loss_func(f9,(f1 - outputs1).clip(0,1))

                loss_cross1 = ((f9*f4)**2).mean()
                loss_cross2 = ((f8*f5)**2).mean()
                loss_cross3 = ((f7*f6)**2).mean()

                loss_dev1 = lossres1m + lossres1 + loss_cross1 * 0.5 #+ loss1
                loss_dev2 = lossres2m + lossres2 + loss_cross2 * 0.5 #+ loss2
                loss_dev3 = lossres3m + lossres3 + loss_cross3 * 0.5 #+ loss3

                loss_dev1.backward()
                loss_dev2.backward()
                loss_dev3.backward()

                running_loss += (loss_dev1.cpu()+loss_dev2.cpu()+loss_dev3.cpu())
                #optimizer.step()
                #optimizer2.step()
                #optimizer3.step()
                optimizer4.step()
                optimizer5.step()
                optimizer6.step()
                optimizer7.step()
                optimizer8.step()
                optimizer9.step()


                if i==1:
                    print(image1s.shape,outputs.shape,f1.shape)
                    outputs = np.array(outputs.cpu().detach().numpy())
                    image1s = np.array(image1s.cpu().detach().numpy())
                    f1 = np.array(f1.cpu().detach().numpy())
                    f2 = np.array(f2.cpu().detach().numpy())
                    f3 = np.array(f3.cpu().detach().numpy())
                    f4 = np.array(f4.cpu().detach().numpy())
                    f5 = np.array(f5.cpu().detach().numpy())
                    f6 = np.array(f6.cpu().detach().numpy())
                    f7 = np.array(f7.cpu().detach().numpy())
                    f8 = np.array(f8.cpu().detach().numpy())
                    f9 = np.array(f9.cpu().detach().numpy())

                    img_in = (image1s[0,:,:]*255).astype('uint8')
                    mask_1 = (np.stack([f1[0,0,:,:],f1[0,1,:,:],f1[0,2,:,:]],axis=2)*255).astype('uint8')
                    mask_2 = (np.stack([f2[0,0,:,:],f2[0,1,:,:],f2[0,2,:,:]],axis=2)*255).astype('uint8')
                    mask_3 = (np.stack([f3[0,0,:,:],f3[0,1,:,:],f3[0,2,:,:]],axis=2)*255).astype('uint8')
                    
                    mask_4 = (np.stack([f4[0,0,:,:],f4[0,1,:,:],f4[0,2,:,:]],axis=2)*255).astype('uint8')
                    mask_5 = (np.stack([f5[0,0,:,:],f5[0,1,:,:],f5[0,2,:,:]],axis=2)*255).astype('uint8')
                    mask_6 = (np.stack([f6[0,0,:,:],f6[0,1,:,:],f6[0,2,:,:]],axis=2)*255).astype('uint8')
                    mask_7 = (np.stack([f7[0,0,:,:],f7[0,1,:,:],f7[0,2,:,:]],axis=2)*255).astype('uint8')
                    mask_8 = (np.stack([f8[0,0,:,:],f8[0,1,:,:],f8[0,2,:,:]],axis=2)*255).astype('uint8')
                    mask_9 = (np.stack([f9[0,0,:,:],f9[0,1,:,:],f9[0,2,:,:]],axis=2)*255).astype('uint8')

                    mask_in = (np.stack([outputs[0,0,:,:],outputs[0,1,:,:],outputs[0,2,:,:]],axis=2)*255).astype('uint8')
                    cv2.imwrite("img_debug\\"+str(epoch)+"_"+"mask_1.png", mask_1)
                    cv2.imwrite("img_debug\\"+str(epoch)+"_"+"img.png", img_in)
                    cv2.imwrite("img_debug\\"+str(epoch)+"_"+"mask_2.png", mask_2)
                    cv2.imwrite("img_debug\\"+str(epoch)+"_"+"mask_3.png", mask_3)
                    cv2.imwrite("img_debug\\"+str(epoch)+"_"+"mask_4.png", mask_4)
                    cv2.imwrite("img_debug\\"+str(epoch)+"_"+"mask_5.png", mask_5)
                    cv2.imwrite("img_debug\\"+str(epoch)+"_"+"mask_6.png", mask_6)
                    cv2.imwrite("img_debug\\"+str(epoch)+"_"+"mask_7.png", mask_7)
                    cv2.imwrite("img_debug\\"+str(epoch)+"_"+"mask_8.png", mask_8)
                    cv2.imwrite("img_debug\\"+str(epoch)+"_"+"mask_9.png", mask_9)

                    cv2.imwrite("img_debug\\"+str(epoch)+"_"+"mask_in.png", mask_in)
            #feature_encoder_scheduler.step()  
            #feature_encoder_scheduler2.step()    
            #feature_encoder_scheduler3.step() 
            feature_encoder_scheduler4.step()      
            feature_encoder_scheduler5.step()   
            feature_encoder_scheduler6.step()   
            feature_encoder_scheduler7.step()   
            feature_encoder_scheduler8.step()
            feature_encoder_scheduler9.step()

                    

            running_loss = running_loss / (i+1)
            l_his.append(running_loss.cpu().detach().numpy())
            correct = 0
            total = 0
            f1_conv = 0
            out_conv = 0
            val_acc = 0
            evaled =0
            net.eval()
            with torch.no_grad():
                val_running_loss = 0.0 
                for j,data in enumerate(val_loader):
                    image1s,img_size,outputs1=data
                    if torch.cuda.is_available():
                        image1s = image1s.cuda(0)
                        image2s = image1s.cuda(1)
                        image3s = image1s.cuda(2)
                        outputs = outputs1.cpu()
                    
                    image1s = Variable(image1s.float())
                    image2s = Variable(image2s.float())
                    image3s = Variable(image3s.float())

                    image2s = F.gaussian_blur(image2s,5)
                    image3s = nn.functional.interpolate(image3s.unsqueeze(1),(int(image3s.shape[-2]/2),int(image3s.shape[-1]/2)),mode='bilinear')
                    image3s = F.gaussian_blur(image3s,5)
                    image3s = nn.functional.interpolate(image3s,(image2s.shape[-2],image2s.shape[-1]),mode='bilinear')

                    image3s = image3s.squeeze(1)

                    outputs = outputs.view(-1,3,shape,shape)
                    f1= net(image1s)
                    f2=net2(image2s)
                    f3=net3(image3s)
                    f4=net4(torch.cat((image1s.unsqueeze(1),f1),1))
                    f5=net5(torch.cat((image2s.unsqueeze(1), f2),1))
                    f6=net6(torch.cat((image3s.unsqueeze(1), f3),1))
    
                    f7=net7(torch.cat((image3s.unsqueeze(1), f3),1))
                    f8=net8(torch.cat((image2s.unsqueeze(1), f2),1))
                    f9=net9(torch.cat((image1s.unsqueeze(1), f1),1))
                    
                    f_conv1 = f1 + f4 - f9
                    f_conv2 = f2 + f5 - f8
                    f_conv3 = f3 + f6 - f7
                    f_est = f_conv1.cpu()/3 + f_conv2.cpu()/3 + f_conv3.cpu()/3
                    f_pred = f_conv1.cpu()*2/3 + f_conv2.cpu()/3 + f_conv3.cpu()/3

                    
                    loss = loss_func(f_est, outputs)
                    val_running_loss += loss

                    f1_conv = np.round(f_pred.numpy().clip(0.,1.))
                    out_conv = np.array(np.round(outputs.numpy()))
                    img_conv = np.array(image1s.squeeze(0).squeeze(0).cpu().numpy()*255).astype('uint8')
                    if math.isnan(np.mean(np.sum(f1_conv*out_conv)/(np.sum(out_conv)+np.sum(f1_conv)-np.sum(f1_conv*out_conv)))):
                        val_acc += 0
                    else:
                        val_acc += np.mean(np.sum(f1_conv*out_conv)/(np.sum(out_conv)+np.sum(f1_conv)-np.sum(f1_conv*out_conv)))
                    if j==0:
                        outputs = np.array(outputs.detach().numpy())
                        image1s = np.array(image1s.cpu().detach().numpy())
                        #mask_in = (np.stack([outputs[0,0,:,:],outputs[0,1,:,:],outputs[0,2,:,:]],axis=2)*255).astype('uint8')
                        img_in = (image1s[0,:,:]*255).astype('uint8')
                        mask_in = (np.stack([outputs[0,0,:,:],outputs[0,1,:,:],outputs[0,2,:,:]],axis=2)*255).astype('uint8')
                        mask_out = (np.stack([f1_conv[0,0,:,:],f1_conv[0,1,:,:],f1_conv[0,2,:,:]],axis=2)*255).astype('uint8')
                        #concatimg = np.concatenate((mask_out,np.round(out_conv*255).astype('uint8')),axis=1)
                        cv2.imwrite("val\\"+str(epoch)+"_"+"mask_in.png", mask_in)
                        cv2.imwrite("val\\"+str(epoch)+"_"+"img.png", img_in)
                        cv2.imwrite("val\\"+str(epoch)+"out.png", mask_out)

            print(f1_conv.shape,out_conv.shape,img_conv.shape)
            

            val_loss = val_running_loss /(j+1)
            val_acc = val_acc/(j+1)
            if val_loss < acc:
                save_model(name,net,net2,net3,net4,net5,net6,net7,net8,net9,file_dict,optimizer)
                acc = val_loss
                print("model_saved")
                last_epoch = epoch + 1
            print('[%d] train_loss: %.8f  val_loss %.8f  val_acc %.8f  last_saved_epoch %d ' % 
                  (epoch + 1, running_loss, val_loss, val_acc, last_epoch))

            iou_hist.append(val_acc)
            acc_hist.append(val_loss.cpu().numpy())
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.plot(iou_hist,label='IOU Accuracy')    
            plt.xlabel('Epoch')  
            plt.ylabel('Acc') 
            try:
                fig.savefig('plots\\plot_val_loss_'+clas+'.png') 
            except:
                print('save failed for some reason')
            plt.close()
            fig = plt.figure()
            ax = plt.subplot(111)
            ax.plot(l_his,'r',label='training loss')
            ax.plot(acc_hist,'b',label='validation loss')    
            plt.legend()
            plt.xlabel('Epoch')  
            plt.ylabel('Loss')  
            try:
                fig.savefig('plots\\plot_train_loss'+clas+'.png') 
            except:
                print('save failed for some reason')
            plt.close()
            if (np.array(acc_hist[-patiance:])<max(acc_hist)).all():
                break

        print('Finished Training')
        
        save_model('weight\\weight_final'+clas+'.pt',net,file_dict,optimizer)

    if evaluate==1:
        net_dic1,net_dic2,net_dic3,net_dic4,net_dic5,net_dic6,net_dic7,net_dic8,net_dic9,_,optimizer = load_model(load_name)
        
        net.load_state_dict(net_dic1)
        net2.load_state_dict(net_dic2)
        net3.load_state_dict(net_dic3)
        net4.load_state_dict(net_dic4)
        net5.load_state_dict(net_dic5)
        net6.load_state_dict(net_dic6)
        net7.load_state_dict(net_dic7)
        net8.load_state_dict(net_dic8)
        net9.load_state_dict(net_dic9)
        file_dict = eval_set('C:\\Updated_ML\\found_gb_image_only\\')
        eval_set = custom_dset(file_dict,'eval',args)
        train_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=1,pin_memory=True,persistent_workers=True)
        with torch.no_grad():
         for i,data in enumerate(train_loader, 0):
            image1s,imp_shape,name=data

            if torch.cuda.is_available():
                image1s = image1s.cuda()
                image2s = image1s.cuda()
                image3s = image1s.cuda()
                    
            image1s = Variable(image1s.float())
            image2s = Variable(image2s.float())
            image3s = Variable(image3s.float())
            image2s = F.gaussian_blur(image2s,5)
            image3s = nn.functional.interpolate(image3s.unsqueeze(1),(int(image3s.shape[-2]/2),int(image3s.shape[-1]/2)),mode='bilinear')
            image3s = F.gaussian_blur(image3s,5)
            image3s = nn.functional.interpolate(image3s,(image2s.shape[-2],image2s.shape[-1]),mode='bilinear')
            image3s = image3s.squeeze(1)

            f1= net(image1s)
            #f2=net2(image2s)
            #f3=net3(image3s)
            f4=net4(torch.cat((image1s.unsqueeze(1),f1),1))
            #f5=net5(torch.cat((image2s.unsqueeze(1), f2),1))
            #f6=net6(torch.cat((image3s.unsqueeze(1), f3),1))
            #f7=net7(torch.cat((image3s.unsqueeze(1), f3),1))
            #f8=net8(torch.cat((image2s.unsqueeze(1), f2),1))
            f9=net9(torch.cat((image1s.unsqueeze(1), f1),1))
            
            f_conv1 = f1 + f4/2 - f9/2
            #f_conv2 = f2 + f5 - f8
            #f_conv3 = f3 + f6 - f7
            f_pred = f_conv1.cpu()#/3 + f_conv2.cpu()/3 + f_conv3.cpu()/3
            f_pred[0,0,:,:][f_pred[0,0,:,:]>0.3]=1
            f_pred[0,1,:,:][f_pred[0,1,:,:]>0.4]=1
            f_pred[0,2,:,:][f_pred[0,2,:,:]>0.2]=1
            
            outputs = np.round(f_pred.detach().numpy().clip(0.,1.))
            image1s = np.array(image1s.detach().cpu().numpy())
            #concatimg = np.concatenate((outputs.astype('uint8'),image1s.astype('uint8')))
            print(name)
            n = name[0].replace(".jpg",'').replace('.bmp','').replace('.png','')
            mask_out = (np.stack([outputs[0,0,:,:],outputs[0,1,:,:],outputs[0,2,:,:]],axis=2)*255).astype('uint8')
            img_in = (image1s*255).astype('uint8')
            cv2.imwrite("eval\\found_ensamble\\"+n+"_mask.png",mask_out.astype('uint8'))
            #cv2.imwrite("eval\\val_ensamble\\"+n+"_mask.png",mask_out.astype('uint8'))
            print(i)

   