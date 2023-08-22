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
parser.add_argument("-n","--run_name",type = str, default = '640-singleunet-050923')
parser.add_argument("-l","--load_weight_name",type = str, default = "640-singleunet-050923.pt")
parser.add_argument("-t","--test_only",type = int, default = 0)
parser.add_argument("-tr","--training",type = int, default = 0)
parser.add_argument("-ev","--evaluate",type = int, default = 1)
parser.add_argument("-m","--model_only",type = int, default = 0)
parser.add_argument("-ts","--train_size",type = int, default= 125)
parser.add_argument("-lr","--learning_rate", type = float, default = 0.0002)
parser.add_argument("-ap","--augment_prob", type = float, default = 0.05)
parser.add_argument("-mp","--mosiac_prob", type = float, default = 0.0)
parser.add_argument("-dr","--dropout_prob", type = float, default = 0.00)
parser.add_argument("-il","--initial_layer", type = int, default = 64)
parser.add_argument("-s","--input_shape", type = int, default = 640)
parser.add_argument("-eol","--eol_loss", type = int, default = 0)
parser.add_argument("-v","--augment_validation", type = bool, default = False)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-b","--batch_size",type=int, default = 2)
parser.add_argument("-e","--epoch",type=int, default=1000)
parser.add_argument("-dk","--decay",type=int, default=100)
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
    def __init__(self):
        super().__init__()
        l =[args.initial_layer*(2**i) for i in range(9)]
        #l = [4,8,16,32,64, 128, 256, 512,1024,2048, 4096]
        # input channel encoder
        self.blur=False
        self.maxpl7 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpl6 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpl5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpl4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpl3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpl2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpl1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpl0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.d1 =  self.down_sample(1, l[0], 3)
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
        self.out = nn.Conv2d(l[0], 3, kernel_size=1)
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
        print('x5',x5.shape) if self.print else None
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
        s = x.shape
        img_size = [s[1],s[2]]
        x = x.view(-1,1, img_size[0], img_size[1])
        x1 = self.unet(x)
        out =  self.cl(x1)
        return out


def save_model(name,model,file_dict,optimizer):
    save_dict = {'state_dict':model.state_dict(),
                 'data_dict':file_dict,
                 'optimizer' : optimizer.state_dict()}
    x = torch.ones(1,768,768,1).to('cuda')
    model.test = True
    net_trace = jit.trace(model,x)
    jit.save(net_trace, name)
    torch.save(save_dict, name.replace('.','_torch.'))
    model.test = False

def load_model(name):
    save_dict = torch.load(name.replace('.','_torch.'), map_location="cuda")
    optimizer = save_dict['optimizer']
    net_dic = save_dict['state_dict']
    file_dict = save_dict['data_dict']
    return net_dic,file_dict, optimizer

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


if __name__ == '__main__': 
    torch.cuda.empty_cache ()
    
    file_dict=generate_sets(workingdir)

    net=Cnn()

    if test_only:
        print("loading")
        # load saved sets from model
        net_dic,file_dict = load_model(load_name)

        net.load_state_dict(net_dic)
    elif model_only:
        print("loading")
        net_dic,_,optimizer = load_model(load_name)
        net.load_state_dict(net_dic)

    if torch.cuda.is_available() :
        net = net.cuda()  
    
    train_set = custom_dset(file_dict,'train',args)
    train_set.aug_indx = [3,8,9,14]
    train_loader = DataLoader(train_set, batch_size=N, shuffle=True, num_workers=N,pin_memory=True,persistent_workers=True)
    val_set = custom_dset(file_dict,'val',args)
    val_set.aug_indx =   []
    val_loader = DataLoader(val_set, batch_size=5, shuffle=True, num_workers=5,pin_memory=True,persistent_workers=True)  

    

    lr = args.learning_rate
    num_epoches = args.epoch

    optimizer = torch.optim.Adam(net.parameters(), lr)
    feature_encoder_scheduler = StepLR(optimizer,step_size=10,gamma=0.95)


    np.set_printoptions(threshold=sys.maxsize)
    loss_func = nn.MSELoss()
    l_his=[]
    acc_hist = []
    iou_hist = []

    if training==1:
        acc = 1000
        for epoch in range(num_epoches):
            net.train()
            print('Epoch:', epoch + 1, 'Training...')
            running_loss = 0.0 

            for i,data in enumerate(train_loader, 0):
                image1s,img_size,outputs,_,_,_=data
                
                if torch.cuda.is_available():
                    image1s = image1s.cuda()
                    outputs = outputs.cuda()
                image1s,  outputs = Variable(image1s.float()), Variable(outputs.float())

                optimizer.zero_grad()
                
                f1=net(image1s)
                #
                outputs = outputs.view(-1,3,shape,shape)
                loss = loss_func(f1,outputs)
                loss.backward()
                running_loss += loss
                optimizer.step()
                
            
                if i==1:
                    print(image1s.shape,outputs.shape,f1.shape)
                    outputs = np.array(outputs.cpu().detach().numpy())
                    image1s = np.array(image1s.cpu().detach().numpy())
                    f1 = np.array(f1.cpu().detach().numpy())
                    mask_in = (np.stack([outputs[0,0,:,:],outputs[0,1,:,:],outputs[0,2,:,:]],axis=2)*255).astype('uint8')
                    img_in = (image1s[0,:,:]*255).astype('uint8')
                    mask_out = (np.stack([f1[0,0,:,:],f1[0,1,:,:],f1[0,2,:,:]],axis=2)*255).astype('uint8')
                    cv2.imwrite("img_debug\\"+str(epoch)+"_"+"mask_in.png", mask_in)
                    cv2.imwrite("img_debug\\"+str(epoch)+"_"+"img.png", img_in)
                    cv2.imwrite("img_debug\\"+str(epoch)+"_"+"mask_out.png", mask_out)

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
                        image1s = image1s.cuda()
                        outputs = outputs1.cuda()
                    
                    image1s, outputs = Variable(image1s.float()), Variable(outputs.float())

                    f1=net(image1s)

                    loss = loss_func(f1,outputs)

                    val_running_loss += loss
                    #f1_conv = torch.argmax(torch.nn.functional.softmax(f1, 1), dim=1)
                    #f1[f1>0.3]=1
                    f1_conv = np.array(np.round(f1.cpu().numpy()))
                    out_conv = np.array(np.round(outputs.cpu().numpy()))
                    img_conv = np.array(image1s.squeeze(0).squeeze(0).cpu().numpy()*255).astype('uint8')
                    if math.isnan(np.mean(np.sum(f1_conv*out_conv)/(np.sum(out_conv)+np.sum(f1_conv)-np.sum(f1_conv*out_conv)))):
                        val_acc += 0
                    else:
                        val_acc += np.mean(np.sum(f1_conv*out_conv)/(np.sum(out_conv)+np.sum(f1_conv)-np.sum(f1_conv*out_conv)))
                    if j==0:
                        outputs = np.array(outputs.cpu().detach().numpy())
                        image1s = np.array(image1s.cpu().detach().numpy())
                        mask_in = (np.stack([outputs[0,0,:,:],outputs[0,1,:,:],outputs[0,2,:,:]],axis=2)*255).astype('uint8')
                        img_in = (image1s[0,:,:]*255).astype('uint8')
                        mask_out = np.stack([f1_conv[0,0,:,:],f1_conv[0,1,:,:],f1_conv[0,2,:,:]],axis=2).astype('uint8')*255
                        #concatimg = np.concatenate((mask_out,np.round(out_conv*255).astype('uint8')),axis=1)
                        cv2.imwrite("val\\"+str(epoch)+"_"+"mask_in.png", mask_in)
                        cv2.imwrite("val\\"+str(epoch)+"_"+"img.png", img_in)
                        cv2.imwrite("val\\"+str(epoch)+"out.png", mask_out)

            print(f1_conv.shape,out_conv.shape,img_conv.shape)
            

            val_loss = val_running_loss /(j+1)
            val_acc = val_acc/(j+1)
            if val_loss < acc:
                save_model(name,net,file_dict,optimizer)
                acc = val_loss
                print("model_saved")
                last_epoch = epoch + 1
            print('[%d] train_loss: %.8f  val_loss %.8f  val_acc %.8f  last_saved_epoch %d' %
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
        net_dic,_,optimizer = load_model(load_name)
        net.load_state_dict(net_dic)
        #file_dict = eval_set('C:\\Updated_ML\\found_gb_image_only\\')
        file_dict = eval_set('C:\\projects\\combine_msks\\unused\\')
        eval_set = custom_dset(file_dict,'eval',args)
        train_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=1,pin_memory=True,persistent_workers=True)
        for i,data in enumerate(train_loader, 0):
            image1s,imp_shape,name=data
            if torch.cuda.is_available():
                image1s = image1s.cuda()


            
            f1=net(image1s.float())
            f1[0,0,:,:][f1[0,0,:,:]>0.3]=1
            f1[0,1,:,:][f1[0,1,:,:]>0.3]=1
            f1[0,2,:,:][f1[0,2,:,:]>0.2]=1
            outputs = np.array(np.round(f1.detach().cpu().numpy()))
            image1s = np.array(image1s.detach().cpu().numpy())*255
            print(name)
            n = name[0].replace(".jpg",'').replace('.bmp','').replace('.png','')
            mask_out = (np.stack([outputs[0,0,:,:],outputs[0,1,:,:],outputs[0,2,:,:]],axis=2)*255).astype('uint8')
            img_in = (image1s*255).astype('uint8')
            cv2.imwrite("eval\\test_unet64\\"+n+"_mask.png",mask_out.astype('uint8'))
            #cv2.imwrite("eval\\val_UNet64\\"+n+"_mask.png",mask_out.astype('uint8'))
            print(i)

   