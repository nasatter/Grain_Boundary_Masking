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
from torchvision.models import resnet18,ResNet18_Weights
from collections import OrderedDict 

# see run.bat and test.bat for examples
parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-d","--directory",type = str, default = 'C:\\projects\\combine_msks\\combined\\')
parser.add_argument("-c","--class_name",type = str, default = 'new')
parser.add_argument("-n","--run_name",type = str, default = '512-resunet_nonu')
parser.add_argument("-l","--load_weight_name",type = str, default = "512-resunet_nonu.pt")
parser.add_argument("-t","--test_only",type = int, default = 0)
parser.add_argument("-tr","--training",type = int, default = 1)
parser.add_argument("-ev","--evaluate",type = int, default = 0)
parser.add_argument("-m","--model_only",type = int, default = 0)
parser.add_argument("-ts","--train_size",type = int, default= 125)
parser.add_argument("-lr","--learning_rate", type = float, default = 0.001)
parser.add_argument("-ap","--augment_prob", type = float, default = 0.75)
parser.add_argument("-mp","--mosiac_prob", type = float, default = 0.0)
parser.add_argument("-dr","--dropout_prob", type = float, default = 0.00)
parser.add_argument("-il","--initial_layer", type = int, default = 16)
parser.add_argument("-s","--input_shape", type = int, default = 512)
parser.add_argument("-eol","--eol_loss", type = int, default = 0)
parser.add_argument("-v","--augment_validation", type = bool, default = False)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-b","--batch_size",type=int, default = 4)
parser.add_argument("-e","--epoch",type=int, default=2000)
parser.add_argument("-dk","--decay",type=int, default=100)
parser.add_argument("-ch","--channels",type=int, default=3)
parser.add_argument("-w","--weak_boost",type=float, default=0.00)

args = parser.parse_args()
shape = args.input_shape

clas = args.class_name
workingdir = args.directory

il = args.initial_layer
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

resnet34 = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
# here we copy the weights into a single channel - comment out for single channel
temp = resnet34.conv1.weight[:,0:1,:,:].clone()
new_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,bias=False)
new_layer.weight[:,:,:,:].data[...] =  Variable(temp, requires_grad=True)
resnet34.conv1 = new_layer
# here we copy the weights into a more than 3 channels - comment out for more than 3 channels
#temp = resnet34.conv1.weight[:,0:3,:,:].clone()
#new_layer = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
#new_layer.weight[:,0:3,:,:].data[...] =  Variable(temp, requires_grad=True)
#resnet34.conv1 = new_layer
my_model = nn.Sequential(*list(resnet34.children())[:-2])

my_model = my_model.cuda()
            
class Cnn(nn.Module):
    def __init__(self):
        super().__init__()
        l = [il, il*2, il*4,il*8,il*16,il*32, il*64]
        self.output_layers = [0,1,2,3,4,5,6,7]
        self.k = ['0', '1','2', '3', '4', '5', '6', '7']
        self.pretrained = my_model
        self.selected_out = OrderedDict()
        self.fhooks = []
        self.print = False
        cin = 512
        self.su4 =  nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                    #nn.MaxPool2d(2),
                    nn.ReLU(inplace=True))
        self.su3 =  nn.Sequential(nn.ConvTranspose2d(int(cin/2**5), int(cin/2**5), kernel_size=4, stride=4),
                    nn.MaxPool2d(4),
                    nn.ReLU(inplace=True))
        self.su2 =  nn.Sequential(nn.ConvTranspose2d(int(cin/2**5), int(cin/2**5), kernel_size=2, stride=2),
                    #nn.MaxPool2d(2),
                    nn.ReLU(inplace=True))
        self.su1 =  nn.Sequential(nn.ConvTranspose2d(int(cin/2**5), int(cin/2**5), kernel_size=2, stride=2),
                    #nn.MaxPool2d(2),
                    nn.ReLU(inplace=True))
        self.su0 =  nn.Sequential(nn.ConvTranspose2d(int(cin/4**4), 1, kernel_size=2, stride=2))

        self.bninp = nn.BatchNorm2d(1)
        self.u0 =  self.up_sample(cin,int(cin/2),2)
        self.ud0 = self.down_sample(cin, int(cin/2), 3)
        self.u1 =  self.up_sample(int(cin/2),int(cin/2**1),2)
        self.ud1 = self.down_sample(int(cin/2**1), int(cin/2**2), 3)
        self.u2 =  self.up_sample(int(cin/2**2),int(cin/2**2),2)
        self.ud2 = self.down_sample(int(cin/2**2),int(cin/2**2), 3)
        self.u2c =  self.up_sample(int(cin/2**2),int(cin/2**3),2)
        self.ud2c = self.down_sample(int(cin/2**3),int(cin/2**5), 3)
        self.udp2 = self.down_sample(int(cin/2**3+3),int(cin/2**3), 3)
        self.u3 =  self.up_sample(int(cin/2**3),int(cin/2**3),2)
        self.ud3 = self.down_sample(int(cin/2**3), int(cin/2**3), 3)
        self.u4 =  self.up_sample(int(cin/2**3),int(cin/2**3),2)
        self.uu4 = nn.MaxPool2d(2)
        self.u22 = nn.MaxPool2d(4)
        self.ud4 = self.down_sample(int(cin/2**3), int(cin/2**3), 3)
        self.out1 = nn.Conv2d(64,3,1)
        self.out2 = nn.Conv2d(16,3,1)
        self.up = nn.Upsample(scale_factor=4, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2 = nn.Sigmoid()
        self.cl = nn.Sigmoid()
        self.d = nn.Conv2d(int(cin/2**1), cin, 17)
        for i,l in enumerate(list(self.pretrained._modules.keys())):
            if i in self.output_layers:
                self.fhooks.append(getattr(self.pretrained,l).register_forward_hook(self.forward_hook(l)))

    def down_sample(self,ci, co, k, s=1, d=1, p=1):
        block = nn.Sequential(nn.Conv2d(ci, co, kernel_size=k, stride=s, dilation=d, padding=p),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True),
        nn.Conv2d(co, co, kernel_size=k, stride=s, dilation=d, padding=p),
        nn.BatchNorm2d(co),
        nn.ReLU(inplace=True))
        return block

    def up_sample(self,ci, co, k, s=2, d=1, p=0):
       block = nn.Sequential(nn.ConvTranspose2d(ci, co, kernel_size=k, stride=s, dilation=d, padding=p),
       nn.BatchNorm2d(co),
       nn.ReLU(inplace=True))
       return block

    def forward_hook(self,layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):
        x = x.view(-1,1, x.shape[-2], x.shape[-1])


        xo = self.pretrained(x)

        y0 = self.u0(xo);print("0",y0.shape,self.selected_out['7'].shape) if self.print else None
        cropo = self.selected_out['7']
        y0 = self.d(y0);print("0",y0.shape,self.selected_out['7'].shape) if self.print else None
        y0b = self.ud0(cropo+y0)
        y1 = self.u1(y0b);print("1",y1.shape,self.selected_out['6'].shape) if self.print else None
        cropo = self.selected_out['6']
        y1b = self.ud1(cropo+y1)
        y2 = self.u2(y1b);print("2",y2.shape,self.selected_out['4'].shape) if self.print else None
        cropo = self.selected_out['5']
        y2b = self.ud2(cropo+y2)
        y2 = self.u2c(y2b);print("2",y2.shape,self.selected_out['4'].shape) if self.print else None
        cropo = self.selected_out['4']
        y2b = self.ud2c(cropo+y2)

        y2 = self.up(y2b);print("su3",y2.shape,self.selected_out['4'].shape) if self.print else None
        yo=self.out2(y2)

        yo =  self.c2(yo)
        y = nn.functional.interpolate(yo,(self.selected_out['3'].shape[2], self.selected_out['3'].shape[3]))
        cropo = self.selected_out['3']
        y2b = self.udp2(torch.cat((cropo,y2),1))
        y3 = self.u3(y2b);print("3",y3.shape,self.selected_out['2'].shape) if self.print else None
        cropo = self.selected_out['2']
        y3b = self.ud3(cropo+y3)
        y4 = self.u4(y3b);print("4",y4.shape,self.selected_out['1'].shape) if self.print else None
        y4 = self.uu4(y4);print("5",y4.shape,self.selected_out['1'].shape) if self.print else None
        cropo =self.selected_out['1']
        y4b = self.ud4(cropo+y4)
        x = self.su4(y4b);print("u4",x.shape) if self.print else None
        x=self.out1(x)
        x =  self.cl(x)
        return yo,x

    def unet1(self,y,t="u"):
        #xo = self.pretrained(y)
        y = nn.functional.interpolate(y,(self.selected_out['3'].shape[2], self.selected_out['3'].shape[3]))
        cropo = self.selected_out['3'];print("unet1",y.shape,self.selected_out['3'].shape) if self.print else None
        
        y2b = self.udp2(torch.cat((cropo,y),1))
        y3 = self.u3(y2b);print("3",y3.shape,self.selected_out['2'].shape) if self.print else None
        cropo = self.selected_out['2']
        y3b = self.ud3(cropo+y3)
        y4 = self.u4(y3b);print("4",y4.shape,self.selected_out['1'].shape) if self.print else None
        y4 = self.uu4(y4);print("5",y4.shape,self.selected_out['1'].shape) if self.print else None
        cropo =self.selected_out['1']
        y4b = self.ud4(cropo+y4)
        x = self.su4(y4b);print("u4",x.shape) if self.print else None
        x=self.out1(x)
        x =  self.cl(x)
        return x

    def unet2(self,x,t="u"):
        xo = self.pretrained(x)

        y0 = self.u0(xo);print("0",y0.shape,self.selected_out['7'].shape) if self.print else None
        cropo = self.selected_out['7']
        y0 = self.d(y0);print("0",y0.shape,self.selected_out['7'].shape) if self.print else None
        y0b = self.ud0(cropo+y0)
        y1 = self.u1(y0b);print("1",y1.shape,self.selected_out['6'].shape) if self.print else None
        cropo = self.selected_out['6']
        y1b = self.ud1(cropo+y1)
        y2 = self.u2(y1b);print("2",y2.shape,self.selected_out['4'].shape) if self.print else None
        cropo = self.selected_out['5']
        y2b = self.ud2(cropo+y2)
        y2 = self.u2c(y2b);print("2",y2.shape,self.selected_out['4'].shape) if self.print else None
        cropo = self.selected_out['4']
        y2b = self.ud2c(cropo+y2)

        y2 = self.up(y2b);print("su3",y2.shape,self.selected_out['4'].shape) if self.print else None
        y2=self.out2(y2)
        y2 =  self.c2(y2)
        return y2

    def forward_mask(self, x):
        s = x.shape
        img_size = [s[1],s[2]]
        x = x.view(-1,1, img_size[0], img_size[1])
        #x1 = self.erode(x)
        x1 = self.unet2(x)
        #x1 = nn.functional.interpolate(x1,(img_size[0], img_size[1]))
        return x1

    def forward_refine(self, x):
        s = x.shape
        img_size = [s[-2],s[-1]]
        x = x.view(-1,3, img_size[0], img_size[1])

        #x1 = self.erode(x)
        x1 = self.unet1(x)

        return x1

def save_model(name,model,file_dict,optimizer):
    save_dict = {'state_dict':model.state_dict(),
                 'data_dict':file_dict,
                 'optimizer' : optimizer.state_dict()}
    x = torch.ones(1,768,768,1).to('cuda')
    #net_trace = jit.trace(model,x)
    #jit.save(net_trace, name)
    torch.save(save_dict, name.replace('.','_torch.'))

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
    i=0
    for d in dirs:
        if "mask" in d:
            try:
                fpath = path+d
                img = cv2.imread(fpath)
                #img1 = cv2.imread(fpath.replace("_mask",""))
                if (np.sum(img)>0): #and (np.sum(img)>0):
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
    train = int(size*0.8)
    tst_val = int(size*0.2)
    #index = np.argsort(scale[0:train,0])
    #index = index.astype(int).tolist()
    #print(index)

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
    train_set.aug_indx =[3,8,9,14]
    train_loader = DataLoader(train_set, batch_size=N, shuffle=True, num_workers=N,pin_memory=True,persistent_workers=True)
    val_set = custom_dset(file_dict,'val',args)
    val_set.aug_indx =   []
    val_loader = DataLoader(val_set, batch_size=1, shuffle=True, num_workers=1,pin_memory=True,persistent_workers=True)  

    

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
                
                f1=net.forward_mask(image1s)
                loss = loss_func(f1,outputs)
                f2 = net.forward_refine(f1)
                loss += loss_func(f2,torch.abs(outputs-f1))
                
                loss.backward()
                running_loss += loss
                optimizer.step()
                outputs = outputs.view(-1,3,shape,shape)
            
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

                    f1,f2=net(image1s)
                    #f1=net.forward_refine(image1s,f1)
                    f1 = f1+f2
                    loss = loss_func(f1,outputs)

                    val_running_loss += loss
                    #f1_conv = torch.argmax(torch.nn.functional.softmax(f1, 1), dim=1)
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
            print('[%d] train_loss: %.8f  val_loss %.8f  val_acc %.8f' %
                  (epoch + 1, running_loss, val_loss, val_acc))


            if val_loss < acc:
                save_model(name,net,file_dict,optimizer)
                acc = val_loss
                print("model_saved")
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
        file_dict = eval_set('C:\\Updated_ML\\localize\\')
        eval_set = custom_dset(file_dict,'eval',args)
        train_loader = DataLoader(eval_set, batch_size=1, shuffle=False, num_workers=1,pin_memory=True,persistent_workers=True)
        for i,data in enumerate(train_loader, 0):
            image1s,imp_shape,name=data
            if torch.cuda.is_available():
                image1s = image1s.cuda()

            image1s = image1s.squeeze(0).unsqueeze(3)
            
            f1=net(image1s.float())
            image1s = image1s.view(1,-1,shape,shape)
            outputs = f1.view(1,-1,shape,shape)
            print(image1s.shape,outputs.shape)
            image1s = reconstruct_img(image1s,imp_shape,shape)
            outputs = reconstruct_img(outputs,imp_shape,shape,'mask')
            outputs = np.array(np.round(outputs.detach().cpu().numpy()))*255
            image1s = np.array(image1s.detach().cpu().numpy())*255
            concatimg = np.concatenate((outputs.astype('uint8'),image1s.astype('uint8')))
            n = name[0].replace(".jpg",'').replace('.bmp','')
            cv2.imwrite("eval\\out\\"+n+"_out.jpg",concatimg)
            if 'jpg' in name[0]:
                cv2.imwrite("eval\\mask\\"+n+"_mask.jpg",outputs.astype('uint8'))
            else:
                cv2.imwrite("eval\\mask\\"+n+"_mask.bmp",outputs.astype('uint8'))
            print(i)

   