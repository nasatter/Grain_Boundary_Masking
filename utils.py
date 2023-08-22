import torch
import copy
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2, csv,json,random,math
from torch.utils.data import Dataset

# This class takes the intial shot guess and the training set
# to optimize the few-shot data set
# Theory: Outliers and miss-classifications included in the few-shot list can 
# 		  dranstically reduce the classifier performance especially when 
#		  few (~1-10) shot is employed. It is expected that by selecting
#		  the few-shot comparison to be within a range of the mean cluster
#		  we can eliminate the outlier inclusion.
# Inputs:
#		data_dict:		dictionary containing the sets
# 
# Outputs: dictionary containing updated sets
#
# Fuctions: 
#		load_batch:    loads batch data and cats to tensor for each class
#		load_shot:     loads shot data and cats to tensor for each class
#		clear_batch:   clears all data
#		optimize_shot: runs clustering and randomly selects candidates
#					   with a specified range	
#		upate_dict: updates the dictionary with new shot guess
#
class optimize_shot():
	def __init__(self, data_dict,num_shot,skip_iter=0,iterations=25, greedy = False):
		self.data_dict = data_dict
		self.num_classes = len(data_dict['train'].keys())
		self.train_clusters = [[]]*self.num_classes
		self.img_names = [[]]*self.num_classes
		self.num_shot = num_shot
		self.iterations = iterations+skip_iter
		self.skip_iter = skip_iter
		self.num_iters = 0
		self.greedy = greedy

	def active(self):
		return ((self.num_iters < self.iterations) and (self.skip_iter<=self.num_iters))


	def load_batch(self,batch_feature,batch_class,img_name):
		if self.active():
			batch_feature = batch_feature.clone().detach().cpu()
			batch_class = batch_class.clone().detach().cpu()
			for i in range(self.num_classes):
				index = (batch_class==i)
				index = index.nonzero()
				index = index[:,0]
				if len(self.train_clusters[i])==0:
					self.train_clusters[i] = batch_feature[index,:]
					#print(img_name,index)
					self.img_names[i] = [img_name[j] for j in index]
				else:
					#print(self.img_names[i],img_name)
					self.train_clusters[i] = torch.cat((self.train_clusters[i],batch_feature[index,:]),dim=0)
					self.img_names[i] += [img_name[j] for j in index]



	def optimize_shot(self):
		if self.active():
			for i in range(self.num_classes):
				all_training=self.train_clusters[i]
				cluster_center = torch.mean(all_training,dim=0)
				distance = F.cosine_similarity(cluster_center,all_training)
				std_dev = torch.std(distance)
				average = torch.mean(distance)
				mx = torch.max(distance)
				mn = torch.min(distance)
				index_class = self.data_dict['index_class'][i]
				keep = self.check_shot(self.img_names[i],self.data_dict['shot'][index_class],distance,std_dev,average)
				if (self.num_shot-len(keep))>0:
					if self.greedy:
						_,candidates = distance.sort()
						perm = candidates[-(self.num_shot-len(keep)):]
					else:
						candidates = torch.gt(distance,average+std_dev/2).nonzero()
						bias = std_dev/2
						iters = 0
						while candidates.shape[0] < self.num_shot:
							bias = bias*1.1
							candidates = torch.gt(distance,average+bias).nonzero()
							print(candidates.shape[0],bias,average)
							# perform greedy if no candidates
							if iters>10:
								_,candidates = distance.sort()
								perm = candidates[-(self.num_shot-len(keep)):]

							iters+=1
							
						perm = torch.randperm(candidates.shape[0])[0:self.num_shot-len(keep)]
					
					selected = [self.img_names[i][j] for j in perm]+keep
					print(mx,mn,std_dev,average,candidates.shape,len(keep),len(selected),len(self.data_dict['shot'][index_class]))
					
					self.data_dict['shot'][index_class]=selected
					#if (self.num_iters+1) == self.iterations:
					#	self.data_dict['train'][index_class]=selected
			self.clear_batch()
		self.num_iters += 1
		
		return self.data_dict

	def check_shot(self,shot_list,img_list,distance,std_dev,average):
		shot_index = torch.tensor([(i in shot_list)*ind for ind,i in enumerate(img_list)]).nonzero().squeeze()
		check = torch.gt(distance[shot_index],average).nonzero()
		keep = [shot_list[i] for i in check]
		#print(keep)
		return keep

	def clear_batch(self):
		#print(self.clusters)
		self.train_clusters = [[]]*self.num_classes
		self.img_names = [[]]*self.num_classes


class custom_dset(Dataset):
    def __init__(self,
                 file_dict,
                 study,
                 args
                 ):
        self.args = args
        self.study = study
        self.file_dict = file_dict
        self.size1 = args.input_shape
        self.crop1 = int(args.input_shape/4)
        self.size = int(args.input_shape*1.5)
        self.crop = args.input_shape
        self.epoch = 0 
        self.bg = 0

        self.decay = args.decay
        self.mosiac = args.mosiac_prob
        self.aug = args.augment_prob
        self.channels = args.channels
        self.aug_val = args.augment_validation
        self.aug_indx =             [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
        self.D = Distort()

        
    def __getitem__(self, index):
        if (self.study == 'eval') or ((self.study=="val") and not self.aug_val):
            try:
                img_path = self.file_dict["val"]['inp'][index]
            except:
                img_path = self.file_dict["eval"]['inp'][index]
            img1 = cv2.imread(img_path)
            if self.study == 'val':
                mask_path = self.file_dict[self.study]['out'][index]
                inps = self.file_dict[self.study]['scale'][index]
                msk1 = cv2.imread(mask_path)
                msk1 = cv2.resize(msk1,[self.size1,self.size1])
                if self.channels==1:
                    msk1 = cv2.cvtColor(msk1, cv2.COLOR_BGR2GRAY)
                    ret, msk1 = cv2.threshold(msk1, 20., 255., cv2.THRESH_BINARY)
                    #msk1 = cv2.resize(msk1,[self.size1,self.size1])
                    msk1 = msk1.astype(float)/255
                else:
                    msk = []
                    for i in range(self.channels):
                        temp = cv2.resize(msk1[:,:,i],[self.size1,self.size1])#msk1[:,:,i]
                        ret, temp = cv2.threshold(temp, 20., 255., cv2.THRESH_BINARY)
                        msk.append(temp.astype(float)/255)
                    msk1 = msk

                img1=cv2.resize(img1,[self.size1,self.size1])
                
                
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                img1 = img1.astype(float)/255
                msk1 = np.stack(msk1,axis=0)
                return img1,inps,msk1
            else:
                # we need to split up the image to maintain same input size
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                img1 = cv2.resize(img1,[self.size1,self.size1])

                #img1 = img1.astype(float)/255
                #img_out = []
                #skip = int(3*self.size1/4)
                #y1=0;y2=self.size1
                #while y2<img1.shape[0]:
                #    x1=0;x2=self.size1
                #    while x2<img1.shape[1]:
                #        img_out.append(img1[y1:y2,x1:x2])
                #        x1+=skip; x2+=skip
                #    if x1<img1.shape[1]:
                #        x2 = img1.shape[1]
                #        x1 = x2-self.size1
                #        img_out.append(img1[y1:y2,x1:x2])
                #    y1+=skip; y2+=skip
                #if y1<img1.shape[0]:
                #    y2 = img1.shape[0]
                #    y1 = y2-self.size1
                #    x1=0; x2=self.size1
                #    while x2<img1.shape[1]:
                #        img_out.append(img1[y1:y2,x1:x2])
                #        x1+=skip; x2+=skip
                #    if x1<img1.shape[1]:
                #        x2 = img1.shape[1]
                #        x1 = x2-self.size1
                #        img_out.append(img1[y1:y2,x1:x2])
                #return np.array(img_out).astype(float)/255,img1.shape,img_path.split("\\")[-1]
                return img1.astype(float)/255,img1.shape,img_path.split("\\")[-1]
        else:
            shape = self.args.input_shape
            crop = self.crop1
            size = self.size1
            mosiac = self.mosiac
            aug = self.aug

            nindex = index
            weight = np.exp(-np.log(2)/self.decay*self.epoch)

            img_arr = np.array(np.zeros((shape,shape)),dtype=np.uint8)
            mask_arr = np.array(np.zeros((shape,shape)),dtype=np.uint8)
            space=1
            i=0
            iters = random.randint(175,350)
            if random.random()<self.args.mosiac_prob:
                while (space>0):
                    if i+index < self.__len__():
                        nindex = i+index
                        i+=1
                    else:
                        break
                    if i>iters:
                        break
                    img_path = self.file_dict[self.study]['inp'][nindex]
                    inps = self.file_dict[self.study]['scale'][nindex]
                    inps = [int(inps[0]),int(inps[1])]
                    mask_path = self.file_dict[self.study]['out'][nindex]
                    img1 = cv2.imread(img_path)
                    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                    if i==1:
                        self.bg = np.mean(img1)
                    msk1 = cv2.imread(mask_path)
                    msk1 = cv2.cvtColor(msk1, cv2.COLOR_BGR2GRAY)

                    

                    ret, msk1 = cv2.threshold(msk1, 10, 255, cv2.THRESH_BINARY)
                    img1,msk1 = self.rotate(img1,msk1)
                    if random.random()>0.5:
                        img1,msk1 = self.rand_rotate(img1,msk1)
                    if img1.shape[0]>shape or img1.shape[1]>shape:
                        msk1 = cv2.resize(msk1,[int(img1.shape[1]/2),int(img1.shape[0]/2)])
                        img1 = cv2.resize(img1,[int(img1.shape[1]/2),int(img1.shape[0]/2)],interpolation=cv2.INTER_CUBIC)
                    rect = self.fit_image(img1,img_arr)

                    #print(rect,inps)
                    if np.sum(rect)>0:
                        space=1

    
                    else:
                        space = 0
                        index = nindex
    
                    if space==1:
                        img_arr[rect[0]:rect[2],rect[1]:rect[3]] = np.clip(img1,1,255)
                        mask_arr[rect[0]:rect[2],rect[1]:rect[3]] = msk1

                img_arr[img_arr==0] = self.bg
    
                img1 = img_arr
                msk1 = mask_arr
            else:
                img_path = self.file_dict[self.study]['inp'][nindex]
                inps = self.file_dict[self.study]['scale'][nindex]
                inps = [int(inps[0]),int(inps[1])]
                mask_path = self.file_dict[self.study]['out'][nindex]
                img1 = cv2.imread(img_path)
                img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
                
                msk1 = cv2.imread(mask_path)
                if self.channels==1:
                    msk1 = cv2.cvtColor(msk1, cv2.COLOR_BGR2GRAY)
                    ret, msk1 = cv2.threshold(msk1, 20., 255., cv2.THRESH_BINARY)
                    msk1 = cv2.resize(msk1,[self.size1,self.size1])
                else:
                    msk = []
                    for i in range(self.channels):
                        temp = msk1[:,:,i]
                        ret, temp = cv2.threshold(temp, 20., 255., cv2.THRESH_BINARY)
                        msk.append(temp.astype(float)/255)
                    msk1 = msk
                #


            aug_prob = np.array([1000,0.5,1,2,1,1,1,1,1,1,1,1,1,1000,1,10])*self.args.augment_prob*weight
            aug_list = [self.distort_inps,        #0
                        self.invert_corners,      #1
                        self.heal_mask,           #2
                        self.gamma,               #3
                        self.random_gradient,     #4
                        self.heal_lines,          #5
                        self.rand_warp,           #6
                        self.rand_rotate,         #7
                        self.mask_lines,          #8
                        self.color_lines,         #9
                        self.erase_perim,         #10
                        self.erode,               #11
                        self.dilate,              #12
                        self.random_crop,         #13
                        self.add_dots,            #14
                        self.rotate]              #15
            random.shuffle(self.aug_indx)
            for augmen_idx in [7,13,15]:
                augmen_fun = aug_list[augmen_idx]
                if (random.random()<aug_prob[augmen_idx]) or (augmen_idx==13):
                    img1,msk1 = augmen_fun(img1,msk1)
                if random.random()<0.5:
                    img1,msk1 = self.distort_inps(img1,msk1)

            img2 = img1.copy()    
            img3 = img1.copy()  
            img4 = img1.copy()    

            #img6 = img1.copy()  
            for augmen_idx in self.aug_indx:
                augmen_fun = aug_list[augmen_idx]
                if (random.random()<aug_prob[augmen_idx]):
                    img1,_ = augmen_fun(img1)

                if (random.random()<aug_prob[augmen_idx]):
                    img2,_ = augmen_fun(img2)

                if (random.random()<aug_prob[augmen_idx]):
                    img3,_ = augmen_fun(img3)

                if (random.random()<aug_prob[augmen_idx]):
                    img4,_ = augmen_fun(img4)


            img1 = cv2.resize(img1,[self.size1,self.size1],interpolation=cv2.INTER_CUBIC)
            img2 = cv2.resize(img2,[self.size1,self.size1],interpolation=cv2.INTER_CUBIC)
            img3 = cv2.resize(img3,[self.size1,self.size1],interpolation=cv2.INTER_CUBIC)
            img4 = cv2.resize(img4,[self.size1,self.size1],interpolation=cv2.INTER_CUBIC)
            #img5 = cv2.resize(img5,[self.size1,self.size1],interpolation=cv2.INTER_CUBIC)
            #img6 = cv2.resize(img6,[self.size1,self.size1],interpolation=cv2.INTER_CUBIC)

            img1 = img1.astype(float)/255
            img2 = img2.astype(float)/255
            img3 = img3.astype(float)/255
            img4 = img4.astype(float)/255
            #img5 = img5.astype(float)/255
            #img6 = img6.astype(float)/255
#
            
            #img2 = img2.astype(float)/255

            for i in range(self.channels):
                temp = msk1[i]
                msk1[i] = cv2.resize(temp,[self.size1,self.size1])
            msk1 = np.stack(msk1,axis=0)
            return img1,inps,msk1,img2,img3,img4#,img5,img6

    def __len__(self):
        if self.study == 'eval':
            try:
                return len(self.file_dict['val']['inp'])
            except:
                return len(self.file_dict['eval']['inp'])
        return len(self.file_dict[self.study]['inp'])

    def rotate(self,img1,msk1):
        t = random.randint(0,4)
        if t==0:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_CLOCKWISE)
            for i in range(self.channels):
                msk1[i] = cv2.rotate(msk1[i], cv2.ROTATE_90_CLOCKWISE)
        elif t==1:
            img1 = cv2.rotate(img1, cv2.ROTATE_180)
            for i in range(self.channels):
                msk1[i] = cv2.rotate(msk1[i], cv2.ROTATE_180)
        elif t==2:
            img1 = cv2.rotate(img1, cv2.ROTATE_90_COUNTERCLOCKWISE)
            for i in range(self.channels):
                msk1[i] = cv2.rotate(msk1[i], cv2.ROTATE_90_COUNTERCLOCKWISE)
        if random.random()<0.5:
            img1 = cv2.flip(img1,1)
            for i in range(self.channels):
                msk1[i] = cv2.flip(msk1[i],1)
        return img1,msk1

    def erode(self,img1,msk1=None):
        s = np.random.randint(2,10,(2,))
        kernel = np.ones(s,np.uint8)
        imc = cv2.erode(img1.copy(),kernel,iterations = 1)
        cropy = np.random.randint(5, int(img1.shape[1]))
        y = np.random.randint(0, img1.shape[1] - cropy)
        cropx = np.random.randint(5, int(img1.shape[0]))
        x = np.random.randint(0, img1.shape[0] - cropx)
        img1[x:x + cropx, y:y + cropy]=imc[x:x + cropx, y:y + cropy]
        return img1, msk1

    def dilate(self,img1,msk1=None):
        s = np.random.randint(2,10,(2,))
        kernel = np.ones(s,np.uint8)
        imc = cv2.dilate(img1.copy(),kernel,iterations = 1)
        cropy = np.random.randint(5, int(img1.shape[1]))
        y = np.random.randint(0, img1.shape[1] - cropy)
        cropx = np.random.randint(5, int(img1.shape[0]))
        x = np.random.randint(0, img1.shape[0] - cropx)
        img1[x:x + cropx, y:y + cropy]=imc[x:x + cropx, y:y + cropy]
        return img1, msk1

    def erase_perim(self,img1,msk1=None):
        msk = np.zeros(shape=img1.shape,dtype=np.uint8)
        if random.random()>0.25:
            c = np.random.randint(2,int(img1.shape[0]/20))
            msk[0:c,:] = 1
        if random.random()>0.25:
            c = np.random.randint(2,int(img1.shape[0]/20))
            msk[img1.shape[0]-c:img1.shape[0],:] = 1
        if random.random()>0.25:
            c = np.random.randint(2,int(img1.shape[0]/20))
            msk[:,0:c] = 1
        if random.random()>0.25:
            c = np.random.randint(2,int(img1.shape[0]/20))
            msk[:,img1.shape[1]-c:img1.shape[1]] = 1
        msk, _ = self.distort_inps(msk)
        img1[msk>0] = 0
        return img1, msk1

    def add_dots(self,img1,msk1=None):
        msk = np.zeros(shape=img1.shape,dtype=np.uint8)
        alpha = random.random()*4
        beta = random.random()*20
        img_c = cv2.convertScaleAbs(img1, alpha=alpha, beta=beta)
        lines = int(np.random.normal(100,100,1))
        if lines<1:
            lines=1
        a = np.random.randint(0,img1.shape[0],(lines,2))
        b = np.random.randint(0,2,(lines,))
        #c = np.random.randint(1,10,(lines,))
        c = np.random.normal(5, 3, (lines,))
        c[c<1]=1
        rgb = np.random.randint(0,255,(lines,3),dtype='uint8')
        for i in range(lines):
            color = ( int (rgb [i][ 0 ]), int (rgb [i][ 1 ]), int (rgb [i][ 2 ])) 
            cv2.circle(img1, a[i], b[i], color, int(c[i]))
        return img1, msk1

    def random_crop(self,img1,msk1):
        s = img1.shape
        if (s[0]<self.size1) and (s[0]<self.size1):
            crop_shapex = random.randint(int(self.size1),s[1]-1)
            crop_shapey = random.randint(int(self.size1),s[0]-1)
            y = np.random.randint(0, s[0] - crop_shapey)
            x = np.random.randint(0, s[1] - crop_shapex)
            #print(x,y,crop_shapex,crop_shapey)
            img1 = img1[x:x+crop_shapex,y:y+crop_shapey]
        
        img1 = cv2.resize(img1,[self.size1,self.size1])
        for i in range(self.channels):
            temp = msk1[i]
            if (s[0]<self.size1) and (s[0]<self.size1):
                temp = temp[x:x+crop_shapex,y:y+crop_shapey]
            msk1[i] = cv2.resize(temp,[self.size1,self.size1])
        return img1,msk1

    def invert_corners(self,img1,msk1):
        s = img1.shape
        scale1 = s[1]/s[0]
        crop_shapex= int(s[1]/2)
        crop_shapey = int(s[0]/2)
        img_crop = cv2.flip(img1[0:crop_shapex,0:crop_shapey],-1)
        img1[0:crop_shapex,0:crop_shapey]=img_crop
        img_crop = cv2.flip(img1[s[0]-crop_shapex:s[0],0:crop_shapey],-1)
        img1[s[0]-crop_shapex:s[0],0:crop_shapey]=img_crop
        img_crop = cv2.flip(img1[0:crop_shapex,s[1]-crop_shapey:s[1]],-1)
        img1[0:crop_shapex,s[1]-crop_shapey:s[1]]=img_crop
        img_crop = cv2.flip(img1[s[0]-crop_shapex:s[0],s[1]-crop_shapey:s[1]],-1)
        img1[s[0]-crop_shapex:s[0],s[1]-crop_shapey:s[1]]=img_crop
        if msk1 is not None:
        	msk_crop = cv2.flip(msk1[0:crop_shapex,0:crop_shapey],-1)
        	msk1[0:crop_shapex,0:crop_shapey]=msk_crop
        	msk_crop = cv2.flip(msk1[s[0]-crop_shapex:s[0],0:crop_shapey],-1)
        	msk1[s[0]-crop_shapex:s[0],0:crop_shapey]=msk_crop
        	msk_crop = cv2.flip(msk1[0:crop_shapex,s[1]-crop_shapey:s[1]],-1)
        	msk1[0:crop_shapex,s[1]-crop_shapey:s[1]]=msk_crop
        	msk_crop = cv2.flip(msk1[s[0]-crop_shapex:s[0],s[1]-crop_shapey:s[1]],-1)
        	msk1[s[0]-crop_shapex:s[0],s[1]-crop_shapey:s[1]]=msk_crop
        return img1,msk1

    def heal_mask(self,img1,msk1):
        msk = np.zeros(shape=img1.shape,dtype=np.uint8)
        bg_msk1 = cv2.dilate(msk1, np.ones((8, 8), np.uint8), iterations=1)
        n = np.random.randint(1,5)
        fixed_img = cv2.inpaint(img1.astype(np.uint8),bg_msk1.astype(np.uint8),n,cv2.INPAINT_NS)
        lines = random.randint(1,5)
        a = np.random.randint(0,img1.shape[0],(lines,2))
        b = np.random.randint(0,img1.shape[1],(lines,2))
        c = np.random.randint(2,int(img1.shape[0]/20),(lines,))
        rgb = [255,255,255]
        for i in range(lines):
            cv2.line(msk, a[i], b[i], tuple (rgb), c[i])
        lines = random.randint(1,5)
        a = np.random.randint(0,img1.shape[0],(lines,2))
        b = np.random.randint(0,img1.shape[1],(lines,))
        c = np.random.randint(2,int(img1.shape[0]/20),(lines,))
        for i in range(lines):
            cv2.circle(msk, a[i], b[i], tuple (rgb), c[i])
        index = msk==255
        #fixed_img = cv2.flip(fixed_img,-1)
        img1[index] = fixed_img[index]
        return img1,msk1

    def random_gradient(self,img1,msk1=None):
        indx = np.where(np.reshape(img1*0==0,img1.shape))[0]
        indx = np.reshape(indx,img1.shape)
        sc = np.ones(indx.shape)
        for i in range(4):
            if random.random()<0.5:
                if random.random()<0.5:
                    grad = random.random()*2
                    sc *= (indx/indx.shape[0])**grad
                else:
                    
                    base = np.random.randint(3,100)
                    grad = np.random.random()*base
                    sc *= np.log(indx/np.max(indx)*grad+1)/np.log(base)
                sc = cv2.rotate(sc, cv2.ROTATE_90_CLOCKWISE)
        sc/=np.max(sc)
        if random.random()<0.5:
            sc=1-sc
        grad = random.random()*0.1
        return (img1*sc*(1-grad)+sc*grad).astype(np.uint8),msk1

    def random_gamma(self,img1,msk1=None):
        for i in range(np.random.randint(4)):
            crop1 = np.random.randint(10,img1.shape[0]-10)
            crop2 = np.random.randint(10,img1.shape[1]-10)
            y = np.random.randint(0, img1.shape[0] - crop1)
            x = np.random.randint(0, img1.shape[1] - crop2)
            img_c = img1[x:x+crop2,y:y+crop1]
            alpha = random.random()*4
            beta = random.random()*20
            img_c = cv2.convertScaleAbs(img_c, alpha=alpha, beta=beta)
            img1[x:x+crop2,y:y+crop1] = img_c
        return img1,msk1

    def gamma(self,img,msk1=None):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        mean = np.mean(img)
        mu, sigma = 0.5, 0.1
        mid = np.random.normal(mu, sigma, 1)[0].clip(1e-4,1-1e-4)
        gamma = math.log(mid*255)/math.log(mean)
        img = np.power(img, gamma).clip(0,255).astype(np.uint8)

        return img,msk1

    def heal_lines(self,img1,msk1=None):
        lines = random.randint(1,5)
        a = np.random.randint(0,img1.shape[0],(lines,2))
        b = np.random.randint(0,img1.shape[1],(lines,2))
        c = np.random.randint(2,int(img1.shape[0]/20),(lines,))
        rgb = np.mean(img1)
        for i in range(lines):
            color = ( int (rgb ), int (rgb ), int (rgb )) 
            cv2.line(img1, a[i], b[i], tuple (color), c[i])
        lines = random.randint(1,5)
        a = np.random.randint(0,img1.shape[0],(lines,2))
        b = np.random.randint(0,img1.shape[1],(lines,))
        c = np.random.randint(2,int(img1.shape[0]/20),(lines,))
        rgb = np.mean(img1)
        for i in range(lines):
            color = ( int (rgb ), int (rgb ), int (rgb )) 
            cv2.circle(img1, a[i], b[i], tuple (color), c[i])
        return img1,msk1

    def rand_warp(self,img1,msk1):
        bg = 0
        angle=random.random()*360-180
        if len(img1.shape)==3:
            rows,cols,cn = img1.shape
        else:
            rows,cols = img1.shape
        c = np.float32([[0,0],[cols,0],[0,rows],[cols,rows]])
        r = np.float32([[random.randint(0,int(cols/4)),random.randint(0,int(rows/4))],[random.randint(int(cols*3/4),cols),random.randint(0,int(rows/4))],[random.randint(0,int(cols/4)),random.randint(int(rows*3/4),rows)],[random.randint(int(cols*3/4),cols),random.randint(int(rows*3/4),rows)]])
        M = cv2.getPerspectiveTransform(c,r)
        img = cv2.flip(img1.copy(),-1)
        img1[img1<255] = img1[img1<255]+1
        img1 = cv2.warpPerspective(img1,M,(cols,rows),borderValue = bg)
        img1[img1==0] = img[img1==0]
        msk = cv2.flip(msk1.copy(),-1)
        bg = 1
        msk1 = cv2.warpPerspective(msk1,M,(cols,rows),borderValue = bg)
        msk1[msk1==1] = msk[msk1==1]
        return img1,msk1

    def rand_rotate(self,img1,msk1):
        bg = self.bg
        angle=random.random()*45
        if len(img1.shape)==3:
            rows,cols,cn = img1.shape
        else:
            rows,cols = img1.shape
        
        wl,hl,wi,hi = rotatedRectWithMaxArea(cols,rows,math.radians(angle))
        img1 = np.array(Image.fromarray(img1).rotate(angle,expand=True))
        img1 = img1[int(hi):int(hi+hl),int(wi):int(wi+wl)]
        img1 = cv2.resize(img1,(cols,rows))
        if msk1 is not None:
            for i in range(self.channels):
                msk1[i] = np.array(Image.fromarray(msk1[i]).rotate(angle,expand=True))
                msk1[i] = msk1[i][int(hi):int(hi+hl),int(wi):int(wi+wl)]
                msk1[i] = cv2.resize(msk1[i],(cols,rows))
        return img1,msk1

    def mask_lines(self,img1,msk1=None):
        lines = random.randint(1,10)
        rgb = np.mean(img1)
        msk = np.zeros(shape=img1.shape,dtype=np.uint8)
        a = np.random.randint(0,img1.shape[0],(lines,2))
        b = np.random.randint(0,img1.shape[1],(lines,2))
        c = np.random.randint(3,int(img1.shape[0]/20),(lines,))
        for i in range(lines):
            cv2.line(msk, a[i], b[i], [255,255,255], c[i])
            color = ( int (rgb ), int (rgb ), int (rgb )) 
            cv2.line(img1, a[i], b[i], tuple (color), c[i])
        lines = random.randint(1,5)
        a = np.random.randint(0,img1.shape[0],(lines,2))
        b = np.random.randint(0,img1.shape[1],(lines,))
        c = np.random.randint(3,int(img1.shape[0]/20),(lines,))
        for i in range(lines):
            color = [255,255,255]
            cv2.circle(msk, a[i], b[i], tuple (color), c[i])
            color = ( int (rgb ), int (rgb ), int (rgb )) 
            cv2.circle(img1, a[i], b[i], tuple (color), c[i])
        n = np.random.randint(5,15)
        n = n+(n%2+1)
        msk = cv2.GaussianBlur(msk,(n,n),0.35)
        msk,_ = self.distort_inps(msk)
        n = np.random.randint(2,7)
        img1 = cv2.inpaint(img1,msk,n,cv2.INPAINT_NS)
        return img1,msk1

    def color_lines(self,img1,msk1=None):
        lines = random.randint(1,5)
        a = np.random.randint(0,img1.shape[0],(lines,2))
        b = np.random.randint(0,img1.shape[1],(lines,2))
        c = np.random.randint(2,int(img1.shape[0]/20),(lines,))
        rgb = np.random.randint(0,255,(lines,3),dtype='uint8')
        for i in range(lines):
            color = ( int (rgb [i][ 0 ]), int (rgb [i][ 1 ]), int (rgb [i][ 2 ])) 
            cv2.line(img1, a[i], b[i], tuple (color), c[i])
        lines = random.randint(1,5)
        a = np.random.randint(0,img1.shape[0],(lines,2))
        b = np.random.randint(0,img1.shape[1],(lines,))
        c = np.random.randint(2,int(img1.shape[0]/20),(lines,))
        rgb = np.random.randint(0,255,(lines,3),dtype='uint8')
        for i in range(lines):
            color = ( int (rgb [i][ 0 ]), int (rgb [i][ 1 ]), int (rgb [i][ 2 ])) 
            cv2.circle(img1, a[i], b[i], tuple (color), c[i])
        return img1,msk1

    def distort_inps(self,img1,msk1=None):
        img1 = Image.fromarray(np.uint8(img1))
        self.D.build_grid(img1)
        img1 =  np.asarray(self.D.distort_img(img1))
        if msk1 is not None:
            for i in range(self.channels):
               msk1[i] =  np.asarray(self.D.distort_img(Image.fromarray(msk1[i])))
        return img1,msk1

    def fit_image(self,mask,mask_array):
        kernel = torch.ones(mask.shape)
        stride_len = random.randint(5,13)
        l=F.conv2d(torch.tensor(mask_array).float().unsqueeze(0).unsqueeze(0),kernel.unsqueeze(0).unsqueeze(0),stride=stride_len)
        indexes = torch.where(l.squeeze(0).squeeze(0)==0)
        if torch.sum(indexes[0])==0:
            return [0,0,0,0]
        index = torch.randint(0,len(indexes[0]),(1,))
        y1x1y2x2 = [indexes[0][index].item()*stride_len,indexes[1][index].item()*stride_len,indexes[0][index].item()*stride_len+kernel.shape[0],indexes[1][index].item()*stride_len+kernel.shape[1]]
        return y1x1y2x2

class Distort():
    def quad_as_rect(self,quad):
        if quad[0] != quad[2]: return False
        if quad[1] != quad[7]: return False
        if quad[4] != quad[6]: return False
        if quad[3] != quad[5]: return False
        return True
    
    def quad_to_rect(self,quad):
        assert(len(quad) == 8)
        assert(self.quad_as_rect(quad))
        return (quad[0], quad[1], quad[4], quad[3])
    
    def rect_to_quad(self,rect):
        assert(len(rect) == 4)
        return (rect[0], rect[1], rect[0], rect[3], rect[2], rect[3], rect[2], rect[1])
    
    def shape_to_rect(self,shape):
        assert(len(shape) == 2)
        return (0, 0, shape[0], shape[1])
    
    def griddify(self,rect, w_div, h_div):
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        x_step = w / float(w_div)
        y_step = h / float(h_div)
        y = rect[1]
        grid_vertex_matrix = []
        for _ in range(h_div + 1):
            grid_vertex_matrix.append([])
            x = rect[0]
            for _ in range(w_div + 1):
                grid_vertex_matrix[-1].append([int(x), int(y)])
                x += x_step
            y += y_step
        grid = np.array(grid_vertex_matrix)
        return grid
    
    def distort_grid(self,org_grid, max_shift):
        new_grid = np.copy(org_grid)
        x_min = np.min(new_grid[:, :, 0])
        y_min = np.min(new_grid[:, :, 1])
        x_max = np.max(new_grid[:, :, 0])
        y_max = np.max(new_grid[:, :, 1])
        new_grid += np.random.randint(- max_shift, max_shift + 1, new_grid.shape)
        new_grid[:, :, 0] = np.maximum(x_min, new_grid[:, :, 0])
        new_grid[:, :, 1] = np.maximum(y_min, new_grid[:, :, 1])
        new_grid[:, :, 0] = np.minimum(x_max, new_grid[:, :, 0])
        new_grid[:, :, 1] = np.minimum(y_max, new_grid[:, :, 1])
        return new_grid
    
    def grid_to_mesh(self,src_grid, dst_grid):
        assert(src_grid.shape == dst_grid.shape)
        mesh = []
        for i in range(src_grid.shape[0] - 1):
            for j in range(src_grid.shape[1] - 1):
                src_quad = [src_grid[i    , j    , 0], src_grid[i    , j    , 1],
                            src_grid[i + 1, j    , 0], src_grid[i + 1, j    , 1],
                            src_grid[i + 1, j + 1, 0], src_grid[i + 1, j + 1, 1],
                            src_grid[i    , j + 1, 0], src_grid[i    , j + 1, 1]]
                dst_quad = [dst_grid[i    , j    , 0], dst_grid[i    , j    , 1],
                            dst_grid[i + 1, j    , 0], dst_grid[i + 1, j    , 1],
                            dst_grid[i + 1, j + 1, 0], dst_grid[i + 1, j + 1, 1],
                            dst_grid[i    , j + 1, 0], dst_grid[i    , j + 1, 1]]
                dst_rect = self.quad_to_rect(dst_quad)
                mesh.append([dst_rect, src_quad])
        return mesh

    def build_grid(self,im):
        dst_grid = self.griddify(self.shape_to_rect(im.size), np.random.randint(3,5),np.random.randint(3,5))
        src_grid = self.distort_grid(dst_grid, np.random.randint(5,int(im.size[0]/20)))
        self.mesh = self.grid_to_mesh(src_grid, dst_grid)

    def distort_img(self,im):
        return im.transform(im.size, Image.MESH, self.mesh)


# Reconstruct the image with a skip of shape/2
# the outputs are added to the canvas so overlapping
# areas are considered. The final output is clipped
# so it is in the range of 0,255
def reconstruct_img(image,inp_shape,shape,case="img"):
    img1=torch.zeros(inp_shape)
    image = image.view(-1,shape,shape,1)
    print(img1.shape,image.shape)
    skip = int(3*shape/4)
    trim = int((shape-skip)/2)
    if torch.cuda.is_available():
        img1=img1.cuda()
    y1=0;y2=shape;i=0
    while y2<inp_shape[0]:
        x1=0;x2=shape
        while x2<inp_shape[1]:
            if case == 'mask':
                img1[y1+trim*(y1!=0):y2-trim*(y2!=inp_shape[0]),x1+trim*(x1!=0):x2-trim*(x2!=inp_shape[1])]+=image[0,i,trim*(y1!=0):shape-trim*(y2!=inp_shape[0]),trim*(x1!=0):shape-trim*(x2!=inp_shape[1])]
            else:
                img1[y1:y2,x1:x2]=image[0,i,:,:]
            x1+=skip; x2+=skip
            i+=1
        if x1<=inp_shape[1]:
            x2 = inp_shape[1]
            x1 = x2-shape
            if case == 'mask':
                img1[y1+trim*(y1!=0):y2-trim*(y2!=inp_shape[0]),x1+trim*(x1!=0):x2-trim*(x2!=inp_shape[1])]+=image[0,i,trim*(y1!=0):shape-trim*(y2!=inp_shape[0]),trim*(x1!=0):shape-trim*(x2!=inp_shape[1])]
            else:
                img1[y1:y2,x1:x2]=image[0,i,:,:]
            i+=1
        y1+=skip; y2+=skip
    if y1<=inp_shape[0]:
        y2 = inp_shape[0]
        y1 = y2-shape
        x1=0; x2=shape
        while x2<inp_shape[1]:
            if case == 'mask':
                img1[y1+trim*(y1!=0):y2-trim*(y2!=inp_shape[0]),x1+trim*(x1!=0):x2-trim*(x2!=inp_shape[1])]+=image[0,i,trim*(y1!=0):shape-trim*(y2!=inp_shape[0]),trim*(x1!=0):shape-trim*(x2!=inp_shape[1])]
            else:
                img1[y1:y2,x1:x2]=image[0,i,:,:]
            i+=1
            x1+=skip; x2+=skip
        if x1<=inp_shape[1]:
            x2 = inp_shape[1]
            x1 = x2-shape
            if case == 'mask':
                img1[y1+trim*(y1!=0):y2-trim*(y2!=inp_shape[0]),x1+trim*(x1!=0):x2-trim*(x2!=inp_shape[1])]+=image[0,i,trim*(y1!=0):shape-trim*(y2!=inp_shape[0]),trim*(x1!=0):shape-trim*(x2!=inp_shape[1])]
            else:
                img1[y1:y2,x1:x2]=image[0,i,:,:]
            i+=1
    return torch.clip(img1,0,1)

def iou(pred, target, n_classes=4):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for undefined class ("9")
    for cls in range(n_classes - 1):  # last class is ignored
        pred_inds = pred == cls
        target_inds = target == cls

        intersection = torch.sum(pred_inds & target_inds)
        union = torch.sum(pred_inds + target_inds)
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append((intersection / union).cpu().numpy())

    return np.array(ious)


def pixel_acc(pred, target, n_classes=4):
    acc = []
    #pred = pred.view(-1)
    #target = target.view(-1)
    correct_pixels = 0
    true_pixels = 0

    # Ignore IoU for undefined class ("9")
    for cls in range(1,n_classes):  # last class is ignored
        pred_inds = pred == cls
        target_inds = target == cls
        correct_pixels = correct_pixels + torch.sum(
            pred_inds & target_inds)  # compute the number of pixels correctly labeled
        true_pixels = true_pixels + torch.sum(target_inds)  # compute the number of pixels in ground truth

    return np.nanmean((correct_pixels / true_pixels).cpu().numpy())  # return the average accuracy

def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
      return 0,0

    width_is_longer = w >= h
    side_long, side_short = (w,h) if width_is_longer else (h,w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
      # half constrained case: two crop corners touch the longer side,
      #   the other two corners are on the mid-line parallel to the longer line
      x = 0.5*side_short
      wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
    else:
      # fully constrained case: crop touches all 4 sides
      cos_2a = cos_a*cos_a - sin_a*sin_a
      wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a
    wi = (w*cos_a + h*sin_a - wr)/2
    hi = (w*sin_a + h*cos_a - hr)/2
    return wr,hr,wi,hi