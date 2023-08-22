import cv2
import numpy as np
from skimage.morphology import skeletonize

def single_blob_detector(mask):
	# find eligible points for floodfill
	coors = np.where(mask == 255)
	if len(coors[0])==0:
		return 0,False
	# select any point
	indx = np.random.randint(0,len(coors[0]))
	coor = [coors[1][indx],coors[0][indx]]
	#print(coor,mask.shape)
	# create mask for floodfill
	temp_mask = np.zeros(shape=(mask.shape[0]+2,mask.shape[1]+2), dtype=np.uint8)
	cv2.floodFill(mask, temp_mask, coor, 1)
	return temp_mask,True


def check_mask_overlap(mask1,mask2):
	# return the index where each mask exists
	grain_indx=np.transpose(np.array(np.where(mask2>0)))
	pore_indx=np.transpose(np.array(np.where(mask1>0)))
	# only compare the bounds of the pore index to increase speed substancially
	#print(pore_indx[:,1])
	x1 = np.min(pore_indx[:,1])
	x2 = np.max(pore_indx[:,1])
	y1 = np.min(pore_indx[:,0])
	y2 = np.max(pore_indx[:,0])
	mask2 = mask2[y1:y2,x1:x2]
	mask1 = mask1[y1:y2,x1:x2]
	grain_indx=np.transpose(np.array(np.where(mask2>0)))
	pore_indx=np.transpose(np.array(np.where(mask1>0)))
	# compare all values of one to one index of the other
	for i in range(pore_indx.shape[0]):
		compare_any = (pore_indx[i,:]==grain_indx)
		#print(com)
		compare_both = np.logical_and(compare_any[:,0], compare_any[:,1]).any()
		#print(compare_both)
		if compare_both:
			return True
	return False

def skeleton(mask):
	return cv2.ximgproc.thinning(mask)


def check_edge(edge_mask):
	#print(edge_mask.shape)
	#edge_mask = cv2.cvtColor(edge_mask, cv2.COLOR_BGR2GRAY)
	edge_mask=cv2.ximgproc.thinning(edge_mask)
	sh = edge_mask.shape
	#print(sh)
	mask_overlap = np.zeros(edge_mask.shape)
	mask_shift_left = np.zeros(edge_mask.shape)
	mask_shift_left[0:-1,:] = edge_mask[1:,:]
	mask_shift_top_left = np.zeros(edge_mask.shape)
	mask_shift_top_left[0:-1,0:-1] = edge_mask[1:,1:]
	mask_shift_top = np.zeros(edge_mask.shape)
	mask_shift_top[0:,0:-1] = edge_mask[:,1:]
	mask_shift_top_right = np.zeros(edge_mask.shape)
	mask_shift_top_right[1:,0:-1] = edge_mask[:-1,1:]
	mask_shift_right = np.zeros(edge_mask.shape)
	mask_shift_right[1:,:] = edge_mask[:-1,:]
	mask_shift_bottom_right = np.zeros(edge_mask.shape)
	mask_shift_bottom_right[1:,1:] = edge_mask[:-1,:-1]
	mask_shift_bottom = np.zeros(edge_mask.shape)
	mask_shift_bottom[:,1:] = edge_mask[:,:-1]
	mask_shift_bottom_left = np.zeros(edge_mask.shape)
	mask_shift_bottom_left[0:-1,1:] = edge_mask[1:,:-1]
	mask_overlap = mask_shift_left+mask_shift_top_left+mask_shift_top+mask_shift_top_right+mask_shift_right+mask_shift_bottom_right+mask_shift_bottom+mask_shift_bottom_left
	mask_overlap[(mask_overlap==255)*(edge_mask==255)] = 1
	mask_overlap[mask_overlap>1] = 0
	mask_overlap[mask_overlap==1] = 1
	mask_overlap[0,:] = 0
	mask_overlap[:,0] = 0
	mask_overlap[sh[0]-1,:]=0
	mask_overlap[:,sh[1]-1]=0
	return mask_overlap


def eolDetect(mask):
	new_mask = np.zeros([mask.shape[0],mask.shape[1],1]).astype(np.uint8);
	kernel = np.ones((10, 10), np.uint8)
	eol_mask = check_edge(mask)
	img_dilation = cv2.dilate(eol_mask, kernel, iterations=1)
	coors = np.where(eol_mask==1)
	#print(coors,coors[0])
	for i in range(len(coors[0])):
		y = coors[0][i]
		x = coors[1][i]
		coor = [x,y]
		new_mask[y,x,:] = 255
	return new_mask,coors

def returnNeighbors(mask, coors):
	xs = [-1,0,1]
	ys = [-1,0,1]
	neighbors = []
	update_coors = []
	new_coors = [coors]
	i = 1
	j = 0
	#print('new')
	while len(new_coors)<4 and (len(neighbors)<11) and (len(new_coors)>0):
		for coors in new_coors:

			for x in xs:
				for y in ys:
					if not (((x==0) and (y==0)) or (coors[0]+y==mask.shape[0]-1) or (coors[1]+x==mask.shape[1]-1)):

						val = mask[coors[1]+x,coors[0]+y]

						#print(val)
						if val>0:
							
							#print(coors,i,j,neighbors)
							#print([coors[0]+y,coors[1]+x])
							update_coors.append([coors[0]+y,coors[1]+x])
							if len(update_coors)==2:
								if ((update_coors[0][0]-update_coors[1][0])**2+(update_coors[0][1]-update_coors[1][1])**2)**0.5 >= 2**0.5:
									j=1000
								else:
									mask[coors[1]+x,coors[0]+y] = 0
									neighbors.append([coors[0]+y,coors[1]+x])
							else:
									mask[coors[1]+x,coors[0]+y] = 0
									neighbors.append([coors[0]+y,coors[1]+x])

							j+=1
		new_coors = update_coors
		update_coors = []

	#print(neighbors)
	return neighbors,mask

def march(mask,coors):
	coors = np.array(coors)
	copy_mask = np.zeros(mask.shape)
	X = np.zeros((coors.shape[0],2))
	Y = np.zeros(coors.shape[0])
	#X[:,0] = 0#coors[:,0]**0.5
	X[:,0] = coors[:,0]
	X[:,1] = 1
	Y= coors[:,1]
	#print(X)
	#print(Y)
	a = np.linalg.lstsq(X,Y, rcond=None)[0]

	#print(a)
	direction = X[-1,0] - X[0,0] + X[-2,0] - X[1,0]
	#print(direction)
	if direction<0 and a[0]> 0:
		newX = np.linspace(int(X[-1,0]),mask.shape[0]-1,mask.shape[0]*20)
		newY =  newX * a[0]  + a[1]
		new_coors = np.array([np.round(newX),np.round(newY)]).T
		new_coors = np.unique(new_coors,axis=0)
		new_coors = new_coors[new_coors[:, 1].argsort()]
	elif direction<0 and a[0]< 0:
		newX = np.linspace(int(X[-1,0]),mask.shape[0]-1,mask.shape[0]*20)
		newY =  newX * a[0]  + a[1]
		new_coors = np.array([np.round(newX),np.round(newY)]).T
		new_coors = np.unique(new_coors,axis=0)
		new_coors = new_coors[new_coors[:, 1].argsort()[::-1]]
	elif direction>0 and a[0]< 0: # Good
		newX = np.linspace(int(X[-1,0]),0,mask.shape[0]*20)
		newY =  newX * a[0]  + a[1]
		new_coors = np.array([np.round(newX),np.round(newY)]).T
		new_coors = np.unique(new_coors,axis=0)
		new_coors = new_coors[new_coors[:, 1].argsort()]
	else: # Good
		newX = np.linspace(int(X[-1,0]),0,mask.shape[0]*20)
		newY =  newX * a[0]  + a[1]
		new_coors = np.array([np.round(newX),np.round(newY)]).T
		new_coors = np.unique(new_coors,axis=0)
		new_coors = new_coors[new_coors[:, 1].argsort()[::-1]]
	new_coors = new_coors[np.where(new_coors[:,1]>0)[0],:]
	new_coors = new_coors[np.where(new_coors[:,1]<mask.shape[0]-1)[0],:]
	final = [int(new_coors[-1,0]),int(new_coors[-1,1])]
	#print(new_coors)
	#print(coors)
	for i in range(new_coors.shape[0]-1):
		i+=1
		x = int(new_coors[i,0])
		y = int(new_coors[i,1])
		#print(y,x)
		#if (x<0) or (y<0) or (x>=mask.shape[1]) or (y>=mask.shape[0]):
		#	break
		#print(mask[y,x])
		if mask[y,x]==255:
			if i>20:
				final=[x,y]
				break
		#mask[y,x] = 255
	start_stop = [int(new_coors[0,0]),int(new_coors[0,1])],final
	mask = cv2.line(mask,start_stop[0],start_stop[1],255,1)

	return mask,start_stop


def min_inersect(lines,msk):
	msk1 = np.zeros(msk.shape).astype(np.uint8)
	msk2 = np.zeros(msk.shape).astype(np.uint8)
	# [x,y]
	for i in range(len(lines)-1):
		msk1*=0
		p1 = lines[i][0]
		p2 = lines[i][1]
		mindist1 = ((p1[1]-p2[1])**2 + (p1[0]-p2[0])**2)**0.5
		msk1 = cv2.line(msk1,p1,p2,255,1)
		for j in range(i+1,len(lines)):
			msk2*=0
			p1i = lines[j][0]
			p2i = lines[j][1]
			mindist2 = ((p2i[0]-p1i[0])**2 + (p2i[1]-p1i[1])**2)**0.5
			msk2 = cv2.line(msk2,p1i,p2i,255,1)
			innersect = np.where(msk1*msk2>0)
			if len(innersect[0])>0:
				index = np.array([innersect[1][0],innersect[0][0]])
				newdist1 = ((p1[1]-index[1])**2 + (p1[0]-index[0])**2)**0.5
				if newdist1<mindist1:
					mindist1 = newdist1
					lines[i][1] = index
				#print(p1,index,newdist1)


	for line in lines:
		msk = cv2.line(msk,line[0],line[1],255,2)
	#cv2.imshow('seeit1',msk)
	return msk