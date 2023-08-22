import os,cv2
import numpy as np
from interpolate.img_proc import *

#grain = 'C:\\projects\\combine_msks\\val\\'
grain = 'C:\\projects\\combine_msks\\eval\\found\\'

def fix_it_up(grain_img,length):
	
	blob_mask = np.zeros([grain_img.shape[0],grain_img.shape[1],1]).astype(np.uint8);

	new_mask = skeleton(grain_img)
	old_mask = new_mask.copy()
	
	eol,coors = eolDetect(new_mask)
	# now we need to trace back the eol until the number of neighbors is more than 1
	# depending on the length, the line will be trimmed or interpolated
	new_lines = []
	for i in range(len(coors[0])):
		y = coors[0][i]
		x = coors[1][i]
		coor = [x,y]
		neighbors,_ = returnNeighbors(new_mask.copy(),coor)
		if not neighbors==[]:
			if len(neighbors)>length:
				old_mask,line = march(old_mask,neighbors)
				new_lines.append(line)
	final_mask = min_inersect(np.array(new_lines),new_mask.copy())
	#cv2.imshow('blob',old_mask)
	##cv2.imshow('in',grain_img)
	#cv2.imshow('out',new_mask)
	##
	#cv2.imshow('eol',eol)
	
	#print('press key')
	#cv2.imwrite('combined//'+p,new_mask)
	#cv2.waitKey(0)
	return final_mask

def close_gaps(grain_img):
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
	#grain_img = cv2.erode(grain_img, kernel, iterations=1)
	
	grain_img = cv2.dilate(grain_img, kernel, iterations=2)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
	grain_img = cv2.GaussianBlur(grain_img,(5,5),0.2)
	grain_img[grain_img>200]=255
	#cv2.imshow('orig',grain_img)
	
	grain_img = fix_it_up(grain_img,5)
	grain_img = cv2.dilate(grain_img, kernel, iterations=2)
	grain_img = cv2.erode(grain_img, kernel, iterations=2)
	grain_img = fix_it_up(grain_img,8)
	#grain_img = cv2.dilate(grain_img, kernel, iterations=2)
	#grain_img = cv2.erode(grain_img, kernel, iterations=2)
	#grain_img = fix_it_up(grain_img,8)
	grain_img = skeleton(grain_img)
	#cv2.imshow('final',grain_img)
	#cv2.waitKey(0)
	return grain_img

def run_dir(grains):
	
	grain_masks = os.listdir(grain)
	for g in grain_masks:
		if 'out' in g:
			print(g)
			grain_img = cv2.imread(grain+g)
			grain_img = grain_img[:,:,0].astype(np.uint8)
			grain_img[grain_img>200]=255
			close_gaps(grain_img)
			
			
					
if __name__ == '__main__': 
	run_dir(grain)