import cv2
import numpy as np

def getCurvature(contour,stride=8):
	curvature=[]
	assert stride<len(contour),"stride must be shorther than length of contour"
	
	for i in range(len(contour)):
		before=i-stride+len(contour) if i-stride<0 else i-stride
		after=i+stride-len(contour) if i+stride>=len(contour) else i+stride
		#print(contour[after][0])
		f1x,f1y=(contour[after][0]-contour[before][0])/stride
		f2x,f2y=(contour[after][0]-2*contour[i][0]+contour[before][0])/stride**2
		denominator=(f1x**2+f1y**2)**3+1e-11
		
		curvature_at_i=np.sqrt(4*(f2y*f1x-f2x*f1y)**2/denominator) if denominator > 1e-12 else -1

		curvature.append(curvature_at_i)
    
	return np.mean(np.array(curvature))

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


def compute_mask_overlap(mask1,mask2):
	# return the index where each mask exists
	#print(np.where(np.logical_and(mask1==1,mask2==1)))
	return np.logical_and(mask1==1,mask2==255).astype(np.uint8)