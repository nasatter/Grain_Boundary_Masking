import os,cv2
import numpy as np
from general_img_proc import single_blob_detector,check_mask_overlap,skeleton

pore = 'pores//'
grain = 'grain_boundary//'

pore_masks = os.listdir(pore)
grain_masks = os.listdir(grain)


for p in pore_masks:
	for g in grain_masks:
		if p==g:
			print(p)
			pore_img = cv2.imread(pore+p)
			pore_img = cv2.cvtColor(pore_img, cv2.COLOR_BGR2GRAY)
			grain_img = cv2.imread(grain+g)
			grain_img = cv2.cvtColor(grain_img, cv2.COLOR_BGR2GRAY)
			new_mask = np.zeros([grain_img.shape[0],grain_img.shape[1],3]).astype(np.uint8);
			ret, grain_img = cv2.threshold(grain_img, 20, 255, cv2.THRESH_BINARY)
			new_mask[:,:,0] = skeleton(grain_img)
			while True:
				blob, cont = single_blob_detector(pore_img)
				if not cont:
					break
				blob = blob[1:pore_img.shape[0]+1,1:pore_img.shape[1]+1]
				overlap = check_mask_overlap(blob,grain_img)
				#print(overlap)
				if overlap:
					new_mask[:,:,1][blob>0]=255
				else:
					new_mask[:,:,2][blob>0]=255
				cv2.imshow('out',pore_img)
				cv2.waitKey(1)
			print('press key')
			cv2.imwrite('combined//'+p,new_mask)
			cv2.waitKey(0)
					

        