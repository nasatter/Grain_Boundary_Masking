import os,cv2
from general_img_proc import single_blob_detector,check_mask_overlap,skeleton

# 1. We need to use skeleton the grain boundary
# 2. We need to compute the grains for each image
#	2.1 the area is minus the internal pores
#	2.2 other features ignore internal pores
#	2.3 no grain boundary pores are included

mask = 'combined//'

masks = os.listdir(mask)
for m in masks:
	img = cv2.imread(mask+m)
	gb = ~skeleton(img[:,:,0])
	gp = img[:,:,1]
	bp = img[:,:,2]
	while True:
		blob, cont = single_blob_detector(gb)
		if not cont:
			break
		blob = blob[1:gb.shape[0]+1,1:gb.shape[1]+1]
		overlap = check_mask_overlap(blob,grain_img)
		print(overlap)
		if overlap:
			new_mask[:,:,1][blob>0]=255
		else:
			new_mask[:,:,2][blob>0]=255
		cv2.imshow('out',gb)
		cv2.waitKey(1)