import os
import numpy as np
import cv2



origin_dir = '../data/fusai/fusai_test/A'

read_dir = "output/predict"
save_dir = "output/predict_merge"


size = 256
stride = 128
names = os.listdir(origin_dir)
print(len(names))

for name in names:
    img_path = os.path.join(origin_dir, name)
    img = cv2.imread(img_path)
    h,w = img.shape[:2]

    mask = np.zeros((h,w))
    basename = name[:-4]

    w_count = w//stride
    h_count = h//stride
    if w%size > size/10:
        w_count+=1
    if h%size > size/10:
        h_count+=1
    
    for y in range(h_count):
        for x in range(w_count):
            pre_path = os.path.join(read_dir,basename+"_x%d_y%d.png" % (x,y))
            img_part = cv2.imread(pre_path, cv2.IMREAD_GRAYSCALE)

            x0 = x*stride
            y0 = y*stride
            x1 = x0+size
            y1 = y0+size

            if x1>w or y1>h:
                crop_w = size
                crop_h = size
                if x1>w:
                    crop_w = w-x0
                if y1>h:
                    crop_h = h-y0
                #print(crop_w, crop_h)
                #mask[y0:y0+crop_h, x0:x0+crop_w] = img_part[:crop_h, :crop_w]
                mask[y0:y0+crop_h, x0:x0+crop_w] = np.max(np.concatenate([mask[y0:y0+crop_h, x0:x0+crop_w].reshape((crop_h,crop_w,1)),img_part[:crop_h, :crop_w].reshape((crop_h,crop_w,1))],axis=-1),axis=-1)
            else:
                #mask[y0:y1, x0:x1] = img_part
                mask[y0:y1, x0:x1] = np.max(np.concatenate([mask[y0:y1, x0:x1].reshape((size,size,1)),img_part.reshape((size,size,1))],axis=-1),axis=-1)

    cv2.imwrite(os.path.join(save_dir,basename+".png"),mask)

     