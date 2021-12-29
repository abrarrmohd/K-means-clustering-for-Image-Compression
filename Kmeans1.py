#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
from PIL import Image


# In[2]:


img = Image.open("/Users/mohamedabrar/Downloads/hw3_part2_data/Koala.jpg")


# In[3]:


img=np.asarray(img)


# In[4]:


original_size=os.path.getsize("/Users/mohamedabrar/Downloads/hw3_part2_data/Koala.jpg")


# In[5]:


img=np.reshape(img,(768*1024,3))


# In[6]:


comp_ratio=[]
temp_i=np.zeros((768*1024,3))


# In[7]:


for p in range(10):
    classes = [-1] * img.shape[0]
    classes=np.array(classes)
    k=2
    rows=np.array([])
    initial_cluster_rows=np.random.randint(0,img.shape[0],size=k)
    rows=(img[initial_cluster_rows])
    prev_rows=np.empty_like(rows)

    while (prev_rows!=rows).any():
        min_dist=[]
        prev_rows=rows
        for i in range(len(rows)):
            x1=rows[i][0]
            y1=rows[i][1]
            z1=rows[i][2]
            cluster=np.array((x1,y1,z1))
            dist=np.linalg.norm(cluster - img,axis=1)
            min_dist.append(dist)
        classes=np.argmin(min_dist,0)
        rows=[]
        for i in range(k):
            index=np.where(classes==i)
            if np.size(index)==0:
                rows.append(prev_rows[i])
            else:
                rows.append(np.mean(img[index],axis=0))
        rows=np.array(rows)
    temp_i=np.zeros((768*1024,3))
    for i in range(k):
        index=np.where(classes==i)
        temp_i[index]=rows[i]
    temp_i=temp_i.astype(np.uint8)
    img1=np.reshape(temp_i,(768,1024,3))
    image_compressed = Image.fromarray(img1)
    image_compressed.show()
    image_compressed.save('/Users/mohamedabrar/Downloads/hw3_part2_data/image.jpg')
    compressed_size=os.path.getsize('/Users/mohamedabrar/Downloads/hw3_part2_data/image.jpg')
    compression_ratio=(compressed_size/original_size)
    comp_ratio.append(compression_ratio)


# In[8]:


comp_ratio


# In[9]:


print(np.mean(comp_ratio))
print(np.var(comp_ratio))

