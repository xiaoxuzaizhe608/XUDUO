import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
def read_image(paths):
    os.listdir(paths)
    filelist = []
    for root, dirs, files in os.walk(paths):
        for file in files:
            if os.path.splitext(file)[1] == ".bmp":
                filelist.append(os.path.join(root, file))
    return filelist
def im_array(paths):
	M=[]
	for filename in paths:
		im=Image.open(filename)
		im_L=im.convert("L")
		Core=im_L.getdata()
		arr1=np.array(Core,dtype='float32')/255.0
		list_img=arr1.tolist()
		M.extend(list_img)
	return M
path_1 = 'figs/weiwang/'
path_2 = 'figs/tuowang/'
path_3 = 'figs/ciwang/'
filelist_1 = read_image(path_1)
filelist_2 = read_image(path_2)
filelist_3 = read_image(path_3)
filelist_all = filelist_1+filelist_2+filelist_3

M = []
M = im_array(filelist_all)

dict_label={0:'围网',1:'拖网',2:'刺网'}
train_images=np.array(M).reshape(len(filelist_all),224,224)
label=[0]*len(filelist_1)+[1]*len(filelist_2)+[2]*len(filelist_3)
train_lables=np.array(label)        #数据标签
train_images = train_images[ ..., np.newaxis ]        #数据图片
print(train_images.shape)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 1)))#过滤器个数，卷积核尺寸，激活函数，输入形状
# model.add(layers.AveragePooling2D((3, 3)))#池化层
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.Flatten())#降维
model.add(layers.Dense(64, activation='relu'))#全连接层
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(3, activation='softmax'))#注意这里参数，几类图片就写几
model.summary()  # 显示模型的架构
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#epochs为训练多少轮、batch_size为每次训练多少个样本
model.fit(train_images, train_lables, epochs=10)
model.save('my_model.h5') #保存为h5模型
#tf.keras.models.save_model(model,"F:\python\moxing\model")#这样是pb模型
print("模型保存成功！")