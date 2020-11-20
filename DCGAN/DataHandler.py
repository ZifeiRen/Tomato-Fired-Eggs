
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio


def read_img2numpy(batch_size=64,img_h=64,img_w=64,path="data/faces"):
    """
    Read disk images and resize them.
    :param batch_size: number of images to read at a time
    :param img_h: image resizing
    :param img_w: image resize width
    :param path: Data storage path
    :return: Image numpy array
    """
    file_list = os.listdir(path) # Image Name List
    data = np.zeros([batch_size,img_h,img_w,3],dtype=np.uint8) # Initializing numpy arrays
    mask = np.random.choice(len(file_list),batch_size,replace=True)
    for i in range(batch_size):
        mm = Image.open(path+"/"+file_list[mask[i]])
        tem =mm.resize((img_w,img_h))  # Resize the image
        data[i,:,:,:] = np.array(tem)
    # Data Normalization -1-1
    data = (data-127.5)/127.5 #-1-1
    return data

def img2gif(img_path="out/dcgan/",gif_path="out/dcgan/"):
    #Get a list of image files
    file_list = os.listdir(img_path)
    imges = []
    for file in file_list:
        if file.endswith(".png"):
            img_name = img_path +file
            imges.append(imageio.imread(img_name))
    imageio.mimsave(gif_path+"result.gif",imges,fps=2)

if __name__ == '__main__':
    data = read_img2numpy()
    print(data.shape)
    data = ((data * 127.5) + 127.5).astype(np.uint8)
    plt.imshow(data[10])
    plt.show()
