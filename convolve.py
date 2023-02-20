import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

def convolve2d(img, kernel, pad=0, stride=1):
        """takes image with only one channel"""
        img_height = img.shape[0]
        img_width = img.shape[1] 
        kernel_height = kernel.shape[0]
        kernel_width = kernel.shape[1]

        out_img_height = int((img_height + 2*pad - kernel_height)/stride +1)
        out_img_width = int((img_width + 2*pad - kernel_width)/stride +1)
        out_img = np.zeros((out_img_height, out_img_width))

        if pad==0:
            padded_img = img
        else:
            padded_img = np.zeros((img_height+pad*2, img_width+pad*2))
            padded_img[pad:pad+img_height, pad:pad+img_width] = img

        for y in tqdm(range(img_width)):
            if y > (img_width-kernel_width):
                break

            if y%stride==0:
                for x in range(img_height):
                    if x > (img_height-kernel_height):
                        break

                    if x%stride==0:
                        out_img[x,y] = (padded_img[x:x+kernel_height, y:y+kernel_width] * kernel).sum()

        return out_img

def color_convolve(img, kernel, pad=0, stride=1):
    """takes BGR image as input"""
    b, g, r = cv2.split(img)
    
    b_conv = convolve2d(b, kernel) 
    g_conv = convolve2d(g, kernel) 
    r_conv = convolve2d(r, kernel) 

    conv_bgr = cv2.merge([b_conv, g_conv, r_conv])

    return conv_bgr


if __name__=='__main__':

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

    img = cv2.imread("/home/nayal/Documents/ImageProcessingTools/images/image.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_conv_img = convolve2d(gray, kernel, pad=1)
    color_conv_img = color_convolve(img, kernel, pad=1)
    
    plt.figure(figsize=(16,10))
    plt.subplot(131); plt.imshow(img); plt.title("BGR Image")
    plt.subplot(132); plt.imshow(gray_conv_img, cmap="gray"); plt.title("Convolve2d Image")
    plt.subplot(133); plt.imshow(color_conv_img); plt.title("Color Convolve Image")
    plt.show()


    