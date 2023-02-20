import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 

def convolve(img, kernel, pad=0, stride=1):
    channels = 1
    try:
        if img.shape[2]==3: # three channels
            c1 = img[:,:,0]
            c2 = img[:,:,1]
            c3 = img[:,:,2]
            channels=3
    except:
        pass

    final_img = []

    for c in range(channels):  
        if channels>1:  
            if c==0:
                img = c1
            if c==1:
                img = c2
            else:
                img==c2
            
        img_height = img.shape[0]
        img_width = img.shape[1] 
        kernel_height = kernel.shape[0]
        kernel_width = kernel.shape[1]

        out_img_height = int((img_height + 2*pad - kernel_height)/stride +1)
        out_img_width = int((img_width + 2*pad - kernel_width)/stride +1)
        out_img = np.zeros((out_img_height, out_img_width))

        if pad==0:
            padded_img = pad
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

        final_img.append(out_img)
    
    final_img = np.array(final_img).transpose(1,2,0)
    
    return final_img


if __name__=='__main__':

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])

    img = cv2.imread("/home/nayal/Documents/ImageProcessingTools/images/image.jpg")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    conv_img_g = convolve(gray, kernel, pad=1)
    conv_img_rgb = convolve(rgb, kernel, pad=1)

    plt.figure(figsize=(16,10))
    plt.subplot(131);plt.imshow(rgb, cmap='gray') ; plt.title("RGB Image") 
    plt.subplot(132);plt.imshow(conv_img_g, cmap="gray") ; plt.title("Gray Convolved Image") 
    plt.subplot(133);plt.imshow(conv_img_rgb) ; plt.title("RGB Convolved Image") 

    plt.show()
