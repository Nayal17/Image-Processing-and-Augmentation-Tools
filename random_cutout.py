import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

def random_cutout(img, cutout_size=(16,16), cutout_counts=10, mask=0):
    if img.shape[0]<cutout_size[0] or img.shape[1]<cutout_size[1]:
        raise Exception("Image should be larger than cutout.")
    
    for _ in range(cutout_counts):
        x_limit = img.shape[0] - cutout_size[0]
        y_limit = img.shape[1] - cutout_size[1] 
        random_x = random.randint(0, x_limit)
        random_y = random.randint(0, y_limit)

        img[random_x:random_x+cutout_size[0], random_y:random_y+cutout_size[1]] = mask
    
    return img

if __name__=='__main__':
    img = cv2.imread("./images/image.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cutout_img = random_cutout(img, cutout_counts=20)
    plt.imshow(img)
    plt.show()
