import cv2
import matplotlib.pyplot as plt
from convolve import *

def blur(image, intensity=5, gray=False):
    kernel = np.ones([intensity, intensity])/float((intensity**2))
    if image.shape[2]!=1:
        # color image
        image = color_convolve(img, kernel=kernel, pad=1)

    else:
        # grayscale image
        image = convolve2d(img, kernel, pad=1)

    return image


if __name__=='__main__':

    img = cv2.imread("./images/image4.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    b_img = blur(img)
    plt.figure(figsize=(16,10))
    plt.subplot(121); plt.imshow(img); plt.title("RGB Image")
    plt.subplot(122); plt.imshow(b_img); plt.title("Blur Image")
    plt.show()
