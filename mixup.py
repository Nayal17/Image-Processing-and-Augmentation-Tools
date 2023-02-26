import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

def mixup(images, labels, alpha=1.5):
    """ 
    images, labels: Takes batch of images and labels as input
    alpha: controls the amount of mixing that is applied to images
    """
    bs = images.shape[0]
    idxs = np.random.permutation(bs)
    lamda = np.random.beta(alpha, alpha, size=bs) # alpha=beta to avoid skewed distribution 
    
    # mixing up images 
    images = lamda.reshape(bs, 1, 1, 1) * images + (1-lamda).reshape(bs, 1, 1, 1) * images[idxs]

    # mixing up labels
    labels = lamda.reshape(bs, 1) * labels + (1-lamda).reshape(bs, 1) * labels[idxs]

    images = images.astype(np.uint8)

    return images, labels

if __name__=='__main__':

    def preprocess(name):
        shape = (512, 512)
        img = cv2.imread(f"./images/{name}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)
        return img

    img1 = preprocess(name='image')
    img2 = preprocess(name='image2')
    img3 = preprocess(name='image3')
    img4 = preprocess(name='image4')

    images = np.array([img1, img2, img3, img4])
    labels = np.array([0, 0, 1, 1]) # dummy labels

    m_images, m_labels = mixup(images, labels)

    plt.figure(figsize=(16,10))
    plt.subplot(221); plt.imshow(m_images[0]); plt.title(f"label: {m_labels[0]}")
    plt.subplot(222); plt.imshow(m_images[1]); plt.title(f"label: {m_labels[1]}")
    plt.subplot(223); plt.imshow(m_images[2]); plt.title(f"label: {m_labels[2]}")
    plt.subplot(224); plt.imshow(m_images[3]); plt.title(f"label: {m_labels[3]}")
    plt.show()

    




