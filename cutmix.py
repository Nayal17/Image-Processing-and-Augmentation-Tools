import cv2
import numpy as np
import matplotlib.pyplot as plt

def rand_bbox(size, lamb):
    """ Generate random bounding box 
    Args:
        - size: [width, breadth] of image
        - lamb: (lambda) cut ratio parameter, sampled from Beta distribution
    Returns:
        - Bounding box
    """
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lamb)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def generate_cutmix_image(image_batch, image_batch_labels, beta=0.2):
    """ Generate a CutMix augmented image from a batch 
    Args:
        - image_batch: a batch of input images
        - image_batch_labels: labels corresponding to the image batch
        - beta: a parameter of Beta distribution.
    Returns:
        - CutMix image batch, updated labels
    """
    # generate mixed sample
    lam = np.random.beta(beta, beta)
    rand_index = np.random.permutation(len(image_batch))
    target_a = image_batch_labels
    target_b = image_batch_labels[rand_index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(image_batch[0].shape, lam)
    image_batch_updated = image_batch.copy()
    image_batch_updated[:, bbx1:bbx2, bby1:bby2, :] = image_batch[rand_index, bbx1:bbx2, bby1:bby2, :]
    
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_batch.shape[1] * image_batch.shape[2]))
    label = target_a * lam + target_b * (1. - lam)
    
    return image_batch_updated, label


if __name__=="__main__":
    img_batch = []
    for i in range(3):
        img = cv2.imread(f"/home/nayal/Documents/github work/ImageProcessingTools/images/image{i+2}.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,512))
        img_batch.append(list(img))
    

    img_batch = np.array(img_batch)
    image_batch_labels = np.array([[0],[1],[2]])
    aug_batch = generate_cutmix_image(img_batch, image_batch_labels)
    
    plt.figure(figsize=(16,10))
    plt.subplot(131); plt.imshow(aug_batch[0][0]); plt.title(f"Label: {aug_batch[1][0]}")
    plt.subplot(132); plt.imshow(aug_batch[0][1]); plt.title(f"Label: {aug_batch[1][1]}")
    plt.subplot(133); plt.imshow(aug_batch[0][2]); plt.title(f"Label: {aug_batch[1][2]}")
    plt.show()
    