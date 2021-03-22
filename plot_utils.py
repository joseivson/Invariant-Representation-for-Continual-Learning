import numpy as np
import matplotlib.pyplot as plt
import cv2
import seaborn as sns

# this function is borrowed from https://github.com/hwalsuklee/tensorflow-mnist-AAE/blob/master/plot_utils.py
class plot_samples():
    def __init__(self, DIR, n_img_x=8, n_img_y=8, img_w=28, img_h=28, channels=3):
        self.DIR = DIR
        assert n_img_x > 0 and n_img_y > 0
        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_total_imgs = n_img_x * n_img_y
        assert img_w > 0 and img_h > 0
        self.img_w = img_w
        self.img_h = img_h
        self.channels = channels

    def save_images(self, images, name='result.jpg'):
        if self.channels > 1:
            images = images.reshape(self.n_img_x*self.n_img_y, self.img_h, self.img_w, self.channels)
            plt.imsave(self.DIR + "/"+name, self._merge(images, [self.n_img_y, self.n_img_x, self.channels]))
        else:
            images = images.reshape(self.n_img_x*self.n_img_y, self.img_h, self.img_w)
            plt.imsave(self.DIR + "/"+name, self._merge(images, [self.n_img_y, self.n_img_x]))

    def _merge(self, images, size):
        h, w = images.shape[1], images.shape[2]

        if self.channels > 1:
            img = np.zeros((h * size[0], w * size[1], self.channels))
        else:
            img = np.zeros((h * size[0], w * size[1]))


        for idx, image in enumerate(images):
            i = int(idx % size[1])
            j = int(idx / size[1])
            try:
                image_ = cv2.resize(image.detach().numpy(), dsize=(w,h), interpolation=cv2.INTER_CUBIC)
            except:
                image_ = cv2.resize(image, dsize=(w,h), interpolation=cv2.INTER_CUBIC)
            if self.channels > 1:
                img[j*h:j*h+h, i*w:i*w+w, :] = image_
            else:
                img[j*h:j*h+h, i*w:i*w+w] = image_
        return img

    def scatter(self, x, y, hue, name='scatter.jpg'):
        plt.figure()
        sns.scatterplot(x=x, y=y, hue=hue)
        plt.savefig(self.DIR + name)
        plt.close()