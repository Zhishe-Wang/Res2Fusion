import os
from os import listdir
from os.path import join
import random
import numpy as np
import torch
from PIL import Image
from scipy.misc import imread, imsave, imresize
from torchvision import transforms

def make_floor(path1,path2):
    path = os.path.join(path1,path2)
    if os.path.exists(path) is False:  # ./models/
        os.makedirs(path)
    return path

def save_feat1(index,feat,result_path,feat_name):

    C = feat.size()[1]
    feat_path1 = make_floor(os.getcwd(), result_path)
    feat_path = make_floor(feat_path1, feat_name)
    index_feat_path = make_floor(feat_path, str(index))

    for c in range(C):
        temp = feat[:, c, :, :].squeeze()
        temp = temp.cpu().clamp(0, 255).data.numpy()
        feat_filenames = feat_name + '_C' + str(c) + '.png'
        path = index_feat_path + '/' + feat_filenames
        imsave(path, temp)



def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])
    return images

# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = imread(path, mode=mode)
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')

    if height is not None and width is not None:
        image = imresize(image, [height, width], interp='nearest')
    return image


def get_train_images_auto(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


def get_test_images(paths, height=None, width=None, mode='RGB'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


def save_images(path, data):
    if data.shape[0] == 1:
        data = data.reshape([data.shape[1], data.shape[2]])
        print(data.shape)
    imsave(path, data)


