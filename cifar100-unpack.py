import os
import pickle
from PIL import Image
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def unpack_imgs_to_dir(basedir, file, labels):
    base_path = os.path.join(basedir, 'unpacked', file)
    make_dir(base_path)
    for l in labels:
        make_dir(os.path.join(base_path, l.decode('UTF-8')))

    dict = unpickle(os.path.join(basedir, file))
    print(dict.keys())
    names = dict[b'filenames']
    data = dict[b'data']
    targets = dict[b'fine_labels']

    for n, d, t in zip(names, data, targets):
        im = Image.fromarray(np.moveaxis(d.reshape(3, 32, 32), 0, -1))
        im.save(os.path.join(base_path, labels[t].decode('UTF-8'), n.decode('UTF-8')), 'PNG')

root_dir = os.path.join('data', 'cifar100')
labels = unpickle(os.path.join(root_dir, 'meta'))[b'fine_label_names']
print(labels)
unpack_imgs_to_dir(root_dir, 'test', labels)
unpack_imgs_to_dir(root_dir, 'train', labels)