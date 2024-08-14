import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
from typing import Callable, Optional
import os


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

NP_EXTENSIONS = ['.npy',]

BAPPS_SPLITS = [
    'cnn',  'color',  'deblur',  'frameinterp',  'superres',  'traditional']

DEFAULT_DIRS_OLD = {
    'nights': '../robust-clip/nights',
    'bapps': '../PerceptualSimilarity/dataset/2afc/val/',
    'bapps-bin': '../PerceptualSimilarity/dataset/2afc/val/',
    'things': '../things_dataset_new/THINGS/Images/',
    # Other files for datasets.
    'things-triplets': '../things_dataset/triplet_dataset/triplets_large_final_correctednc_correctedorder.csv',
    'things-imgpaths': './things_first_imgs_idx.txt',
}

DEFAULT_DIRS = {
    'nights': 'nights',
    'bapps': '../PerceptualSimilarity/dataset/2afc/val/',
    'bapps-bin': '../PerceptualSimilarity/dataset/2afc/val/',
    'things': '../things_dataset_new/THINGS/Images/',
    # Other files for datasets.
    'things-triplets': '../things_dataset/triplet_dataset/triplets_large_final_correctednc_correctedorder.csv',
    'things-imgpaths': './things_first_imgs_idx.txt',
}


# Adapted from https://github.com/ssundaram21/dreamsim/blob/6f4a5182b37a2e255bad9fc471c50be3d8613037/dataset/dataset.py.
class NIGHTSTwoAFCDataset(Dataset):
    def __init__(self, root_dir: str, split: str = "train",
                 #load_size: int = 224,
                 #interpolation: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
                 preprocess: Callable = nn.Identity(), #str = "DEFAULT",
                 subdir: Optional[str] = None,
                 **kwargs):
        self.root_dir = root_dir
        self.csv = pd.read_csv(os.path.join(self.root_dir, "data.csv"))
        self.csv = self.csv[self.csv['votes'] >= 6] # Filter out triplets with less than 6 unanimous votes
        self.split = split
        #self.load_size = load_size
        #self.interpolation = interpolation
        self.preprocess_fn = preprocess #get_preprocess_fn(preprocess, self.load_size, self.interpolation)
        self.raw_images = False

        if self.split == "train" or self.split == "val":
            self.csv = self.csv[self.csv["split"] == split]
        elif split == 'test_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == True]
        elif split == 'test_no_imagenet':
            self.csv = self.csv[self.csv['split'] == 'test']
            self.csv = self.csv[self.csv['is_imagenet'] == False]
        else:
            raise ValueError(f'Invalid split: {split}')

        # To use only a chunk of the datasets.
        if subdir is not None:
            self.csv = self.csv[[subdir in k for k in self.csv['ref_path']]]

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        id = self.csv.iloc[idx, 0]
        p = self.csv.iloc[idx, 2].astype(np.float32)
        if not self.raw_images:
            img_ref = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 4])))
            img_left = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 5])))
            img_right = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 6])))
            return img_ref, img_left, img_right, p, id
        else:
            img_ref = Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 4]))
            img_left = Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 5]))
            img_right = Image.open(os.path.join(self.root_dir, self.csv.iloc[idx, 6]))
            return img_ref, img_left, img_right, p, id


# Adapted from https://github.com/richzhang/PerceptualSimilarity/blob/master/data/dataset/twoafc_dataset.py.
class BAPPSTwoAFCDataset(Dataset):
    def __init__(self, dataroots, load_size=64, binarize_labels=False):
        if(not isinstance(dataroots,list)):
            dataroots = [dataroots,]
        self.roots = dataroots
        self.load_size = load_size
        self.binarize_labels = binarize_labels

        # image directory
        self.dir_ref = [os.path.join(root, 'ref') for root in self.roots]
        self.ref_paths = make_dataset(self.dir_ref)
        self.ref_paths = sorted(self.ref_paths)

        self.dir_p0 = [os.path.join(root, 'p0') for root in self.roots]
        self.p0_paths = make_dataset(self.dir_p0)
        self.p0_paths = sorted(self.p0_paths)

        self.dir_p1 = [os.path.join(root, 'p1') for root in self.roots]
        self.p1_paths = make_dataset(self.dir_p1)
        self.p1_paths = sorted(self.p1_paths)

        transform_list = []
        transform_list.append(transforms.Resize(load_size))
        transform_list += [transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            ]

        self.transform = transforms.Compose(transform_list)

        # judgement directory
        self.dir_J = [os.path.join(root, 'judge') for root in self.roots]
        self.judge_paths = make_dataset(self.dir_J, mode='np')
        self.judge_paths = sorted(self.judge_paths)

    def __getitem__(self, index):
        p0_path = self.p0_paths[index]
        p0_img_ = Image.open(p0_path).convert('RGB')
        p0_img = self.transform(p0_img_)

        p1_path = self.p1_paths[index]
        p1_img_ = Image.open(p1_path).convert('RGB')
        p1_img = self.transform(p1_img_)

        ref_path = self.ref_paths[index]
        ref_img_ = Image.open(ref_path).convert('RGB')
        ref_img = self.transform(ref_img_)

        judge_path = self.judge_paths[index]
        # judge_img = (np.load(judge_path)*2.-1.).reshape((1,1,1,)) # [-1,1]
        judge_img = np.load(judge_path).reshape((1,1,1,)) # [0,1]
        lab = np.load(judge_path).astype(np.float32).item()
        if self.binarize_labels:  # Get binary classification.
            lab = float(lab > .5)

        judge_img = torch.FloatTensor(judge_img)

        #return {'p0': p0_img, 'p1': p1_img, 'ref': ref_img, 'judge': judge_img,
        #    'p0_path': p0_path, 'p1_path': p1_path, 'ref_path': ref_path, 'judge_path': judge_path}

        # To preserve the format of NIGHTS dataset.
        return ref_img, p0_img, p1_img, lab, index

    def __len__(self):
        return len(self.p0_paths)


class THINGSDataset(Dataset):
    def __init__(self, data_dir, img_file, triplets_file, preprocess_fn, n_triplets_max=-1):
        self.root_dir = data_dir
        with open(img_file, 'r') as f:
            self.img_paths = f.readlines()  # Paths to the example 1854 images.
            self.img_paths = [c.replace('\n', '') for c in self.img_paths]
            assert len(self.img_paths) == 1854
        triplets = pd.read_csv(triplets_file)
        triplets_data = triplets.values.squeeze().tolist()[:n_triplets_max]
        self.triplets = [c.split('\t')[:4] for c in triplets_data]
        self.preprocess_fn = preprocess_fn
    
    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        idx0, idx1, idx2, lab = [int(c) - 1 for c in self.triplets[idx]]
        
        img0 = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.img_paths[idx0])))
        img1 = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.img_paths[idx1])))
        img2 = self.preprocess_fn(Image.open(os.path.join(self.root_dir, self.img_paths[idx2])))
        return img0, img1, img2, lab, idx
    
        


def is_image_file(filename, mode='img'):
    if(mode=='img'):
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)
    elif(mode=='np'):
        return any(filename.endswith(extension) for extension in NP_EXTENSIONS)


def make_dataset(dirs, mode='img'):
    if(not isinstance(dirs,list)):
        dirs = [dirs,]

    images = []
    for dir in dirs:
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname, mode=mode):
                    path = os.path.join(root, fname)
                    images.append(path)

    # print("Found %i images in %s"%(len(images),root))
    return images


def load_dataset(args, **kwargs):
    """Load datasets for perceptual tasks."""

    if args.data_dir is None:
        _data_dir = DEFAULT_DIRS[args.dataset]
    else:
        _data_dir = os.path.join(args.data_dir, DEFAULT_DIRS[args.dataset])

    if args.dataset == 'nights':
        preprocess = kwargs['preprocess']
        ds = NIGHTSTwoAFCDataset(
            _data_dir, args.split, preprocess=preprocess, #subdir='/000/'
            )
    
    elif args.dataset in ['bapps', 'bapps-bin']:
        if args.split == 'all':
            data_dir = [os.path.join(args.data_dir, split) for split in BAPPS_SPLITS]
        else:
            data_dir = os.path.join(args.data_dir, args.split)
        ds = BAPPSTwoAFCDataset(data_dir, args.img_res, args.dataset == 'bapps-bin')

    elif args.dataset == 'things':
        ds = THINGSDataset(
            args.data_dir, DEFAULT_DIRS['things-imgpaths'], DEFAULT_DIRS['things-triplets'],
            preprocess_fn=kwargs['preprocess'])
    
    logger = kwargs['logger']
    batch = next(iter(ds))
    try:
        print(len(ds), batch[0].shape, batch[0].max(), batch[0].min())
    except Exception as e:
        print(e)
    if args.n_ex == -1:
        args.n_ex = len(ds)
        logger.log(f'Using {args.n_ex} test points.')

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=0,
        shuffle=False,
    )

    return ds, loader


if __name__ == '__main__':
    pass

