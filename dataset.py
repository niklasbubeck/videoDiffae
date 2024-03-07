import os
from io import BytesIO
from pathlib import Path
from random import randint
import pickle
import glob 
import numpy as np
from tqdm import tqdm
import regex as re
import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, LSUNClass
import torch
import pandas as pd
from einops import rearrange
import nibabel as nib
import time

import torchvision.transforms.functional as Ftrans

MEAN_SAX_LV_VALUE = 222.7909
MAX_SAX_VALUE = 487.0
MEAN_4CH_LV_VALUE = 224.8285
MAX_4CH_LV_VALUE = 473.0

def get_torchvision_transforms(cfg, mode):
    assert mode in {'train', 'test'}
    if mode == 'train':
        transforms_cfg = cfg.dataset.train.transforms
    else:
        transforms_cfg = cfg.dataset.test.transforms

    transforms = []
    for t in transforms_cfg:
        if hasattr(torchvision.transforms, t['name']):
            transform_cls = getattr(torchvision.transforms, t['name'])(**t['params'])
        else:
            raise ValueError(f'Tranform {t["name"]} is not defined')
        transforms.append(transform_cls)
    transforms = torchvision.transforms.Compose(transforms)

    return transforms

def normalize_image_with_mean_lv_value(im, mean_value=MEAN_SAX_LV_VALUE, target_value=0.5):
    """ Normalize such that LV pool has value of 0.5. Assumes min value is 0.0. """
    im = im / (mean_value / target_value)
    im = im.clip(min=0.0, max=1.0)
    return im

class ImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts=['jpg'],
        do_augment: bool = True,
        do_transform: bool = True,
        do_normalize: bool = True,
        sort_names=False,
        has_subdir: bool = True,
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size

        # relative paths (make it shorter, saves memory and faster to sort)
        if has_subdir:
            self.paths = [
                p.relative_to(folder) for ext in exts
                for p in Path(f'{folder}').glob(f'**/*.{ext}')
            ]
        else:
            self.paths = [
                p.relative_to(folder) for ext in exts
                for p in Path(f'{folder}').glob(f'*.{ext}')
            ]
        if sort_names:
            self.paths = sorted(self.paths)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = os.path.join(self.folder, self.paths[index])
        img = Image.open(path)
        # if the image is 'rgba'!
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class SubsetDataset(Dataset):
    def __init__(self, dataset, size):
        assert len(dataset) >= size
        self.dataset = dataset
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        assert index < self.size
        return self.dataset[index]


class BaseLMDB(Dataset):
    def __init__(self, path, original_resolution, zfill: int = 5):
        self.original_resolution = original_resolution
        self.zfill = zfill
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.original_resolution}-{str(index).zfill(self.zfill)}'.encode(
                'utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        return img


def make_transform(
    image_size,
    flip_prob=0.5,
    crop_d2c=False,
):
    if crop_d2c:
        transform = [
            d2c_crop(),
            transforms.Resize(image_size),
        ]
    else:
        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
    transform.append(transforms.RandomHorizontalFlip(p=flip_prob))
    transform.append(transforms.ToTensor())
    transform.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    transform = transforms.Compose(transform)
    return transform


class FFHQlmdb(Dataset):
    def __init__(self,
                 path=os.path.expanduser('datasets/ffhq256.lmdb'),
                 image_size=256,
                 original_resolution=256,
                 split=None,
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=5)
        self.length = len(self.data)

        if split is None:
            self.offset = 0
        elif split == 'train':
            # last 60k
            self.length = self.length - 10000
            self.offset = 10000
        elif split == 'test':
            # first 10k
            self.length = 10000
            self.offset = 0
        else:
            raise NotImplementedError()

        transform = [
            transforms.Resize(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class Crop:
    def __init__(self, x1, x2, y1, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, img):
        return Ftrans.crop(img, self.x1, self.y1, self.x2 - self.x1,
                           self.y2 - self.y1)

    def __repr__(self):
        return self.__class__.__name__ + "(x1={}, x2={}, y1={}, y2={})".format(
            self.x1, self.x2, self.y1, self.y2)


def d2c_crop():
    # from D2C paper for CelebA dataset.
    cx = 89
    cy = 121
    x1 = cy - 64
    x2 = cy + 64
    y1 = cx - 64
    y2 = cx + 64
    return Crop(x1, x2, y1, y2)



class UKBB_lmdb(Dataset):
    def __init__(self,
                 path=os.path.expanduser('datasets/ukbb_MedMAE.lmdb'),
                 image_size=128,
                 original_resolution=128,
                 split=None,
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.slice_res = 8
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(
                txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length
    

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:

            key = f'{str(index).zfill(5)}'.encode(
            'utf-8')
            temp = txn.get(key)
            npz = pickle.loads(temp)
            sa = npz['sa']
            sa = torch.from_numpy(sa).unsqueeze(0).type(torch.float)         # c, h ,w
            # rearrange
            # sa = rearrange(sa, "c h w s t -> c s t h w")

            # only use middle 8 slices
            # if self.slice_res is not None: 
            #     start_slice = (sa.shape[1] - self.slice_res) // 2
            #     end_slice = start_slice + 8

            #     sa = sa[:, start_slice:end_slice, ...]
        

            # rand_time = randint(0, sa.shape[2] - 1)
            # rand_slice = randint(0, sa.shape[1] - 1)
            sa = torch.repeat_interleave(sa, 3, dim=0)
            sa = sa * 2 - 1
        return {'img': sa, 'index': index}




class UKBB(Dataset):

    def __init__(self, config, sbj_file=None, transforms=None) -> None:
        """
        Constructor Method
        """

        self.target_resolution = 128
        self.root_dir = "/vol/aimspace/projects/ukbb/data/cardiac/cardiac_segmentations/projects/MedMAE"
        self.transforms = transforms
        self.slice_res = 8



        self.fnames = []
        # self.la_fnames =[]
        # self.seg_fnames = []
        # self.meta_fnames = []
        subjects = os.listdir(self.root_dir)
        if sbj_file is not None:
            subjects = self.read_subject_numbers(sbj_file)

        for subject in tqdm(subjects):
            if len(self.fnames) >= 500:
                break
            try:
                self.fnames += glob.glob(f'{self.root_dir}/{subject}/processed_seg_allax.npz', recursive=True) 
                # self.la_fnames += glob.glob(f'{self.root_dir}/{subject}/la_2ch.nii.gz', recursive=True) 
                # self.seg_fnames += glob.glob(f'{self.root_dir}/{subject}/seg_sa_cropped.nii.gz', recursive=True)
            except:
                ImportError('No data found in the given path')
        
        subject_list = [self.extract_seven_concurrent_numbers(fname) for fname in tqdm(self.fnames)]
        with open("used_sbj.txt", 'w') as f:
            for item in tqdm(subject_list):
                f.write(str(item) + '\n')
        # some subjects dont have both, check for edge cases 
        # sa_subjects = [self.extract_seven_concurrent_numbers(file) for file in self.sa_fnames]
        # la_subjects = [self.extract_seven_concurrent_numbers(file) for file in self.la_fnames]
        # seg_subjects = [self.extract_seven_concurrent_numbers(file) for file in self.seg_fnames]  
        # common_subjects = self.common_subjects(la_subjects, sa_subjects)
        # common_subjects = self.common_subjects(common_subjects, seg_subjects)
        # self.sa_fnames = [f'{self.root_dir}/{subject}/sa_cropped.nii.gz' for subject in common_subjects]
        # self.la_fnames = [f'{self.root_dir}/{subject}/la_2ch.nii.gz' for subject in common_subjects]
        # self.seg_fnames = [f'{self.root_dir}/{subject}/seg_sa_cropped.nii.gz' for subject in common_subjects]

        print(f'{len(self.fnames)} files found in {self.root_dir}')
        # print(f'{len(self.la_fnames)} files found in {self.root_dir}/{folder}')
        # assert len(self.sa_fnames) == len(self.la_fnames) == len(self.seg_fnames), f"number of sa {len(self.sa_fnames)} and la {len(self.la_fnames)} and seg {len(self.seg_fnames)} not equal"
        # assert len(self.sa_fnames) != 0, f"Given directory contains 0 images. Please check on the given root: {self.root_dir}"


    def extract_seven_concurrent_numbers(self, text):
        pattern = r'\b\d{7}\b'
        seven_numbers = re.findall(pattern, text)
        return seven_numbers[0]

    def common_subjects(self, la_subjects, sa_subjects):
        # Convert both lists to sets
        la_set = set(la_subjects)
        sa_set = set(sa_subjects)
        
        # Find the common subjects using set intersection
        common_subjects_set = la_set.intersection(sa_set)
        
        # Convert the result back to a list
        common_subjects_list = list(common_subjects_set)
    
        return common_subjects_list

    def read_subject_numbers(self, file_path):
        try:
            with open(file_path, 'r') as file:
                subject_numbers = [line.strip() for line in file.readlines()]
                subject_numbers = [num for num in subject_numbers if num.isdigit() and len(num) == 7]
                return subject_numbers
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return []
        except Exception as e:
            print(f"Error: {e}")
            return []

    @property
    def indices(self):
        return self._indices
    
    @indices.setter
    def indices(self, value):
        self._indices = value 

    # @property
    # def fnames(self):
    #     return self.targets_fnames

    
    def load_nifti(self, fname:str):
        nii = nib.load(fname).get_fdata()
        return nii

    def load_meta_patient(self, fname:str):

        file = open(fname, 'r')
        content = file.read()
        config_dict = {}
        lines = content.split("\n") #split it into lines
        for path in lines:
            split = path.split(": ")
            if len(split) != 2: 
                break
            key, value = split[0], split[1]
            config_dict[key] = value

        return config_dict

    def min_max(self, x, min, max):
        # Scale to SA 
        std = (x- x.min()) / (x.max() - x.min())
        x = std * (max - min) + min
        return x 

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        start_time = time.time()
        process_npy = np.load(self.fnames[idx])
        sa = process_npy['sax'] # [H, W, S, T]
        la = process_npy['lax'] # [H, W, S, T]
        sa_seg = process_npy['seg_sax'] # [H, W, S, T]

        sa = normalize_image_with_mean_lv_value(sa)
        la = normalize_image_with_mean_lv_value(la)

        h, w, s ,t = sa.shape

        # # load the short axis-image
        # sa = self.load_nifti(self.sa_fnames[idx]) # h, w, s, t
        # sa = self.min_max(sa, 0, 1) 

        # # load the segmentation short-axis image
        # sa_seg = self.load_nifti(self.seg_fnames[idx]) #h ,w, s, t 


        # # load the long-axis image 
        # la = self.load_nifti(self.la_fnames[idx]) # h, w, s, t
        # la = self.min_max(la, 0, 1)
    
        # la_o_h, la_o_w, la_o_s, la_o_t = la.shape

        # # crop la image 
        # left = (la_o_w - self.target_resolution) // 2
        # top = (la_o_h - self.target_resolution) // 2
        # right = left + self.target_resolution
        # bottom = top + self.target_resolution
        # la = la[top:bottom, left:right, 0, 0]

        # # error handling for la images smaller than 128
        # la_h, la_w = la.shape
        # if la_h != self.target_resolution or la_w != self.target_resolution:
        #     print(f"Weird stuff: {la_o_h} {la_o_w} --> {la_h} {la_w}")
        #     return self.__getitem__(idx + 1)
    
        # add channel dimension and float it 
        sa = torch.from_numpy(sa).unsqueeze(0).type(torch.float)         # c, h ,w ,s ,t 
        la = torch.from_numpy(la).unsqueeze(0).type(torch.float)         # c, h, w
        sa_seg = torch.from_numpy(sa_seg).unsqueeze(0).type(torch.float) # c, h, w, s, t 
        
        # rearrange
        sa = rearrange(sa, "c h w s t -> c s t h w")
        la = rearrange(la, "c h w s t -> c s t h w")
        sa_seg = rearrange(sa_seg, "c h w s t -> c s t h w")
        
        # apply transformations
        # sa = self.transforms(sa)
        # la = self.transforms(la)
        # sa_seg = self.transforms(sa_seg)

        if self.slice_res is not None: 
            start_slice = (sa.shape[1] - self.slice_res) // 2
            end_slice = start_slice + 8

            sa = sa[:, start_slice:end_slice, ...]
            sa_seg = sa_seg[:, start_slice:end_slice, ...]
        

        # if self.normalize:
        #     sa= normalize_neg_one_to_one(sa)
        #     la = normalize_neg_one_to_one(la)

        # if self.crop_along_bbox:
        #     _, _, ymin, ymax, xmin, xmax = find_bounding_box3D(sa_seg[0,5,...])

        #     #bounderis to ensure that every crop is target_res x target_res
        #     min_x, max_x = self.target_resolution//2, w - self.target_resolution//2
        #     min_y, max_y = self.target_resolution//2, h - self.target_resolution//2

        #     cy, cx = (ymin + ymax)//2, (xmin+xmax)//2
        #     cx = max(min(cx, max_x), min_x)
        #     cy = max(min(cy, max_y), min_y)


        #     x_top_left = max(0, cx - self.target_resolution // 2)
        #     y_top_left = max(0, cy - self.target_resolution // 2)
            
        #     # Calculate the coordinates of the bottom-right corner
        #     x_bottom_right = min(w, cx + self.target_resolution // 2)
        #     y_bottom_right = min(h, cy + self.target_resolution // 2)

        #     sa = sa[..., y_top_left:y_bottom_right, x_top_left:x_bottom_right]
        #     sa_seg = sa_seg[..., y_top_left:y_bottom_right, x_top_left:x_bottom_right]

        rand_time = randint(0, sa.shape[2] - 1)
        rand_slice = randint(0, sa.shape[1] - 1)
        sa = torch.repeat_interleave(sa, 3, dim=0)
        fnames = self.fnames[idx]

        end_time = time.time()
        print("Time needed: ", end_time - start_time)
        return {'img': sa[:, rand_slice, rand_time, ...], "index": idx}


class CelebAlmdb(Dataset):
    """
    also supports for d2c crop.
    """
    def __init__(self,
                 path,
                 image_size,
                 original_resolution=128,
                 split=None,
                 as_tensor: bool = True,
                 do_augment: bool = True,
                 do_normalize: bool = True,
                 crop_d2c: bool = False,
                 **kwargs):
        self.original_resolution = original_resolution
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)
        self.crop_d2c = crop_d2c

        if split is None:
            self.offset = 0
        else:
            raise NotImplementedError()

        if crop_d2c:
            transform = [
                d2c_crop(),
                transforms.Resize(image_size),
            ]
        else:
            transform = [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ]

        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if as_tensor:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index < self.length
        index = index + self.offset
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class Horse_lmdb(Dataset):
    def __init__(self,
                 path=os.path.expanduser('datasets/horse256.lmdb'),
                 image_size=128,
                 original_resolution=256,
                 do_augment: bool = True,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        print(path)
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index}


class Bedroom_lmdb(Dataset):
    def __init__(self,
                 path=os.path.expanduser('datasets/bedroom256.lmdb'),
                 image_size=128,
                 original_resolution=256,
                 do_augment: bool = True,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 **kwargs):
        self.original_resolution = original_resolution
        print(path)
        self.data = BaseLMDB(path, original_resolution, zfill=7)
        self.length = len(self.data)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = self.data[index]
        img = self.transform(img)
        return {'img': img, 'index': index}


class CelebAttrDataset(Dataset):

    id_to_cls = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 folder,
                 image_size=64,
                 attr_path=os.path.expanduser(
                     'datasets/celeba_anno/list_attr_celeba.txt'),
                 ext='png',
                 only_cls_name: str = None,
                 only_cls_value: int = None,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 d2c: bool = False):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.ext = ext

        # relative paths (make it shorter, saves memory and faster to sort)
        paths = [
            str(p.relative_to(folder))
            for p in Path(f'{folder}').glob(f'**/*.{ext}')
        ]
        paths = [str(each).split('.')[0] + '.jpg' for each in paths]

        if d2c:
            transform = [
                d2c_crop(),
                transforms.Resize(image_size),
            ]
        else:
            transform = [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        with open(attr_path) as f:
            # discard the top line
            f.readline()
            self.df = pd.read_csv(f, delim_whitespace=True)
            self.df = self.df[self.df.index.isin(paths)]

        if only_cls_name is not None:
            self.df = self.df[self.df[only_cls_name] == only_cls_value]

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        name = row.name.split('.')[0]
        name = f'{name}.{self.ext}'

        path = os.path.join(self.folder, name)
        img = Image.open(path)

        labels = [0] * len(self.id_to_cls)
        for k, v in row.items():
            labels[self.cls_to_id[k]] = int(v)

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'index': index, 'labels': torch.tensor(labels)}


class CelebD2CAttrDataset(CelebAttrDataset):
    """
    the dataset is used in the D2C paper. 
    it has a specific crop from the original CelebA.
    """
    def __init__(self,
                 folder,
                 image_size=64,
                 attr_path=os.path.expanduser(
                     'datasets/celeba_anno/list_attr_celeba.txt'),
                 ext='jpg',
                 only_cls_name: str = None,
                 only_cls_value: int = None,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 d2c: bool = True):
        super().__init__(folder,
                         image_size,
                         attr_path,
                         ext=ext,
                         only_cls_name=only_cls_name,
                         only_cls_value=only_cls_value,
                         do_augment=do_augment,
                         do_transform=do_transform,
                         do_normalize=do_normalize,
                         d2c=d2c)


class CelebAttrFewshotDataset(Dataset):
    def __init__(
        self,
        cls_name,
        K,
        img_folder,
        img_size=64,
        ext='png',
        seed=0,
        only_cls_name: str = None,
        only_cls_value: int = None,
        all_neg: bool = False,
        do_augment: bool = False,
        do_transform: bool = True,
        do_normalize: bool = True,
        d2c: bool = False,
    ) -> None:
        self.cls_name = cls_name
        self.K = K
        self.img_folder = img_folder
        self.ext = ext

        if all_neg:
            path = f'data/celeba_fewshots/K{K}_allneg_{cls_name}_{seed}.csv'
        else:
            path = f'data/celeba_fewshots/K{K}_{cls_name}_{seed}.csv'
        self.df = pd.read_csv(path, index_col=0)
        if only_cls_name is not None:
            self.df = self.df[self.df[only_cls_name] == only_cls_value]

        if d2c:
            transform = [
                d2c_crop(),
                transforms.Resize(img_size),
            ]
        else:
            transform = [
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
            ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        name = row.name.split('.')[0]
        name = f'{name}.{self.ext}'

        path = os.path.join(self.img_folder, name)
        img = Image.open(path)

        # (1, 1)
        label = torch.tensor(int(row[self.cls_name])).unsqueeze(-1)

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'index': index, 'labels': label}


class CelebD2CAttrFewshotDataset(CelebAttrFewshotDataset):
    def __init__(self,
                 cls_name,
                 K,
                 img_folder,
                 img_size=64,
                 ext='jpg',
                 seed=0,
                 only_cls_name: str = None,
                 only_cls_value: int = None,
                 all_neg: bool = False,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True,
                 is_negative=False,
                 d2c: bool = True) -> None:
        super().__init__(cls_name,
                         K,
                         img_folder,
                         img_size,
                         ext=ext,
                         seed=seed,
                         only_cls_name=only_cls_name,
                         only_cls_value=only_cls_value,
                         all_neg=all_neg,
                         do_augment=do_augment,
                         do_transform=do_transform,
                         do_normalize=do_normalize,
                         d2c=d2c)
        self.is_negative = is_negative


class CelebHQAttrDataset(Dataset):
    id_to_cls = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]
    cls_to_id = {v: k for k, v in enumerate(id_to_cls)}

    def __init__(self,
                 path=os.path.expanduser('datasets/celebahq256.lmdb'),
                 image_size=None,
                 attr_path=os.path.expanduser(
                     'datasets/celeba_anno/CelebAMask-HQ-attribute-anno.txt'),
                 original_resolution=256,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        super().__init__()
        self.image_size = image_size
        self.data = BaseLMDB(path, original_resolution, zfill=5)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        with open(attr_path) as f:
            # discard the top line
            f.readline()
            self.df = pd.read_csv(f, delim_whitespace=True)

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.name
        img_idx, ext = img_name.split('.')
        img = self.data[img_idx]

        labels = [0] * len(self.id_to_cls)
        for k, v in row.items():
            labels[self.cls_to_id[k]] = int(v)

        if self.transform is not None:
            img = self.transform(img)
        return {'img': img, 'index': index, 'labels': torch.tensor(labels)}


class CelebHQAttrFewshotDataset(Dataset):
    def __init__(self,
                 cls_name,
                 K,
                 path,
                 image_size,
                 original_resolution=256,
                 do_augment: bool = False,
                 do_transform: bool = True,
                 do_normalize: bool = True):
        super().__init__()
        self.image_size = image_size
        self.cls_name = cls_name
        self.K = K
        self.data = BaseLMDB(path, original_resolution, zfill=5)

        transform = [
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
        ]
        if do_augment:
            transform.append(transforms.RandomHorizontalFlip())
        if do_transform:
            transform.append(transforms.ToTensor())
        if do_normalize:
            transform.append(
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        self.transform = transforms.Compose(transform)

        self.df = pd.read_csv(f'data/celebahq_fewshots/K{K}_{cls_name}.csv',
                              index_col=0)

    def pos_count(self, cls_name):
        return (self.df[cls_name] == 1).sum()

    def neg_count(self, cls_name):
        return (self.df[cls_name] == -1).sum()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.name
        img_idx, ext = img_name.split('.')
        img = self.data[img_idx]

        # (1, 1)
        label = torch.tensor(int(row[self.cls_name])).unsqueeze(-1)

        if self.transform is not None:
            img = self.transform(img)

        return {'img': img, 'index': index, 'labels': label}


class Repeat(Dataset):
    def __init__(self, dataset, new_len) -> None:
        super().__init__()
        self.dataset = dataset
        self.original_len = len(dataset)
        self.new_len = new_len

    def __len__(self):
        return self.new_len

    def __getitem__(self, index):
        index = index % self.original_len
        return self.dataset[index]
