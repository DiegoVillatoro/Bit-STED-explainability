import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import os
import cv2
import warnings
#import rasterio as rio
from tifffile import imread
from utils.utils import VI
import random
#import time

#load png data
class Load_data_agave(Dataset):
    def __init__(self, data_folder: str, split = 'Val'):
        
        split = split.lower()
        if split not in ['train', 'val']:
            split = 'val' 
        self.split = split
        self.data_folder = data_folder
        
        #self.folder_tiles = "/home/a01328525/Datasets Tiles/Zone108_octubre/"
        
        print('Load '+split+" data")
        file_names = os.listdir(data_folder+split+'/images/')
        file_names = [file_name[:-4] for file_name in file_names if file_name[-3:]=='png']
        self.file_names = file_names
        self.batch_count = 0


    def __getitem__(self, index):
        # 1. Image
        # -----------------------------------------------------------------------------------
        image_path = self.data_folder + self.split+ '/images/' + self.file_names[index] + '.png'
        #with rio.open(image_path) as img :
        #    image = img.read()[0:N_channels,:,:]
        flag_1 = time.time()
        image = cv2.imread(image_path)
        flag_2 = time.time()
        # Extract image as PyTorch tensor
        image = torch.Tensor(image).permute(2, 0, 1) # shape: n_channels x W x H
        flag_3 = time.time()
        # 2. Label
        # -----------------------------------------------------------------------------------
        label_path = self.data_folder + self.split+'/labels/' + self.file_names[index] + '.txt'
        flag_4 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5)) 
        
        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes
        flag_5 = time.time()
        return image_path, image, targets

    def __len__(self):
        return len(self.file_names)

    def collate_fn(self, batch):
        paths, images, targets = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        try:
            targets = torch.cat(targets, 0)
        except RuntimeError:
            targets = None  # No boxes for an image

        # Resize images to input shape
        images = torch.stack([image for image in images])

        self.batch_count += 1

        return paths, images, targets

def get_corners(bboxes):
    
    """Get corners of bounding boxes

    Parameters
    ----------

    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      

    """
    width = (bboxes[:,2]).reshape(-1,1)
    #print(width.shape)
    height = (bboxes[:,3]).reshape(-1,1)

    x1 = (bboxes[:,0]-bboxes[:,2]/2).reshape(-1,1)
    y1 = (bboxes[:,1]-bboxes[:,3]/2).reshape(-1,1)
    #print(x1.shape)
    x2 = x1 + width
    y2 = y1 

    x3 = x1
    y3 = y1 + height

    x4 = x2.reshape(-1,1)
    y4 = y3.reshape(-1,1)

    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))

    return corners
def rotate_box(corners,angle,  cx, cy, h, w):
    
    """Rotate the bounding box.


    Parameters
    ----------

    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`

    angle : float
        angle by which the image is to be rotated

    cx : int
        x coordinate of the center of image (about which the box will be rotated)

    cy : int
        y coordinate of the center of image (about which the box will be rotated)

    h : int 
        height of the image

    w : int 
        width of the image

    Returns
    -------

    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))

    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)


    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T

    calculated = calculated.reshape(-1,8)

    return calculated
def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
    
    Returns 
    -------
    
    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    """
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]
    
    xmin = np.clip(np.min(x_,1).reshape(-1,1), 0, 0.999)
    ymin = np.clip(np.min(y_,1).reshape(-1,1), 0, 0.999)
    xmax = np.clip(np.max(x_,1).reshape(-1,1), 0, 0.999)
    ymax = np.clip(np.max(y_,1).reshape(-1,1), 0, 0.999)
    
    xc = (xmax+xmin)/2
    yc = (ymax+ymin)/2
    w = (xmax-xmin)
    h = (ymax-ymin)
    
    final = np.hstack((xc, yc, w, h,corners[:,8:]))
    
    return final

def horisontal_flip(images, targets, masks=None):
    images = torch.flip(images, [-1])
    targets[:, 1] = 0.9999 - targets[:, 1] #xc coordinate
    
    masks = torch.flip(masks, [-1]) if masks!=None else None
    return images, targets, masks

class Load_data_agave_multispectral(Dataset):
    def __init__(self, data_folder: str, split = 'Val', augment=False, prob=0.5):
        
        split = split.lower()
        if split not in ['train', 'val']:
            split = 'val' 
        self.split = split
        self.data_folder = data_folder
        
        #for each txt should be exist at least a tif image
        file_names = os.listdir(data_folder+split+'/images/')
        file_names = np.array([file_name[:-4] for file_name in file_names if file_name[-3:]=='tif'])
        np.random.shuffle(file_names)
        #file_names = file_names[0:(len(file_names))//2]#select only a fraction of entire dataset
        
        self.file_names = file_names
        print('Load '+split+" data with "+str(len(file_names))+" images")
        self.batch_count = 0
        self.augment = augment
        self.prob = prob

    def __getitem__(self, index):
        # 1. Image
        # -----------------------------------------------------------------------------------
        #flag_1 = time.time()
        image_path = self.data_folder + self.split+'/images/' + self.file_names[index] + '.tif'
        image = imread(image_path)
        
        #flag_2 = time.time()
        image = torch.Tensor(image.transpose(2, 0, 1)) # shape: n_channels x W x H
        #flag_3 = time.time()
        # 2. Label
        # -----------------------------------------------------------------------------------
        label_path = self.data_folder + self.split+'/labels/' + self.file_names[index] + '.txt'

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5)) 
        #flag_4 = time.time()
        if self.augment:
            if np.random.random() < self.prob:
                p = np.random.random()
                if p<0.33:
                    ##################### Reflexion ######################
                    image, boxes = horisontal_flip(image, boxes)
                if p>=0.33 and p<0.66:
                ##################### Color ######################
                    image = torchvision.transforms.GaussianBlur(kernel_size=5)(image)
                if p>=0.66:
                    ##################### Rotation ######################
                    angle = np.random.randint(-10, 10)
                    image = torchvision.transforms.functional.rotate(image, angle)
                    if len(boxes)>0:#if there are boxes in target then rotate the boxes
                        corners = get_corners(boxes[:,1:5])
                        corners = rotate_box(corners, angle, 0.5, 0.5, 1, 1)
                        new_bbox = get_enclosing_box(corners)
                        boxes[:,1:5] = torch.from_numpy(new_bbox)
                        boxes = boxes[(boxes[:,3]*boxes[:,4] >= 15/(224**2))]
        #flag_5 = time.time()
        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes
        #flag_6 = time.time()
        #print("Time-> rioOp: %0.4f, Tensor: %0.4f, lab: %0.4f, aug: %0.4f, final: %0.4f, all:%0.4f"%(flag_2-flag_1, flag_3-flag_2,  flag_4-flag_3, flag_5-flag_4, flag_6-flag_5, flag_6-flag_1))
        return image_path, image, targets

    def __len__(self):
        return len(self.file_names)

    def collate_fn(self, batch):

        paths, images, targets = zip(*batch)

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets]
        
        # Add a sample index to identify the target image it belongs to in the batch size
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        targets = torch.cat(targets, 0)

        images = torch.stack(images, dim=0)

        self.batch_count += 1

        return paths, images, targets

class Load_data_agave_circles(Dataset):
    def __init__(self, data_folder: str, split = 'Val'):
        
        split = split.lower()
        if split not in ['train', 'val']:
            split = 'val' 
        self.split = split
        self.data_folder = data_folder
        
        file_names = os.listdir(data_folder+split+'/images/')
        file_names = [file_name[:-4] for file_name in file_names if file_name[-3:]=='png']
        self.file_names = file_names
        self.batch_count = 0


    def __getitem__(self, index):
        # 1. Image
        # -----------------------------------------------------------------------------------
        image_path = self.data_folder + self.split + '/images/' + self.file_names[index] + '.png'
        #with rio.open(image_path) as img :
        #    image = img.read()[0:N_channels,:,:]
        image = cv2.imread(image_path)

        # Extract image as PyTorch tensor
        image = torch.Tensor(image).permute(2, 0, 1) # shape: n_channels x W x H

        # 2. Label
        # -----------------------------------------------------------------------------------
        label_path = self.data_folder + self.split + '/labels/' + self.file_names[index] + '.txt'

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 4)) 
        
        targets = torch.zeros((len(boxes), 5))
        targets[:, 1:] = boxes


        return image_path, image, targets

    def __len__(self):
        return len(self.file_names)

    def collate_fn(self, batch):
        paths, images, targets = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        try:
            targets = torch.cat(targets, 0)
        except RuntimeError:
            targets = None  # No boxes for an image

        # Resize images to input shape
        images = torch.stack([image for image in images])

        self.batch_count += 1

        return paths, images, targets
    
def rotate_point(point, angle, origin):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point[:,0], point[:,1]

    qx = ox + np.cos(angle*np.pi/180) * (px - ox) - np.sin(angle*np.pi/180) * (py - oy)
    qy = oy + np.sin(angle*np.pi/180) * (px - ox) + np.cos(angle*np.pi/180) * (py - oy)

    return torch.cat((qx.view(-1, 1), qy.view(-1, 1)), axis=-1 )

def randomize_background(img, mask):
    bg = (mask == 0)
    
    bg = torch.stack((bg,) * 3, axis=-1)
    bg = bg.permute(2, 0, 1)
    
    noisy = img.clone()
    #noisy[bg] = torch.ones_like(img[bg])
    
    p = np.random.random()
    if p < 0.25:
        noise = torch.randn_like(img) * 0.25
        return img * mask + (0.75*img + 0.25*noise) * (1 - mask)
    elif p>=0.25 and p<0.5:
        noise = torch.ones_like(img)
        return img * mask + (noise) * (1 - mask)
    elif p>=0.5 and p<0.75:
        noise = torch.ones_like(img)*p
        return img * mask + (noise) * (1 - mask)
    else:
        noise = torch.zeros_like(img)
        return img * mask + (noise) * (1 - mask)
def erase_background(image, mask):
    
    bg = (~mask.bool())
    h, w = image.shape[-2:]
    x0 = torch.randint(0, w//2, (1,))
    y0 = torch.randint(0, h//2, (1,))
    x1 = x0 + torch.randint(w//4, w//2, (1,))
    y1 = y0 + torch.randint(h//4, h//2, (1,))

    image[:, y0:y1, x0:x1] *= (~bg[y0:y1, x0:x1])
    return image
    
def randomize_plant(img, mask):
    bg = (mask == 0)
    
    bg = torch.stack((bg,) * 3, axis=-1)
    bg = bg.permute(2, 0, 1).float()
    
    #noisy = img.clone()
    #noisy[bg] = torch.ones_like(img[bg])
    
    #p = np.random.random()
    noise = torch.randn_like(img)*img.max()
    
    return img * bg + 0.5*(img + noise) * ( 1-bg)
    
def random_box_cutout(img, mask=None, min_frac=0.1, max_frac=0.2):
    """
    Randomly removes a rectangular region (cutout).
    img: Tensor (3,H,W)
    mask: Tensor (1,H,W) optional, also cutout same region
    """
    box_cut = img.clone()
    C, H, W = img.shape

    box_w = int(random.uniform(min_frac, max_frac) * W)
    box_h = int(random.uniform(min_frac, max_frac) * H)

    x0 = random.randint(0, W - box_w)
    y0 = random.randint(0, H - box_h)
    #print(x0,x0+box_w, y0,y0+box_h)
    
    region_fg = mask[y0:y0+box_h, x0:x0+box_w].sum()
    if region_fg > 0:
        return img
    # Zero the box
    box_cut[:,y0:y0+box_h, x0:x0+box_w] = 1

    return box_cut
    
def safe_object_erasing(image, boxes, max_ratio=0.5):
    
    for box in boxes:
        if np.random.rand() < 0.5:
            _, xc, yc, r = image.shape[-1]*box
            x1 = xc-r if xc-r>0 else 0
            x2 = xc+r if xc+r<image.shape[-1] else image.shape[-1]
            y1 = yc-r if yc-r>0 else 0
            y2 = yc+r if yc+r<image.shape[-2] else image.shape[-2]

            W = x2 - x1
            H = y2 - y1
    
            # Erase size
            ew = int(np.random.uniform(0.1, max_ratio) * W)
            eh = int(np.random.uniform(0.1, max_ratio) * H)

            # Select corner
            corner = np.random.choice(["tl", "tr", "bl", "br"])
        
            if corner == "tl":
                ex, ey = x1, y1
            elif corner == "tr":
                ex, ey = x2 - ew, y1
            elif corner == "bl":
                ex, ey = x1, y2 - eh
            else:  # br
                ex, ey = x2 - ew, y2 - eh

            #print(ey, eh, ex, ew)
            # Apply erase (mean, noise, or blur)
            image[:,int(ey):int(ey+eh), int(ex):int(ex+ew)] = 0

    return image#, boxes
    
def blur_background(image, mask):
    blurred = torchvision.transforms.GaussianBlur(11)(image)
    image[:, ~mask[0].bool()] = blurred[:, ~mask[0].bool()]
    return image
def plant_spectral_jitter(image, mask):
    jitter = 0.06 * torch.randn_like(image)
    image[:, mask[0].bool()] += jitter[:, mask[0].bool()]
    return torch.clamp(image, 0, 1)
    
class Load_data_agave_circles_multispectral(Dataset):
    def __init__(self, data_folder: str, split = 'Val', augment=False, prob = 0.5, extract_masks=False, n_channels=3, n_feats=4):
        
        split = split.lower()
        if split not in ['train', 'val']:
            split = 'val' 
        self.split = split
        self.data_folder = data_folder
        self.extract_masks = extract_masks
        
        #self.folder_tiles = "/home/a01328525/Datasets Tiles/Zone108_octubre/"
        if split not in ['train', 'val']:
            split = 'val' 
        file_names = os.listdir(data_folder+split+'/images/')
        file_names = np.array([file_name[:-4] for file_name in file_names if file_name[-3:]=='tif'])
        np.random.shuffle(file_names)
        #file_names = file_names[0:(len(file_names))//5]#select only a fraction of entire dataset
        
        print('Load '+split+" data with "+str(len(file_names))+" images")
        self.file_names = file_names
        self.batch_count = 0
        self.augment = augment
        self.prob = prob
        self.n_channels = n_channels
        self.n_feats = n_feats
        
        #self.test_i = 0
    def __getitem__(self, index):

    # --------------------------------------------------
        # 1. Load image
        # --------------------------------------------------
        file_id = self.file_names[index]
        image_path = os.path.join(self.data_folder, self.split, 'images', file_id + '.tif')

        image = imread(image_path)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        # --------------------------------------------------
        # 2. Extract mask
        # --------------------------------------------------
        mask = None
        if self.extract_masks:
            mask = (image[8:9] > 0).float()
        
        image = image[:self.n_channels]

        # --------------------------------------------------
        # 3. Load labels safely
        # --------------------------------------------------
        label_path = os.path.join(self.data_folder, self.split, 'labels', file_id + '.txt')

        if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
            boxes = torch.from_numpy(
                np.loadtxt(label_path).reshape(-1, self.n_feats)
            ).float()
        else:
            boxes = torch.empty((0, self.n_feats), dtype=torch.float32)

        # --------------------------------------------------
        # 4. Apply augmentation
        # --------------------------------------------------
        if self.augment and torch.rand(1) < self.prob:
            image, boxes, mask = self.apply_augmentation(image, boxes, mask)

        # --------------------------------------------------
        # 5. Build targets
        # --------------------------------------------------
        targets = torch.zeros((len(boxes), self.n_feats + 1))
        if len(boxes) > 0:
            targets[:, 1:] = boxes

        if self.extract_masks:
            return image_path, image, targets, mask
        else:
            return image_path, image, targets
    def apply_augmentation(self, image, boxes, mask):

        aug_functions = [
            self.aug_flip,
            self.aug_rotate,
            self.aug_randomize_background,
            self.aug_erase_background,
            self.aug_safe_object_erasing,
            self.aug_randomize_plant,
            self.aug_blur,
            self.aug_gaussian_noise,
            self.aug_spectral_jitter,
            self.aug_not
        ]
        
        aug = random.choice(aug_functions)
        #for plot testing of each augmentation
        #aug = aug_functions[self.test_i]
        #self.test_i+=1
        #if self.test_i == len(aug_functions):
        #    self.test_i=0
        return aug(image, boxes, mask)
    
    def aug_flip(self, image, boxes, mask):
        return horisontal_flip(image, boxes, mask)
    def aug_rotate(self, image, boxes, mask):
        angle = torch.randint(-20, 21, (1,)).item()

        image = torchvision.transforms.functional.rotate(image, angle)

        if len(boxes) > 0:
            boxes[:, 1:3] = rotate_point(boxes[:, 1:3], -angle, (0.5, 0.5))
            boxes = boxes[
                (boxes[:, 1] >= 0) & (boxes[:, 1] <= 1) &
                (boxes[:, 2] >= 0) & (boxes[:, 2] <= 1)
            ]

        if mask is not None:
            mask = torchvision.transforms.functional.rotate(mask, angle)
                    
        return image, boxes, mask
    def aug_randomize_background(self, image, boxes, mask):
        return randomize_background(image, mask[0]), boxes, mask
    def aug_erase_background(self, image, boxes, mask):
        return erase_background(image, mask[0]), boxes, mask #erase a box of the background
    def aug_safe_object_erasing(self, image, boxes, mask):
        return safe_object_erasing(image, boxes), boxes, mask
    def aug_randomize_plant(self, image, boxes, mask):
        image = randomize_plant(image, mask[0])
        boxes = torch.empty(0, self.n_feats)
        return image, boxes, mask
    def aug_blur(self, image, boxes, mask):
        return torchvision.transforms.GaussianBlur(kernel_size=9)(image), boxes, mask
    def aug_gaussian_noise(self, image, boxes, mask):
        std=0.08
        mean=0
        noise = torch.randn_like(image) * std + mean
        image = image + noise
        image = torch.clamp(image, 0.0, 1.0)
        return image, boxes, mask
    def aug_spectral_jitter(self, image, boxes, mask):
        return plant_spectral_jitter(image, mask), boxes, mask
    def aug_not(self, image, boxes, mask):
        return image, boxes, mask
                
    def getitem2(self, index):
        # 1. Image
        # -----------------------------------------------------------------------------------
        image_path = self.data_folder + self.split+'/images/' + self.file_names[index] + '.tif'
        image = imread(image_path)
        #flag_2 = time.time()
        # Extract image as PyTorch tensor
        image = torch.Tensor(image.transpose(2, 0, 1)) # shape: n_channels x W x H
        
        if self.extract_masks:
            #mask = image[8, ...].unsqueeze(0)
            #mask = image[5, ...].unsqueeze(0)
            mask = torch.where(image[8, ...].unsqueeze(0)>0, image[8, ...].unsqueeze(0), 0)
        else:
            mask = None
        image = image[0:self.n_channels, ...]
        # 2. Label
        # -----------------------------------------------------------------------------------
        label_path = self.data_folder + self.split+ '/labels/' + self.file_names[index] + '.txt'

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            #boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 4)) 
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, self.n_feats)) 

        if self.augment and np.random.rand() < self.prob:
            p = np.random.rand()
            n_augs = 9
            #geometric transformation
            if p<1/n_augs:
                ##################### Reflexion ######################
                image, boxes, mask = horisontal_flip(image, boxes, mask)
            elif p<2/n_augs:
                ##################### Rotation ######################
                angle = np.random.randint(-20, 20)
                image = torchvision.transforms.functional.rotate(image, angle)
                boxes[:,1:3] = rotate_point(boxes[:,1:3], -angle, (0.5, 0.5))
                boxes = boxes[(boxes[:,1] >= 0) & (boxes[:,1] <= 1) & (boxes[:,2] >= 0) & (boxes[:,2] <= 1)]
                if self.extract_masks:
                    mask = torchvision.transforms.functional.rotate(mask, angle)
            
            #background manipulation
            elif p<3/n_augs:
                image = randomize_background(image, mask[0])
            elif p<7/n_augs:
                image = erase_background(image, mask[0])#erase a box of the background
            
            #occlusion
            elif p<4/n_augs:
                image = safe_object_erasing(image, boxes)
            elif p<8/n_augs:
                image = randomize_plant(image, mask[0])
                boxes = torch.empty(0, self.n_feats)  
                
            #spectral variations
            elif p<5/n_augs:
                ##################### Color ######################
                image = torchvision.transforms.GaussianBlur(kernel_size=9)(image)        
            #elif p<0.4:
                ##################### Color with mask ######################
            #    image = blur_background(image, mask)
            elif p<6/n_augs:
                ##################### Color ######################
                std=0.15
                mean=0
                noise = torch.randn_like(image) * std + mean
                image = image + noise
                image = torch.clamp(image, 0.0, 1.0)
            elif p <7/n_augs:
                image = plant_spectral_jitter(image, mask)

        #targets = torch.zeros((len(boxes), 5))
        targets = torch.zeros((len(boxes), self.n_feats+1))
        targets[:, 1:] = boxes
        #targets[:, 1] = 0 #only one class
        
        if self.extract_masks:
            return image_path, image, targets, mask
        else:
            return image_path, image, targets

    def __len__(self):
        return len(self.file_names)

    def collate_fn(self, batch):
        
        if self.extract_masks:
            paths, images, targets, masks = zip(*batch)
            masks = torch.stack(masks, dim=0)
        else:
            paths, images, targets = zip(*batch)

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets]
        
        # Add a sample index to identify the target image it belongs to in the batch size
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        targets = torch.cat(targets, 0)

        images = torch.stack(images, dim=0)
        
        self.batch_count += 1
        
        if self.extract_masks:
            return paths, images, targets, masks
        else:
            return paths, images, targets
    
class Load_data_acuario(Dataset):
    def __init__(self, data_folder: str, split = 'test'):
        
        self.split = split.lower()
        self.data_folder = data_folder
        
        #self.folder_tiles = "/home/a01328525/Datasets Tiles/Zone108_octubre/"
        if self.split not in ['train', 'test']:
            self.split = 'test' 
           
        print('Load '+split+" data")
        file_names = os.listdir(data_folder+split+'/')
        file_names = [file_name[:-4] for file_name in file_names if file_name[-3:]=='jpg']
        self.file_names = file_names
        self.batch_count = 0


    def __getitem__(self, index):
        # 1. Image
        # -----------------------------------------------------------------------------------
        image_path = self.data_folder + self.split+'/' + self.file_names[index] + '.jpg'
        #with rio.open(image_path) as img :
        #    image = img.read()[0:N_channels,:,:]
        image = cv2.imread(image_path)

        # Extract image as PyTorch tensor
        image = torch.Tensor(image).permute(2, 0, 1) # shape: n_channels x W x H

        # 2. Label
        # -----------------------------------------------------------------------------------
        label_path = self.data_folder + self.split+ '/' + self.file_names[index] + '.txt'
        
        boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
        #boxes[:,0] = 0
        
        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes
        
        #resize image
        image = torchvision.transforms.Resize((224,224))(image)


        return image_path, image, targets

    def __len__(self):
        return len(self.file_names)

    def collate_fn(self, batch):
        paths, images, targets = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        try:
            targets = torch.cat(targets, 0)
        except RuntimeError:
            targets = None  # No boxes for an image

        # Resize images to input shape
        images = torch.stack([image for image in images])

        self.batch_count += 1

        return paths, images, targets

"""    
from pycocotools.coco import COCO

class COCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.batch_count = 0

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        image = cv2.imread("/home/a01328525/fiftyone/coco-2017/train/data/"+path)
        #print(path)
        # Extract image as PyTorch tensor
        image = torch.Tensor(image).permute(2, 0, 1)
        
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in range(num_objs):
            label = coco_annotation[i]["category_id"]
            if label in [1, 3]:
                xmin = coco_annotation[i]["bbox"][0]
                ymin = coco_annotation[i]["bbox"][1]
                w = coco_annotation[i]["bbox"][2] if xmin+coco_annotation[i]["bbox"][2]<image.shape[-1] else image.shape[-1]-xmin-1
                h = coco_annotation[i]["bbox"][3] if ymin+coco_annotation[i]["bbox"][3]<image.shape[-2] else image.shape[-2]-ymin-1
                boxes.append([xmin, ymin, w, h])
                if label == 1: #person
                    labels.append(0)
                elif label == 3: #car
                    labels.append(1)
                #elif label == 17: #cat
                #    labels.append(2)
                #elif label == 18: #dog
                #    labels.append(3)
                #labels.append(coco_annotation[i]["category_id"])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.as_tensor(labels, dtype=torch.long)# torch.ones((num_objs,), dtype=torch.int64)
          
        #print(img.shape)
        #print(boxes)
        targets = torch.zeros((len(boxes), 6))
        boxes[:,0] = boxes[:,0] / image.shape[-1]
        boxes[:,1] = boxes[:,1] / image.shape[-2]
        boxes[:,2] = boxes[:,2] / image.shape[-1]
        boxes[:,3] = boxes[:,3] / image.shape[-2]
        
        targets[:, 2:6] = boxes
        targets[:, 1] = labels
        
        #print(labels)
        image = torchvision.transforms.Resize((224,224))(image)
        if image.shape == (1, 224, 224):
            image = torch.cat([image, image, image], 0)

        return  img_id, image, targets

    def __len__(self):
        return len(self.ids)
    def collate_fn(self, batch):
        paths, images, targets = list(zip(*batch))

        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]

        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i

        try:
            targets = torch.cat(targets, 0)
        except RuntimeError:
            targets = None  # No boxes for an image

        # Resize images to input shape
        images = torch.stack([image for image in images])

        self.batch_count += 1

        return paths, images, targets
"""
