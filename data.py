from utils import *

# we need to crop the >3 dimensional images with a new function, because PIL only accepts 3-4 dimensions
class RandomNumpyCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    @staticmethod
    def get_params(sample, output_size):
        """Get parameters for a random crop"""
        h, w = sample["image"].shape[:2]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        # random height starting point of crop
        h_start = random.randint(0, h - th)
        # random width starting point of crop
        w_start = random.randint(0, w - tw)
        return h_start, w_start, h_start + th, w_start + tw

    def __call__(self, sample):
        h_start, w_start, h_end, w_end = self.get_params(sample, self.size)
        return {"image": sample["image"][h_start:h_end, w_start:w_end],
                "mask": sample["mask"][h_start:h_end, w_start:w_end]}

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NumpyResize(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        mask = sample["mask"]
        if mask.shape[-1]==1: mask = np.expand_dims(cv2.resize(sample["mask"], self.size), -1)
        else: mask = cv2.resize(sample["mask"], self.size)
        return {"image": cv2.resize(sample["image"], self.size), "mask": mask}

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomNumpyHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            return {"image": sample["image"][:, ::-1], "mask": sample["mask"][:, ::-1]}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomNumpyVerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            return {"image": sample["image"][::-1], "mask": sample["mask"][::-1]}
        return sample

    def __repr__(self):
        return self.__class__.__name__ + '()'

class RandomNumpyRotate(object):
    def __init__(self, augment_rotations=10):
        self.augment_rotations = augment_rotations

    def __call__(self, sample):
        img, mask = sample["image"], sample["mask"]
        angle = (2 * random.random() - 1.) * self.augment_rotations
        size = img.shape[:2]
        center = tuple(np.array(size) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, rot_mat, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, rot_mat, size, flags=cv2.INTER_LINEAR)
        return {"image": img, "mask": mask}

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomNumpyFlips(object):
    def __init__(self, augment_rotations=10):
        self.augment_rotations = augment_rotations

    def __call__(self, sample):
        img, mask = sample["image"], sample["mask"]
        angle = (2 * random.random() - 1.) * self.augment_rotations
        size = img.shape[:2]
        center = tuple(np.array(size) / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, rot_mat, size, flags=cv2.INTER_LINEAR)
        mask = cv2.warpAffine(mask, rot_mat, size, flags=cv2.INTER_LINEAR)
        return {"image": sample["image"][:, ::-1], "mask": sample["mask"][:, ::-1]}

    def __repr__(self):
        return self.__class__.__name__ + '()'


# class OwnToNormalizedTensor(object):
#     def __call__(self, sample):
#         img, mask = sample["image"], sample["mask"]
#         img = torch.from_numpy(np.flip(img.transpose((2, 0, 1)), axis=0).copy())
#         mask = torch.from_numpy(np.flip(np.expand_dims(mask, 0), axis=0).copy())
#         img = normalize(img)
#         return {"image": img, "mask": mask}


class OwnToNormalizedTensor(object):
    def __call__(self, sample):
        img, mask = sample["image"], sample["mask"]
        img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
        # pdb.set_trace()
        mask = torch.from_numpy(np.transpose(mask, (2, 0, 1)).copy())
        img = normalize(img)
        return {"image": img, "mask": mask}


# class OwnToTensor(object):
#     def __call__(self, sample):
#         img, mask = sample["image"], sample["mask"]
#         img = torch.from_numpy(np.flip(img.transpose((2, 0, 1)), axis=0).copy())
#         mask = torch.from_numpy(np.flip(np.expand_dims(mask, 0), axis=0).copy())
#         return {"image": img, "mask": mask}
#
#     def __repr__(self):
#         return self.__class__.__name__ + '()'

class OwnToTensor(object):
    def __call__(self, sample):
        img, mask = sample["image"], sample["mask"]
        img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
        mask = torch.from_numpy(np.transpose(mask, (2, 0, 1)).copy())
        return {"image": img, "mask": mask}

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_images_and_labels(ids, which_dataset):
    print("Loading Images and Masks")
    if os.path.exists("{}_12_band.bc".format(which_dataset)):
        samples = load_array("{}_12_band.bc".format(which_dataset))
    else:
        samples = []
        for id_ in ids:
            img = M(id_, dims=12)
            mask = generate_mask_for_image_and_class((img.shape[0], img.shape[1]), id_, 1)
            samples.append((img, mask))
        save_array("{}_12_band.bc".format(which_dataset), samples)
    return samples


class DatasetDSTL(Dataset):
    def __init__(self, ids, imgs, masks, classes, pick_random_idx=True, oversample=0., samples_per_epoch=1000,
                 which_dataset="train", transform=None):
        self.ids = ids
        self.which_dataset = which_dataset
        self.oversample = oversample
        self.pick_random_idx = pick_random_idx
        if self.which_dataset == "train":
            self.samples = []
            for j in range(20):
                masks_ = []
                for class_ in classes:
                    masks_.append(masks[10*j+class_])
                self.samples.append((imgs[j], np.transpose(np.array(masks_), (1,2,0))))
        elif self.which_dataset == "val":
            self.samples = []
            for j in range(20, 25):
                masks_ = []
                for class_ in classes:
                    masks_.append(masks[10 * j + class_])
                self.samples.append((imgs[j], np.transpose(np.array(masks_), (1, 2, 0))))
        else:
            self.samples = []
            for j in range(len(imgs)):
                masks_ = []
                for class_ in classes:
                    masks_.append(masks[10 * j + class_])
                self.samples.append((imgs[j], np.transpose(np.array(masks_), (1, 2, 0))))
        self.samples_per_epoch = samples_per_epoch
        self.transform = transform

    def __getitem__(self, index):
        if self.oversample:
            if random.random() < self.oversample:
                while True:
                    if self.pick_random_idx:
                        index = random.choice(np.arange(len(self.ids)))
                    img, mask = self.samples[index]
                    sample = {"image": img, "mask": mask}
                    if self.transform is not None:
                        sample = self.transform(sample)
                    # print(sample["mask"].sum().item())
                    #                     pdb.set_trace()
                    if sample["mask"].sum().item() > 0:
                        break
                return sample
        if self.pick_random_idx:
            index = random.choice(np.arange(len(self.ids)))
        img, mask = self.samples[index]
        sample = {"image": img, "mask": mask}
        #         print(sample["mask"].sum().item())
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.samples_per_epoch