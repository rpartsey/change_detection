"""
Module with custom augmentations.
"""
import albumentations as albu


class StandardAugmentation:
    def __init__(self, p=0.6):
        self.p = p
        self.aug = self.__build_augmentator()

    def __call__(self, image):
        res = self.aug(image=image)
        return res["image"]

    def __build_augmentator(self):
        return albu.Compose([
            albu.OneOf([
                albu.VerticalFlip(p=0.5),
                albu.HorizontalFlip(p=0.5),
            ], p=0.5),
        ], p=self.p)