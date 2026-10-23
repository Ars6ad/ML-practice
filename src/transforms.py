import torchvision.transforms as transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transforms(split: str, config: dict, is_augmentation_exp: bool = False):
    """
    Returns image transformation pipeline based on split and experimental setting.
    
    Args:
        split (str): 'train', 'validation', or 'test'.
        config (dict): Configuration dictionary.
        is_augmentation_exp (bool): If True and split=='train', applies minority-focused
                                    conservative augmentations.
    """
    img_size = config.get("dataset", {}).get("image_size", 224)
    
    if split in ["validation", "test"]:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    
    if split == "train":
        if is_augmentation_exp:
            aug_cfg = config.get("augmentation", {})
            rot_deg = aug_cfg.get("rotation_degrees", 15)
            hflip_p = aug_cfg.get("horizontal_flip_prob", 0.5)
            cj = aug_cfg.get("color_jitter", {})
            crop_scale = tuple(aug_cfg.get("random_crop_scale", [0.85, 1.0]))
            
            return transforms.Compose([
                transforms.RandomResizedCrop(size=(img_size, img_size), scale=crop_scale),
                transforms.RandomHorizontalFlip(p=hflip_p),
                transforms.RandomRotation(degrees=rot_deg),
                transforms.ColorJitter(
                    brightness=cj.get("brightness", 0.15),
                    contrast=cj.get("contrast", 0.15),
                    saturation=cj.get("saturation", 0.15),
                    hue=cj.get("hue", 0.05),
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            # Baseline training: minimal standard preprocessing
            return transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    raise ValueError(f"Unknown split: {split}")
