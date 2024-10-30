import os

from monai.data import Dataset
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    LoadImage,
    LoadImaged,
    NormalizeIntensityd,
    ScaleIntensityRangePercentilesd,
    ToTensord,
    RandCropByLabelClassesd,
)


def build_dataset_to_pretrain(dataset_path, input_size, template_name, atlas_path, inference=False) -> Dataset:
    ratios = [0] + [1] * 66
    train_transforms = Compose(
        [
            LoadImaged(keys=["image"], image_only=True),
            EnsureChannelFirstd(keys=["image"]),
            ScaleIntensityRangePercentilesd(keys=["image", "template"], lower=1, upper=99, b_min=0, b_max=1),
            NormalizeIntensityd(keys=["image", "template"]),
            RandCropByLabelClassesd(
                keys=["image", "template", "sub_mask"],
                label_key="sub_mask",
                spatial_size=[input_size, input_size, input_size],
                ratios=ratios,
                num_classes=67,
                num_samples=2,
            ),
            ToTensord(keys=["image", "template"], track_meta=False),
        ]
    )

    datalist = []
    with open(dataset_path, "r") as f:
        scan_list = f.readlines()

    # load MNI152 template
    template_t1 = LoadImage(image_only=True, ensure_channel_first=True)(os.path.join(atlas_path, 'MNI152_T1_1mm_brain.nii.gz'))
    mask = LoadImage(image_only=True, ensure_channel_first=True)(os.path.join(atlas_path, 'HarvardOxford-sub-cort-maxprob-thr50-1mm.nii.gz'))
    
    for scan in scan_list:
        if 'roi' not in scan and 'seg' not in scan:
            template = template_t1
            datalist.append({'image': scan.strip(), 'template': template, 'sub_mask': mask})

    print('Training dataset: number of data: {}'.format(len(datalist)))
    
    dataset_train = Dataset(data=datalist, transform=train_transforms)
    
    return dataset_train