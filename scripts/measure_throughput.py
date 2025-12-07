import numpy as np
import torch
import torch.nn as nn
from torchvision import tv_tensors
from torchvision.transforms import v2 as tv_transf
from torchvision.transforms.v2 import functional as tv_transf_F

from stenosis.throughput import run_benchmark


class TrainTransforms:
    """Example transformations."""

    def __init__(self, img_size=256, resize_target = True):

        self.img_size = img_size
        self.resize_target = resize_target

        scale = tv_transf.RandomAffine(degrees=0, scale=(0.95, 1.20))
        transl = tv_transf.RandomAffine(degrees=0, translate=(0.05, 0))
        rotate = tv_transf.RandomRotation(degrees=45)
        scale_transl_rot = tv_transf.RandomChoice((scale, transl, rotate))
        brightness, contrast, saturation, hue = 0.25, 0.25, 0.25, 0.01
        jitter = tv_transf.ColorJitter(brightness, contrast, saturation, hue)
        hflip = tv_transf.RandomHorizontalFlip()
        vflip = tv_transf.RandomVerticalFlip()

        self.transform = tv_transf.Compose((
            scale_transl_rot,
            jitter,
            hflip,
            vflip,
        ))

    def __call__(self, img, target):

        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).to(dtype=torch.uint8)
        target = torch.from_numpy(np.array(target)).unsqueeze(0).to(dtype=torch.uint8)

        img = tv_transf_F.resize(img, self.img_size)
        if self.resize_target:
            target = tv_transf_F.resize(target, 
                                        self.img_size, 
                                        interpolation=tv_transf.InterpolationMode.NEAREST_EXACT)

        img = tv_tensors.Image(img)
        target = tv_tensors.Mask(target)

        img, target = self.transform(img, target)

        img = img.data.float()/255
        target = target.data.to(dtype=torch.int64)[0]

        return img, target

def create_model(name):
    """Create a model given its name."""

    if name == "resnet18" or name == "resnet50":
        import segmentation_models_pytorch as smp
        model = smp.Unet(
            encoder_name=name, 
            encoder_weights=None,
            in_channels=3,
            classes=2,
        )

    elif name == "litemedsam":
        from torchtrainer.models.litemedsam.litemedsam import get_model
        class ModelWrapper(nn.Module):
            """Wrapper to adapt the litemedsam output to two channels."""

            def __init__(self):
                super().__init__()
                self.model = get_model()

            def forward(self, x):
                return self.model(x).repeat(1, 2, 1, 1)
        model = ModelWrapper()

    elif name == "unet_lw":
        from torchtrainer.models.unet_lw import get_model
        model = get_model(num_channels=3)

    else:
        raise ValueError(f"Model '{name}' not recognized.")
    
    return model

if __name__ == "__main__":

    num_imgs_dataset=2000

    # Run the command below in the terminal while the GPU is computing:
    # nvidia-smi dmon -s u -d 1
    # sm: percentage of streaming multiprocessor utilization (this is not the same as GPU 
    #     utilization provided by nvidia-smi)
    # mem: memory controller utilization. Higher values indicate a bottleneck in memory speed.
    run_benchmark(
        model=create_model("resnet50"),
        img_size=256, 
        batch_size=8,      
        num_workers=4,
        transforms=TrainTransforms(img_size=256),
        num_batches=200,
        num_imgs_dataset=2000,
        which="dtcmf",
        pin_memory=True,
        non_blocking=False,
        use_amp=True,
    )