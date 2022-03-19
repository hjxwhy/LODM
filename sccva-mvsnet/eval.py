import argparse
import numpy as np
import torch
import random
import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from models import Tandem
from models.datasets import MVSDataset
from models.module import eval_errors
from models.utils.helpers import tensor2numpy, to_device
from models.utils import epoch_end_mean
import cv2
import pickle
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from PIL import Image
import os

def is_tensor(data):
    """Checks if data is a torch tensor."""
    return type(data) == torch.Tensor

def viz_inv_depth(inv_depth, normalizer=None, percentile=95,
                  colormap='magma', filter_zeros=True):
    """
    Converts an inverse depth map to a colormap for visualization.

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map to be converted
    normalizer : float
        Value for inverse depth map normalization
    percentile : float
        Percentile value for automatic normalization
    colormap : str
        Colormap to be used
    filter_zeros : bool
        If True, do not consider zero values during normalization

    Returns
    -------
    colormap : np.array [H,W,3]
        Colormap generated from the inverse depth map
    """
    # If a tensor is provided, convert to numpy
    if is_tensor(inv_depth):
        # Squeeze if depth channel exists
        if len(inv_depth.shape) == 3:
            inv_depth = inv_depth.squeeze(0)
        inv_depth = inv_depth.detach().cpu().numpy()
    cm = get_cmap(colormap)
    if normalizer is None:
        normalizer = np.percentile(
            inv_depth[inv_depth > 0] if filter_zeros else inv_depth, percentile)
    inv_depth /= (normalizer + 1e-6)
    return cm(np.clip(inv_depth, 0., 1.0))[:, :, :3]

def colored_depthmap2(depth, d_min=None, d_max=None, jet_color_map=plt.cm.plasma):
    if is_tensor(depth):
        # Squeeze if depth channel exists
        if len(depth.shape) == 3:
            depth = depth.squeeze(0)
        depth = depth.detach().cpu().numpy()
    if d_min is None:
        d_min = np.min(depth[depth>0])
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min * 1.07) / (d_max - d_min + 10e-8)
    # depth_relative = (depth - d_min) / d_max
    return 255 * jet_color_map(depth_relative)[:, :, :3]  # H, W, C

def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)  # ,optimize=False,compress_level=0

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str, help="Path to pytorch lightning ckpt.",
                    default='/media/hjx/3336-6530/新实验/ours/ss/epoch=049.ckpt')
parser.add_argument("--data_dir", help="Path to replica data.", type=str, default='/media/hjx/Sakura/UE4图片采集/eval-data')
parser.add_argument("--num_save_images", help="Number of images to be saved for viz.", type=int, default=10)

parser.add_argument("--seed", help="Seed.", type=int, default=1)
parser.add_argument("--device", help="Torch device.", type=str, choices=('cpu', 'cuda'), default='cuda')
parser.add_argument("--batch_size", help="Batch size.", type=int, default=1)
parser.add_argument("--num_workers", help="Number of workers.", type=int, default=4)

parser.add_argument("--tuples_ext", help="Tuples Extension.", type=str, default="dso_optimization_windows")
parser.add_argument("--pose_ext", help="Pose Extension.", type=str, default="dso", choices=("dso", "gt"))

parser.add_argument("--height", help="Image height.", type=int, default=480)
parser.add_argument("--width", help="Image width.", type=int, default=640)
parser.add_argument("--depth_min", help="Depth minimum.", type=float, default=50)
parser.add_argument("--depth_max", help="Depth maximum.", type=float, default=180)

parser.add_argument("--split", help="Split file", type=str, default="val")


def main(args: argparse.Namespace):
    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    model = Tandem.load_from_checkpoint(args.ckpt)  # type: Tandem
    model = model.to(device)
    model.eval()
    print(model.hparams)
    outputs_to_dict = model.cva_mvsnet.outputs_to_dict

    dataset = MVSDataset(
        root_dir=args.data_dir,
        split=args.split,
        pose_ext=args.pose_ext,
        tuples_ext=args.tuples_ext,
        ignore_pose_scale=args.pose_ext == "gt",
        height=args.height,
        width=args.width,
        tuples_default_flag=False,
        tuples_default_frame_num=-1,
        tuples_default_frame_dist=-1,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        dtype="float32",
        transform=None,
        use_sparse=model.hparams['TRAIN.USE_SPARSE']
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=6)
    errors = []

    if args.num_save_images > 0:
        image_save_ids = tuple((np.arange(args.num_save_images) * (len(dataset) // args.num_save_images)).tolist())
    else:
        image_save_ids = tuple()
    images = []

    start = time.time()
    num_processed = 0
    target = '/media/hjx/3336-6530/新实验/ours/ss'
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader)):
                batch = to_device(batch, device=device)
                outputs = outputs_to_dict(model(batch))
                errors.append(eval_errors(outputs=outputs, batch=batch))
                num_processed += args.batch_size

                # # print(outputs['stage3']['depth'])
                # d_min = torch.min(batch['depth']['stage3'][batch['depth']['stage3']>0]).cpu().numpy()
                # d_max = torch.max(batch['depth']['stage3']).cpu().numpy()
                #
                # vis_d_gt = colored_depthmap2(batch['depth']['stage3'], 100, 250)
                # # vis_d_gt = viz_inv_depth(batch['depth']['stage3'], colormap='plasma') * 255
                # save_image(vis_d_gt, os.path.join(target, str(batch_idx) +'_vis_gt.png'))
                #
                # d_min = torch.min(outputs['stage3']['depth'][outputs['stage3']['depth']>0]).cpu().numpy()
                # d_max = torch.max(outputs['stage3']['depth']).cpu().numpy()
                # vis_d_pred = colored_depthmap2(outputs['stage3']['depth'], 100, 250)
                # # vis_d_pred = viz_inv_depth(outputs['stage3']['depth'], colormap='plasma') * 255
                # save_image(vis_d_pred, os.path.join(target, str(batch_idx) +'_vis_pred.png'))
                #
                # error_map = batch['depth']['stage3'] - outputs['stage3']['depth']
                # # error_map = 1 / error_map
                # error_map[batch['depth']['stage3']<1] = 0
                # error_map_vis = viz_inv_depth(error_map) * 255
                # save_image(error_map_vis, os.path.join(target, str(batch_idx) +'_vis_error.png'))

                for i, idx in enumerate(range(batch_idx * args.batch_size, (batch_idx + 1) * args.batch_size)):
                    if idx in image_save_ids:
                        gt = tensor2numpy(batch['depth']['stage3'][i]).astype(np.float64) / args.depth_max
                        est = tensor2numpy(outputs['stage3']['depth'][i]).astype(np.float64) / args.depth_max
                        images.append(np.concatenate((gt, est), axis=0))

    except KeyboardInterrupt:
        pass

    elapsed = time.time() - start
    fps = num_processed / elapsed
    ms_per_frame = 1000.0 / fps

    errors = epoch_end_mean(errors)
    errors = tensor2numpy(errors)

    # Save errors
    with open(args.ckpt.rstrip('.ckpt') + '.pkl', 'wb') as fp:
        pickle.dump(obj=errors, file=fp)

    # Save images
    if len(images) > 0:
        image = np.concatenate(images, axis=1)
        if not np.all((image >= 0) & (image <= 1)):
            print(f"Image out of bounds: min/max/median = {np.amin(image)}/{np.amax(image)}/{np.median(image)}")
            image = np.clip(image, 0, 1)
        image = (image * float(np.iinfo(np.uint16).max)).astype(np.uint16)
        cv2.imwrite(args.ckpt.rstrip('.ckpt') + '.png', image)

    # Save output to file too
    with open(args.ckpt.rstrip('.ckpt') + '.txt', 'w') as fp:
        # Stage Table
        error_names = ('abs_rel', 'abs', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3')
        header = ' ' * 14 + ("{:>8s}   " * len(error_names)).format(*error_names)
        fmt_str = "{:>11s}:  " + "{:8.3f}   " * len(error_names)
        print(header, file=fp)
        for stage in errors:
            err = tuple(errors[stage][n].item() for n in error_names)
            print(fmt_str.format(stage.upper(), *err), file=fp)

        # Performance
        print(f"Performance: {fps:5.2f} FPS,  {int(ms_per_frame):5d} ms per image.", file=fp)

        # Eigen String
        print(
            f"Eigen et. al (delta <1.25, <1.25**2, <1.25**3): {errors['stage3']['d1'].item()} {errors['stage3']['d2'].item()} {errors['stage3']['d3'].item()}",
            file=fp)

        # Google Sheets String
        name = args.ckpt.rstrip(".ckpt")
        header = " " * (len(name) + 3)
        header += ("{:>8s}   " * (len(error_names) + 5)).format(
            *error_names, 'width', 'height', 'd_min', 'd_max', 'seed')[:-3]
        fmt_str = "{:>10s}   " + "{:8.4f}   " * len(error_names) + "{:8d}   {:8d}   {:8.4f}   {:8.4f}   {:8d}"
        print("\nPaste last line into Google Sheets", file=fp)
        print("" + header, file=fp)
        err = tuple(errors['stage3'][n].item() for n in error_names)
        print(fmt_str.format(name, *err, args.width, args.height, args.depth_min, args.depth_max, args.seed), file=fp)


if __name__ == "__main__":
    main(parser.parse_args())
