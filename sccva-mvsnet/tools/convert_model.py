import numpy as np
import torch
import os
import pytorch_lightning as pl
from typing import Optional, NamedTuple, Dict, Tuple
from models.tandem import Tandem
from models.datasets import make_dataloader, AugmentationPipeline
from models.cva_mvsnet import CvaMVSNet, StageTensor
import cv2
from models.utils.helpers import to_device, tensor2numpy
from depth_map_filter import read_dso_result, save_scaled_pose

# pt = torch.jit.load('/home/hjx/tandem/tandem/exported/tandem_512x320/model.pt')
# print(pt.code)
class Aerial(pl.LightningModule):
    def __init__(self, hparams: dict):
        super().__init__()
        self.hparams = hparams
        self.cva_mvsnet = CvaMVSNet(
            depth_num=self.hparams["MODEL.DEPTH_NUM"],
            depth_interval_ratio=self.hparams["MODEL.DEPTH_INTERVAL_RATIO"],
            cost_volume_base_channels=self.hparams["MODEL.COST_VOLUME_BASE_CHANNELS"],
            feature_net_base_channels=self.hparams["MODEL.FEATURE_NET_BASE_CHANNELS"],
            view_aggregation=self.hparams.get("MODEL.VIEW_AGGREGATION", False),
            conv2d_normalization=self.hparams.get("MODEL.CONV2D_NORMALIZATION", "batchnorm"),
            conv2d_use_bn_skip=self.hparams.get("MODEL.CONV2D_USE_BN_SKIP", False),
            conv3d_normalization=self.hparams.get("MODEL.CONV3D_NORMALIZATION", "batchnorm"),
            use_sparse=self.hparams.get('TRAIN.USE_SPARSE', False),
            sc2dc=self.hparams.get('TRAIN.SC2DC', False),
        )
        self.augmentation_pipeline = AugmentationPipeline(hparams=self.hparams)

        self.train_batch_fun = None
        self.val_batch_fun = None
        self.test_dir = None
        self.use_sparse = self.hparams.get('TRAIN.USE_SPARSE', False)

    def forward(self, image: torch.Tensor,
                sparse: torch.Tensor,
                confidence: torch.Tensor,
                intrinsic_matrix: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                cam_to_world: torch.Tensor,
                depth_min: torch.Tensor,
                depth_max: torch.Tensor,
                depth_filter_discard_percentage: Optional[torch.Tensor] = None
                ):
        outs = self.cva_mvsnet(
            image=image,
            sparse=sparse if self.use_sparse else None,
            confidence=confidence if self.use_sparse else None,
            intrinsic_matrix=intrinsic_matrix,
            cam_to_world=cam_to_world,
            depth_min=depth_min,
            depth_max=depth_max,
            depth_filter_discard_percentage=depth_filter_discard_percentage
        )

        return outs[:-1]


def convert():
    model = Tandem.load_from_checkpoint("/home/hjx/epoch=035.ckpt").cuda()
    # model = model.cva_mvsnet
    model.eval()
    model.hparams['DATA.ROOT_DIR'] = "/home/hjx/Documents/dji_data/DJI_gen"
    model.hparams['TRAIN.NUM_WORKERS'] = 6
    model.hparams['TRAIN.BATCH_SIZE'] = 1
    train_loader, train_batch_fun = make_dataloader(model.hparams, split='train')
    with torch.no_grad():
        image = torch.randn((1, 7, 3, 1088, 1632)).cuda()
        ss = torch.randn((1, 7, 1, 1088, 1632)).cuda()
        cc = torch.randn((1, 7, 1, 1088, 1632)).cuda()
        intr = torch.randn((1, 3, 3)).cuda()
        intr2 = torch.randn((1, 3, 3)).cuda()
        intr3 = torch.randn((1, 3, 3)).cuda()
        c2w = torch.randn((1, 7, 4, 4)).cuda()
        dmi = torch.tensor(1).unsqueeze(0).cuda()
        dma = torch.tensor(10).unsqueeze(0).cuda()
        per = torch.tensor(0.1).unsqueeze(0).cuda()
        # out = model(image, ss, cc, (intr, intr2, intr3), c2w, dmi, dma, per)
        # print(out)
        # torch.jit.optimized_execution(True)
        # model_trace = torch.jit.trace(model, (image, ss, cc, (intr, intr2, intr3), c2w, dmi, dma, per))
        # torch.jit.save(model_trace, '/home/hjx/Downloads/mvs/model_trace.pt')
        # model_trace = torch.jit.load('/home/hjx/Downloads/mvs/model_trace.pt').cuda()
        for batch in train_loader:
            images = batch['image'].cuda()
            intrs = (batch['intrinsics']['stage3']['K'][:, 0, ...].cuda(), batch['intrinsics']['stage2']['K'][:, 0, ...].cuda(),
                     batch['intrinsics']['stage1']['K'][:, 0, ...].cuda())
            c2w = batch['cam_to_world'].cuda()
            d_min = batch['depth_min'].cuda()
            d_max = batch['depth_max'].cuda()
            per = torch.zeros_like(d_min)
            batch = to_device(batch, device='cuda')
            # output = model(images, ss, cc, intrs, c2w, d_min, d_max, per)
            output = model(batch)[:-1]
            depth = output[-1][0].cpu().numpy()
            print(np.max(depth))
            depth = (depth.transpose([1, 2, 0])/250 * 255).astype(np.uint8)
            cv2.imshow("depth", depth)
            cv2.waitKey(10)
            # print(depth.shape)

            # print(depth)
            # depth = output[]
        # print(pt.code)
    # out = model(image, image, image, (intr, intr, intr), c2w, dmi, dma)
    # model_trace = torch.jit.trace(model, (image, ss, cc, (intr, intr2, intr3), c2w, dmi, dma, per))
    # torch.jit.save(model_trace, '/home/hjx/Downloads/mvs/model_trace_sc.pt')
    # pt = torch.jit.load('/home/hjx/Downloads/mvs/model_trace_sc.pt')
    # print(pt.code)

def save_dso_result_front():
    source_path = "/media/hjx/Sakura/UE4图片采集/eval-data"
    target_path = "/media/hjx/Sakura/UE4图片采集/eval-data"
    scenes = os.listdir(source_path)
    scenes.sort()
    for scene in scenes:
        # window_path = os.path.join(source_path, scene, 'results', 'windows')
        target_dir = os.path.join(target_path, scene)
        # results = open(os.path.join(source_path, scene, 'results', 'results.txt'), 'r')
        # real_id = []
        # for r in results.readlines():
        #     real_id.append(r.strip().split(' ')[0])
        dso_path = os.path.join(source_path, scene, 'results', 'results.txt')
        with open(os.path.join(source_path, scene, 'results', 'aa.txt')) as f:
            line = f.readline().split(' ')
            rate = int(line[1])
            skip = int(line[-1])
            print(rate, skip)
            pose_c2w, pose_w2c = read_dso_result(dso_path, rate, skip)
            save_scaled_pose(pose_c2w, target_dir)


if __name__ == '__main__':
    # img = cv2.imread('/home/hjx/Downloads/mvs/depth.png', cv2.IMREAD_UNCHANGED)
    # cv2.imshow("depth", img)
    # cv2.waitKey(0)
    # convert()
    save_dso_result_front()