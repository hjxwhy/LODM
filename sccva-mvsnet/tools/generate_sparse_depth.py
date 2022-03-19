##############################################
'''
This file and main2.cpp is modified from Indoor-SfMlearner
'''
##############################################
import numpy as np
import ctypes
import os
import cv2
from pyquaternion import Quaternion
import argparse
from models.datasets import add_noise


# so_path = os.path.join(os.path.dirname(__file__), 'libtest.so')
# lib = ctypes.cdll.LoadLibrary(so_path)
# c_float_p = ctypes.POINTER(ctypes.c_float)


class PixelSelector:
    def __init__(self, size=(640, 480)):
        self.so_path = os.path.join(os.path.dirname(__file__), 'libtest.so')
        print(self.so_path)
        self.size = size
        # self.so_path = so_path
        self.lib = ctypes.cdll.LoadLibrary(self.so_path)
        self.c_float_p = ctypes.POINTER(ctypes.c_float)

    def extract_points(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, self.size)
        data_p = image.astype(np.float32).ctypes.data_as(self.c_float_p)

        result = np.zeros((2500, 2)).astype(np.float32)
        result_p = result.ctypes.data_as(self.c_float_p)
        point_num = self.lib.main(self.size[0], self.size[1], data_p, 2000, 2500, result_p)
        points = result[:int(point_num), :]
        # points = points.astype(np.int16)
        # for (y, x) in points:
        #     image = cv2.circle(image, (x, y), 2, (255, 0, 255), -1)
        #
        # cv2.imshow("ii", image)
        # cv2.waitKey(0)
        return points

    def get_depth(self, points, depth):
        points = points.astype(np.int16)
        d = depth[points[:, 0], points[:, 1]]
        return d

    def generate_sparse_depth(self, image, depth):
        depth = cv2.resize(depth, self.size, interpolation=cv2.INTER_AREA)
        sparse_mask = np.zeros_like(depth)
        points = self.extract_points(image)
        points = points.astype(np.int16)
        sparse_mask[points[:, 0], points[:, 1]] = 1
        sparse_depth = sparse_mask * depth
        return sparse_depth

    def save_sparse_depth(self, file_path, sparse_depth):
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        cv2.imwrite(file_path, sparse_depth)
        print('save sparse depth {} to {}'.format(file_path.split('/')[-1], dir_path))


def save_dso_sparse_conf(window_path, target_dir, real_id, rate=10, skip=1, size=(1088, 1632)):
    wins = os.listdir(window_path)
    wins = [int(win.split('.')[0]) for win in wins]
    wins.sort()
    wins = [str(win) + '.txt' for win in wins]
    os.makedirs(os.path.join(target_dir, 'sparse_tuple'), exist_ok=True)
    dso_tuple = open(os.path.join(target_dir, 'tuples_dso_optimization_windows.txt'), 'w')
    target_image_name = os.listdir(os.path.join(target_dir, 'images'))
    for win in wins:
        if not win.endswith('txt'): continue
        name = str('{:06d}'.format(int(win.split('.')[0])))
        window_txt = os.path.join(window_path, win)
        with open(window_txt, 'r') as f:
            lines = f.readlines()
            tuple_depth = {}
            uv = []
            d = []
            c = []
            old_id = ''
            for line in lines:
                line = line.strip().split()
                if len(line) == 1:
                    if len(uv) > 0:
                        uv = np.stack(uv, axis=0)
                        d = np.array(d)
                        c = np.array(c)
                        tuple_depth.update({old_id: {'size': np.array(size), 'uv': uv, 'sparse_depth': d, 'conf': c}})
                    # old_id = str('{:06d}'.format(int(real_id[int(line[0])]) * rate + skip))
                    old_id = str('{:06d}'.format(int(real_id[int(line[0])])))
                    if old_id + '.png' not in target_image_name:
                        old_id = str('{:06d}'.format(int(old_id) + 10))
                    assert old_id + '.png' in target_image_name, "not target image {}".format(old_id)
                    uv = []
                    d = []
                    c = []
                else:
                    uv.append(np.array(line[:2], dtype=np.float32))
                    d.append(np.array(line[2], dtype=np.float32))
                    c.append(np.array(line[3], dtype=np.float32))
            uv = np.stack(uv, axis=0)
            d = np.array(d)
            c = np.array(c)
            tuple_depth.update({old_id: {'size': np.array(size), 'uv': uv, 'sparse_depth': d, 'conf': c}})
            tt = ['7'] + list(tuple_depth.keys()) + ['1'] + [name]
            tt = ' '.join(tt)
            dso_tuple.writelines(tt)
            dso_tuple.write('\n')
            np.save(os.path.join(target_dir, 'sparse_tuple', name + '.npy'), tuple_depth)
            # zz = np.load('/home/hjx/Documents/dji_data/zz.npy', allow_pickle=True).item()
            # print(zz)
            # print(list(zz.keys()))
            print(list(tuple_depth.keys()))


def read_sparse_tuple(sparse_tuple_index: int, current_tuple: tuple):
    # fname = os.path.join(self.scene_dir, 'sparse_tuple', f"{sparse_tuple_index:06d}.npy")
    fname = '/media/hjx/Jianxin_Huang1052/0314/results/windows/tuple_data/000000.npy'
    # assert os.path.exists(fname), "don't have {} sparse tuple".format(sparse_tuple_index)
    sparse_dict = np.load(fname, allow_pickle=True).item()
    sparse_names = list(sparse_dict.keys())
    sparse_depths = []
    confs = []
    for view_index in range(7):
        frame_index = sparse_names[view_index]
        assert frame_index in current_tuple, "dont match tuple name"
        size = sparse_dict[frame_index]["size"]
        uv = sparse_dict[frame_index]["uv"].astype(np.int16)
        d = sparse_dict[frame_index]["sparse_depth"]
        c = sparse_dict[frame_index]["conf"]
        sparse_depth = np.zeros(size)
        conf = np.zeros(size)
        sparse_depth[uv[:, 1], uv[:, 0]] = d
        conf[uv[:, 1], uv[:, 0]] = c
        # cv2.imshow("sd", sparse_depth.astype(np.uint8))
        # cv2.waitKey(000)
        sparse_depths.append(np.expand_dims(sparse_depth, 0))
        confs.append(np.expand_dims(conf, 0))
    sparse_depths = np.stack(sparse_depths, axis=0)
    confs = np.stack(confs, axis=0)
    return sparse_depths, confs


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data_dir', required=False, default='/home/hjx/Downloads/icl')
    # parser.add_argument('--save', action='store_false')
    # args = parser.parse_args()
    # seqs = os.listdir(args.data_dir)
    # seqs.sort()
    # ps = PixelSelector(size=(640, 480))
    # K = [[481.20, 0, 319.50],
    #      [0, -480.00, 239.50],
    #      [0, 0, 1]]
    #
    # for seq in seqs:
    #     print(f'generate sparse depth for {seq}')
    #     images_path = os.path.join(args.data_dir, seq, 'images')
    #     if not os.path.exists(images_path):
    #         continue
    #     image_files = os.listdir(images_path)
    #     # image_files.sort()
    #     for f in image_files:
    #         image_path = os.path.join(args.data_dir, seq, 'images', f)
    #         depth_path = os.path.join(args.data_dir, seq, 'depths', f)
    #         if depth_path.endswith('jpg'):
    #             depth_path = depth_path.replace('jpg', 'png')
    #         image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    #         depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    #         assert depth is not None, 'depth don\'t have the same file, it a invalid data!!!'
    #         if args.save:
    #             sparse_depth = ps.generate_sparse_depth(image, depth)
    #             sparse_depth = sparse_depth.astype(np.float32)
    #             sparse_depth /= 5000.
    #             image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #             image = np.expand_dims(image, -1)
    #             sparse_depth_noise, confidence = add_noise(sparse_depth, image.transpose([2, 0, 1]), 480, 640, K)
    #             # sparse_path = os.path.join(args.data_dir, seq, 'sparse', f)
    #             # if sparse_path.endswith('jpg'):
    #             #     sparse_path = sparse_path.replace('jpg', 'png')
    #             # ps.save_sparse_depth(sparse_path, sparse_depth)
    #             # print(np.min(confidence))
    #             print(sparse_depth[sparse_depth>0])
    #             cv2.imshow('s', sparse_depth)
    #             cv2.imshow('ss', sparse_depth_noise)
    #             cv2.waitKey(500)
            # else:
            #     points = ps.extract_points(image)
            #     points = points.astype(np.int16)
            #     print(points.shape)
            #
            #     image = cv2.resize(image, ps.size)
            #     for (y, x) in points:
            #         image = cv2.circle(image, (x, y), 2, (255, 0, 255), -1)
            #
            #     cv2.imshow("ii", image)
            #     cv2.waitKey(0)

###################################################################################
    # ps = PixelSelector(size=(640, 480))
    # K = [[481.20, 0, 319.50],
    #      [0, -480.00, 239.50],
    #      [0, 0, 1]]
    # win_path = '/home/hjx/Downloads/icl/traj0_frei_png/asdfgasfase/windows'
    # image_path = '/home/hjx/Downloads/icl/traj0_frei_png/images'
    # depth_path = '/home/hjx/Downloads/icl/traj0_frei_png/depths'
    # slide_wins = os.listdir(win_path)
    # # slide_wins.sort()
    # size = (480, 640)
    # target_dir = '/home/hjx/Downloads/icl/traj0_frei_png'
    # os.makedirs(os.path.join(target_dir, 'sparse_tuple'), exist_ok=True)
    # dso_tuple = open(os.path.join(target_dir, 'tuples_dso_optimization_windows.txt'), 'w')
    # for win in slide_wins:
    #     sw = open(os.path.join(win_path, win), 'r')
    #     lines = sw.readlines()
    #     index = []
    #     tuple_depth = {}
    #     name = str('{:06d}'.format(int(win.strip().split('.')[0])))
    #     for line in lines:
    #         i = str('{:06d}'.format(int(line.strip())+1))
    #         # index.append('{:06d}'.format(int(line.strip())))
    #         nn = i + ".png"
    #         image_p = os.path.join(image_path, nn)
    #         depth_p = os.path.join(depth_path, nn)
    #         image = cv2.imread(image_p, cv2.IMREAD_UNCHANGED)
    #         depth = cv2.imread(depth_p, cv2.IMREAD_UNCHANGED)
    #         assert depth is not None, 'depth don\'t have the same file, it a invalid data!!!'
    #         sparse_depth = ps.generate_sparse_depth(image, depth)
    #         sparse_depth = sparse_depth.astype(np.float32)
    #         sparse_depth /= 5000.
    #         image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #         image = np.expand_dims(image, -1)
    #         sparse_depth_noise, confidence = add_noise(sparse_depth, image.transpose([2, 0, 1]), 480, 640, K)
    #         h, w = np.where(sparse_depth_noise > 0)
    #         uv = np.stack([w, h], axis=1)
    #         d = sparse_depth_noise[h, w]
    #         c = confidence[h, w]
    #         print(i)
    #         tuple_depth.update({i: {'size': np.array(size), 'uv': uv, 'sparse_depth': d, 'conf': c}})
    #
    #     tt = ['7'] + list(tuple_depth.keys()) + ['1'] + [name]
    #     tt = ' '.join(tt)
    #     dso_tuple.writelines(tt)
    #     dso_tuple.write('\n')
    #     np.save(os.path.join(target_dir, 'sparse_tuple', name + '.npy'), tuple_depth)

##########################################################################################
    source_path = "/media/hjx/Sakura/UE4图片采集/eval-data"
    target_path = "/media/hjx/Sakura/UE4图片采集/eval-data"
    scenes = os.listdir(source_path)
    scenes.sort()
    for scene in scenes:
        window_path = os.path.join(source_path, scene, 'results', 'windows')
        target_dir = os.path.join(target_path, scene)
        results = open(os.path.join(source_path, scene, 'results', 'results.txt'), 'r')
        real_id = []
        for r in results.readlines():
            real_id.append(r.strip().split(' ')[0])
        with open(os.path.join(source_path, scene, 'results', 'aa.txt')) as f:
            line = f.readline().split(' ')
            rate = int(line[1])
            skip = int(line[-1])
            print(rate, skip)
        save_dso_sparse_conf(window_path, target_dir, real_id, rate, skip)
    # cur = tuple(['000001', '000011', '000021', '000031', '000041', '000051', '000061'])
    # sparse, conf = read_sparse_tuple(1, cur)
    # print(sparse.shape)
