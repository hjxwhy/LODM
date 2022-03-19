import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import exponnorm
import cv2
from PIL import Image
# import open3d as o3d
from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator
import time

# from utils.visualization_utils import colored_depthmap

np.random.seed(0)

K = np.array([[520.532000, 0.000000, 277.925800],
              [0.000000, 520.744400, 215.115000],
              [0.000000, 0.000000, 1.000000]])
# K = np.array([[5.574613109059876024e+02, 0.000000, 3.223667639238458946e+02],
#               [0.000000, 5.347049568266556889e+02, 2.469575328208151745e+02],
#               [0.000000, 0.000000, 1.000000]])
d = 2.

out = exponnorm.rvs(K=4.31, loc=0.44, scale=0.2, size=100)


def euler_angles_to_rotation_matrix(thetas, make_trans=False):
    if thetas.ndim == 1:
        thetas = thetas[np.newaxis, ...]
    batch_size, _ = thetas.shape
    cos = np.cos(thetas)
    sin = np.sin(thetas)
    one = np.ones_like(thetas)
    zero = np.zeros_like(thetas)
    R_x = (one[:, 0], zero[:, 0], zero[:, 0], zero[:, 0], cos[:, 0], -sin[:, 0], zero[:, 0], sin[:, 0], cos[:, 0])
    R_x = np.stack(R_x, -1).reshape((batch_size, 3, 3))

    R_y = (cos[:, 1], zero[:, 1], sin[:, 1], zero[:, 1], one[:, 1], zero[:, 1], -sin[:, 1], zero[:, 1], cos[:, 1])
    R_y = np.stack(R_y, -1).reshape((batch_size, 3, 3))

    R_z = (cos[:, 2], -sin[:, 2], zero[:, 2], sin[:, 2], cos[:, 2], zero[:, 2], zero[:, 2], zero[:, 2], one[:, 2])
    R_z = np.stack(R_z, -1).reshape((batch_size, 3, 3))

    R = np.matmul(R_z, np.matmul(R_y, R_x))
    # print(R)
    return R


def make_transform():
    t = np.append(np.random.normal(0.2, 0.3, 3), 1)
    # t = np.array([1, 1.5, 0, 1])
    # t[2] = 0
    # print('t', t)
    euler_angle = np.random.normal(0, 10. * np.pi / 180, size=3)
    # print("euler_angle", euler_angle / np.pi * 180)

    R = euler_angles_to_rotation_matrix(euler_angle)
    R = np.squeeze(R)
    T = np.concatenate((np.concatenate((R, [[0, 0, 0]]), axis=0), t[np.newaxis, :].transpose([1, 0])), axis=1)
    T_inv = np.concatenate((np.concatenate((R.transpose([1, 0]), [[0, 0, 0]]), axis=0),
                            np.append(-(R.transpose([1, 0]) @ t[np.newaxis, :].transpose([1, 0])[0:3]), 1)[np.newaxis,
                            :].transpose([1, 0])), axis=1)
    return T, T_inv


# print(out)

def reproject(key_point):
    key_points_position = key_point
    positin_3d = np.linalg.inv(K) @ key_points_position * d
    positin_3d_h = np.concatenate([positin_3d, [[1]]], axis=0)
    print(positin_3d_h)

    T, T_inv = make_transform()
    # print("T", T)

    new_frame_position = T @ positin_3d_h
    print("new_frame_position", new_frame_position)
    print("positin_3d", positin_3d)

    # line_dis = np.linalg.norm(np.squeeze(new_frame_position) - T[:, 3])
    line_dis = np.linalg.norm(np.squeeze(new_frame_position)[:3])
    # line_dis = np.linalg.norm(np.squeeze(positin_3d) - T[:, 3][:3])
    line_dis_perturb = line_dis + 1.
    print("line_dis", line_dis)

    # abs_pos = np.squeeze(new_frame_position) - T[:, 3]
    abs_pos = np.squeeze(new_frame_position)[:3]
    sin_alfa = abs_pos[2] / line_dis
    cos_alfa = np.linalg.norm(abs_pos[:2]) / line_dis
    cos_belta = abs_pos[0] / np.linalg.norm(abs_pos[:2])
    sin_belta = abs_pos[1] / np.linalg.norm(abs_pos[:2])
    print(sin_alfa)
    z_new_abs = line_dis_perturb * sin_alfa
    x_new_abs = line_dis_perturb * cos_alfa * cos_belta
    y_new_abs = line_dis_perturb * cos_alfa * sin_belta

    # new_coord = (np.array([x_new_abs, y_new_abs, z_new_abs, 0]) + np.squeeze(new_frame_position))[np.newaxis, :].transpose([1, 0])
    new_coord = np.array([x_new_abs, y_new_abs, z_new_abs, 1])[np.newaxis, :].transpose([1, 0])
    print("new_coord", new_coord)
    target = K @ ((T_inv @ new_coord)[0:3])
    print('source target', target)
    target = target / target[2]
    print("target", target)


def add_gausian_noise(key_points, depth, K):
    '''
    perturb the coordinates and depth separately with a zero mean Gaussian with standard deviation of 3 pixels
    and 0.45 meters for the depths, respectively.
    For the camera pose error, we take the perturbed coordinates and depth, project it into the camera frame ,
    and then apply a randomly generated SE(3) transformation to the camera pose with standard deviation 3 degree and 3cm,
    after which the points are re-projected back into the image plane.

    :param key_points: key points extract
    :param depth: groundtruth depth value of every key point
    :param K: camera intrinsic
    :return: perturb key points coordinates
    '''

    num_points, _ = key_points.shape
    # coord_noise = np.random.normal(0, 3., size=2 * num_points).reshape((num_points, 2))
    # print(coord_noise)
    # TODO make sure the noise mean and var
    depth_noise = np.random.normal(0, 0.1, num_points).reshape((num_points, 1)).transpose([1, 0])
    # key_points_perturb = key_points + coord_noise
    key_points_perturb = key_points
    # print(depth_noise.shape) + coord_noise
    homogeneous = np.ones([num_points, 1])
    key_points_perturb = np.concatenate([key_points_perturb, homogeneous], axis=1).transpose([1, 0])
    depth_perturb = depth + depth_noise
    # print(depth_perturb) + depth_noise
    key_points_perturb_3d = np.linalg.inv(K).dot(key_points_perturb) * depth_perturb
    key_points_perturb_3d = key_points_perturb_3d.transpose([1, 0])[..., np.newaxis]
    # print(key_points_perturb_3d.shape)
    # TODO change the rotation noise yaw large and other small, need to make sure the var
    xy_noise = np.random.normal(0, 0.026, 2 * num_points).reshape((num_points, 2)) / 180. * np.pi
    z_noise = np.random.normal(0, 0.026, num_points).reshape((num_points, 1)) / 180. * np.pi
    angle_noise = np.hstack([xy_noise, z_noise])
    R = euler_angles_to_rotation_matrix(angle_noise)
    t_noise = np.random.normal(0, 0.01, (num_points, 3))
    pose_noise = (np.matmul(R, key_points_perturb_3d).squeeze() + t_noise).transpose([1, 0])
    # pose_noise = key_points_perturb_3d.squeeze().transpose([1, 0])

    coords_noise = np.matmul(K, pose_noise)
    # print(coords_noise)
    # coords_noise = coords_noise / coords_noise[-1, :][np.newaxis, ...]
    return (coords_noise[:2, :] / coords_noise[-1:, :]).astype(np.int16).transpose([1, 0]), coords_noise[-1:, :].transpose(
        [1, 0])
    # print(coords_noise)
    # print(coords_noise[-1, :][np.newaxis, ...].shape)

    # r = Rotation.from_euler('xyz', angle_noise)
    # print(r.as_matrix())


# T, T_inv = make_transform()


def add_noise_on_ray(key_points, depth, K):
    '''
    add gaussian noise on ray as CodeMappimg(https://arxiv.org/pdf/2107.08994.pdf), the gaussian mean and var does not
    offer(I guess it is add gaussian, the author respond my email that he is very busy...)
    :param key_points: key points extract
    :param depth: groundtruth depth value of every key point
    :param K: camera intrinsic
    :return: perturb key points coordinates
    '''
    if key_points.ndim == 1:
        key_points = key_points[np.newaxis, ...]
    T, T_inv = make_transform()
    # print(T)
    num_points, _ = key_points.shape
    homogeneous = np.ones([num_points, 1])
    key_points_homogeneous = np.concatenate([key_points, homogeneous], axis=1).transpose([1, 0])
    ray_means = depth * 0.02
    ray_noise = []
    ray_means = np.squeeze(ray_means)
    for ray_mean in ray_means:
        ray_noise_tmp = np.random.normal(ray_mean, ray_mean / 3, 1)
        if ray_noise_tmp < 0.02:
            ray_noise_tmp = np.array([0.02])
        ray_noise.append(ray_noise_tmp)
    ray_noise = np.array(ray_noise).transpose([1, 0])
    # print(ray_noise.shape)

    # print("T", T, T_inv)
    # print(np.matmul(np.linalg.inv(K), key_points_homogeneous) * depth)
    virtual_kf_key_points = np.matmul(T, np.concatenate(
        [np.matmul(np.linalg.inv(K), key_points_homogeneous) * depth, homogeneous.transpose([1, 0])], axis=0))

    # ray_noise[0, 1] = ray_noise[0, 0]
    # print("ray_noise", ray_noise)
    # print(virtual_kf_key_points[:, 6])

    direc_vec = ray_noise * virtual_kf_key_points[:3, ...] / np.linalg.norm(virtual_kf_key_points[:3, ...], axis=0,
                                                                            keepdims=True)
    # zero = np.zeros([1, num_points])
    direc_vec = np.concatenate([direc_vec, np.zeros([1, num_points])], axis=0)
    # print("direc_vec", direc_vec)

    virtual_kf_key_points_perturb = virtual_kf_key_points + direc_vec

    # virtual_kf_key_points_perturb_2 = virtual_kf_key_points + np.insert(ray_noise * (
    #         virtual_kf_key_points[:3, ...] / np.linalg.norm(virtual_kf_key_points[:3, ...], axis=0, keepdims=True)),
    #                                                                     -1, 0, axis=0)

    source_points = np.matmul(K, np.matmul(T_inv, virtual_kf_key_points_perturb)[:3, ...])
    # source_points = np.matmul(K, curent_point3d)
    source_points = source_points / source_points[2, ...]
    # print('source_points', source_points)

    # return source_points[:2, ...].transpose([1, 0]), ray_noise
    return source_points[:2, ...].transpose([1, 0]), source_points[2, ...][np.newaxis, ...], \
           np.matmul(T_inv, virtual_kf_key_points_perturb)[:3, ...].transpose([1, 0])


class ScanNet:
    # ScanNet dataset depth scale is 1000.
    def __init__(self, root_path, save_sparse=False):
        self.root_path = root_path
        self.scene_files = os.listdir(root_path)
        self.color_paths = []
        self.depth_paths = []
        self.save_sparse = save_sparse
        for scene_file in self.scene_files:
            self.color_paths += glob.glob(self.root_path + '/' + scene_file + '/color/*.jpg')
            self.depth_paths += glob.glob(self.root_path + '/' + scene_file + '/depth/*.png')
        self.color_paths.sort()
        self.depth_paths.sort()

    def __len__(self):
        return len(self.color_paths)

    def __getitem__(self, item):
        color = cv2.imread(self.color_paths[item])
        depth = np.array(Image.open(self.depth_paths[item]), dtype=np.float16)

        # color = cv2.imread('/home/hjx/Documents/ScanNet/scene0035_00/color/1350.jpg')
        # depth = np.array(Image.open('/home/hjx/Documents/ScanNet/scene0035_00/depth/1350.png'), dtype=np.float16)

        # print(self.color_paths[item])
        # print(self.depth_paths[item])
        color = cv2.resize(color, (depth.shape[1], depth.shape[0]))
        intrinsics_path = self.root_path + '/' + self.color_paths[item].split('/')[
            -3] + '/intrinsic' + '/intrinsic_color.txt'
        K = []
        with open(intrinsics_path, 'r') as f:
            for line in f.readlines()[:-1]:
                line = line.strip().split()[:-1]
                K.append(line)
        K = np.array(K, dtype=np.float32)
        # if self.save_sparse:
        #     self.save_sparse_dir = self.root_path +'/' + self.color_paths[item].split('/')[-3] + '/sparse_depth'
        #     self.save_inter_dir = self.root_path +'/' + self.color_paths[item].split('/')[-3] + '/inter_depth'
        #     if not os.path.exists(self.save_sparse_dir):
        #         os.makedirs(self.save_sparse_dir, exist_ok=True)
        #         os.makedirs(self.save_inter_dir, exist_ok=True)

        return color, depth, K

    def save_sparse_and_inter_depth(self, sparse_depth, inter_depth, i):
        i_image_path = self.color_paths[i]
        sparse_depth = (sparse_depth * 1000).astype(np.uint16)
        inter_depth = (inter_depth * 1000).astype(np.uint16)
        name = i_image_path.split('/')[-1][:-4]
        save_sparse_dir = self.root_path + '/' + self.color_paths[i].split('/')[-3] + '/sparse_depth'
        save_inter_dir = self.root_path + '/' + self.color_paths[i].split('/')[-3] + '/inter_depth'
        if not os.path.exists(save_sparse_dir):
            os.makedirs(save_sparse_dir, exist_ok=True)
            os.makedirs(save_inter_dir, exist_ok=True)
        cv2.imwrite(save_sparse_dir + '/' + name + '.png', sparse_depth)
        cv2.imwrite(save_inter_dir + '/' + name + '.png', inter_depth)


class SUNRGBD:
    # unknown depth scale....
    def __init__(self, root_path, train=True):
        self.train = train
        self.root_path = root_path
        train_rgb_path = os.path.join(root_path, 'SUNRGBD/train_rgb.txt')
        train_depth_path = os.path.join(root_path, 'SUNRGBD/train_depth.txt')
        test_rgb_path = os.path.join(root_path, 'SUNRGBD/test_rgb.txt')
        test_depth_path = os.path.join(root_path, 'SUNRGBD/test_depth.txt')
        with open(train_rgb_path, 'r') as f:
            self.train_rgb_files = f.readlines()
        with open(train_depth_path, 'r') as f:
            self.train_depth_files = f.readlines()

    def __len__(self):
        if self.train:
            return len(self.train_rgb_files)
        else:
            # unrealize test dataset, it is easy
            return 0

    def __getitem__(self, item):
        image_dir = os.path.join(self.root_path, self.train_rgb_files[item].strip())
        depth_dir = os.path.join(self.root_path, self.train_depth_files[item].strip())

        color = np.array(cv2.imread(image_dir))
        depth = np.array(Image.open(depth_dir), dtype=np.float16)

        intrinsics_path = self.root_path + '/' + self.train_rgb_files[item].strip()[:82] + 'intrinsics.txt'
        K = []
        with open(intrinsics_path, 'r') as f:
            for line in f.readlines():
                line = line.strip().split()
                K.append(line)
        K = np.array(K, dtype=np.float32)
        # print(K)
        return color, depth, K


def get_keypoints(image, depth):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.equalizeHist(gray_image)
    gray_image = cv2.GaussianBlur(gray_image, (3, 3), 0.5)
    key_points = cv2.goodFeaturesToTrack(gray_image, 1500, 0.01, 10)
    # print(key_points.shape)
    sigma = 0.002
    while key_points.shape[0] < 100:
        key_points = cv2.goodFeaturesToTrack(gray_image, 1500, sigma, 5)
        if key_points.shape[0] < 100:
            sigma = sigma * 0.5
        else:
            print('less key point')
            break
    # if key_points
    # draw_keypoints(image, key_points)
    # cv2.imshow('sb', image)
    # cv2.waitKey(0)
    key_points = np.array(key_points, np.int16)
    key_points = key_points.reshape(-1, 2)

    key_points_mask1 = 10 < key_points[:, 0]
    key_points_mask2 = key_points[:, 0] < image.shape[1] - 10
    key_points_mask3 = 10 < key_points[:, 1]
    key_points_mask4 = key_points[:, 1] < image.shape[0] - 10
    key_points_mask = key_points_mask1 & key_points_mask2 & key_points_mask3 & key_points_mask4
    key_points = key_points[key_points_mask, ...]
    key_points_depth = depth[key_points[:, 1], key_points[:, 0]][np.newaxis, ...] / 1000.

    sparse_depth = np.zeros_like(depth)
    sparse_depth[key_points[:, 1], key_points[:, 0]] = depth[key_points[:, 1], key_points[:, 0]][
                                                           np.newaxis, ...] / 1000.
    sparse_depth_color = colored_depthmap(sparse_depth)
    sparse_depth_color = sparse_depth_color.astype('uint8')
    cv2.imshow('sparse', sparse_depth_color)
    return key_points, key_points_depth, sparse_depth


def draw_keypoints(image, keypoints, keypoint_color=(0, 0, 255)):
    for coord in keypoints:
        # print(coord)
        # coord = (coord[0], coord[1])
        cv2.circle(image, coord, 1, keypoint_color, thickness=2)


def meshgrid_coords(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid.

    Args:
      batch: batch size
      height: height of the grid
      width: width of the grid
      is_homogeneous: whether to return in homogeneous coordinates
    Returns:
      x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    # x_t = np.matmul(np.ones([height, 1]), np.expand_dims(np.linspace(-1, 1, width), axis=0))
    # y_t = np.matmul(np.expand_dims(np.linspace(-1, 1, height), axis=1), np.ones([1, width]))
    # x_t = (x_t + 1) * 0.5 * float(width - 1)
    # y_t = (y_t + 1) * 0.5 * float(height - 1)
    x_t = np.matmul(np.ones([height, 1]), np.expand_dims(np.linspace(0, width - 1, width), axis=0))
    y_t = np.matmul(np.expand_dims(np.linspace(0, height - 1, height), axis=1), np.ones([1, width]))

    if is_homogeneous:
        ones = np.ones_like(x_t)
        coords = np.stack([x_t, y_t, ones], axis=0)
    else:
        coords = np.stack([x_t, y_t], axis=0)
    coords = np.tile(np.expand_dims(coords, axis=0), [batch, 1, 1, 1])
    return coords


def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    """Transforms coordinates in the pixel frame to the camera frame.

    Args:
      depth: [batch, height, width]
      pixel_coords: homogeneous pixel coordinates [batch, 3, height, width]
      intrinsics: camera intrinsics [3, 3]
      is_homogeneous: return in homogeneous coordinates
    Returns:
      Coords in the camera frame [batch, 3 (4 if homogeneous), height, width]
    """
    batch, height, width = depth.shape
    depth = depth.reshape([batch, 1, -1])
    pixel_coords = pixel_coords.reshape([batch, 3, -1])
    intrinsics = np.repeat(np.expand_dims(intrinsics, 0), repeats=batch, axis=0)
    cam_coords = np.matmul(np.linalg.inv(intrinsics), pixel_coords) * depth
    if is_homogeneous:
        ones = np.ones([batch, 1, height * width])
        cam_coords = np.concatenate([cam_coords, ones], axis=1)
    cam_coords = np.reshape(cam_coords, [batch, -1, height, width])
    return cam_coords


def vis_3d(points, colors=np.array([])):
    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points)
    # pts.normals = o3d.utility.Vector3dVector(colors)
    if colors.shape[0] > 0:
        colors = colors / 255
        pts.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pts], 'points', width=1000, height=1000, point_show_normal=True)


def unproject(image, depth, intrinsics, batch=1):
    '''
    This function is unproject a image all coord, it may be more useful for visualization
    :param image:
    :param depth:
    :param intrinsics:
    :param batch:
    :return:
    '''
    height, width, _ = image.shape
    image = cv2.cvtColor(image,
                         cv2.COLOR_BGR2RGB)  # because open3d vis is rgb, if you don't convert, the color is different
    # long long ago, I use the batch, but now I use 1 batch, but I don't want to change the meshgrid_coords and pixel2cam
    # function, so I expand dimension
    depth = np.expand_dims(depth, axis=0)
    image = np.expand_dims(image, axis=0)
    img = image.reshape([batch, -1, 3])
    img = np.squeeze(img)
    pixel_coords = meshgrid_coords(1, height, width)
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=False)
    cam_coords = np.reshape(cam_coords, [batch, 3, -1])
    cam_coords = np.squeeze(cam_coords).transpose([1, 0])
    # vis_3d(cam_coords, img)
    return cam_coords, img


def diff_angle_normal(points_with_normal):
    coplanar = set()
    for key, values in points_with_normal.items():
        values = np.array(values)
        norms = values[:, :3]
        tris = values[:, 3]
        cross = np.cross(norms[0], norms)
        sin = np.linalg.norm(cross, axis=1)
        # print(sin < 0.707)
        # print(tris)
        sin_bool = sin < 0.65
        # print(np.all(sin_bool))
        if np.all(sin_bool):
            tri_id = tris[sin_bool]
            coplanar.update(tri_id)
    coplanar = list(coplanar)
    return np.array(coplanar, dtype=np.int16)


def normal(pt0, pt1, pt2):
    v1 = pt1 - pt0
    v2 = pt2 - pt0
    norm = np.cross(v1, v2)
    norm = norm / (np.linalg.norm(norm) + 1e-20)
    return norm


def delaunay_triangular(keypoints, cam_coords, image, debug=True, depth=None):
    if keypoints.shape[0] < 50:
        return None, False
    tri = Delaunay(keypoints)
    height, width, _ = image.shape
    if debug:
        for corners in keypoints[tri.simplices]:
            cv2.line(image, tuple(corners[0]), tuple(corners[1]), (255, 255, 255))
            cv2.line(image, tuple(corners[0]), tuple(corners[2]), (255, 255, 255))
            cv2.line(image, tuple(corners[1]), tuple(corners[2]), (255, 255, 255))
    cv2.imshow('ssss', image)
    cv2.waitKey(0)
    # pixel_coords = meshgrid_coords(1, height, width, is_homogeneous=False)
    # pixel_coords = np.reshape(pixel_coords, [-1, 2])
    t1 = time.time()

    points_normal_dict = {}
    for i, simplice in enumerate(tri.simplices):
        triangle_coord = cam_coords[simplice]

        norm = normal(triangle_coord[0], triangle_coord[1], triangle_coord[2])
        temp = np.append(norm, i).tolist()
        for idx in simplice:
            if points_normal_dict.get(idx) == None:
                points_normal_dict.setdefault(idx, [temp])
            else:
                points_normal_dict[idx].append(temp)
    coplanar = diff_angle_normal(points_normal_dict)
    if debug:
        for corners in keypoints[tri.simplices[coplanar]]:
            cv2.line(image, tuple(corners[0]), tuple(corners[1]), (0, 0, 255))
            cv2.line(image, tuple(corners[0]), tuple(corners[2]), (0, 0, 255))
            cv2.line(image, tuple(corners[1]), tuple(corners[2]), (0, 0, 255))

    rows, cols, _ = image.shape
    interpolator = LinearNDInterpolator(tri, depth[0, ...], fill_value=0)
    query_row_idx, query_col_idx = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    # because my keypoints coord is the format of (col, row), which is the format return from cv2 function,
    # the query_coord should be change to [col row]
    # query_coord = np.stack([query_row_idx.ravel(), query_col_idx.ravel()], axis=1)
    query_coord = np.stack([query_col_idx.ravel(), query_row_idx.ravel()], axis=1)
    # Z is inter depth
    Z = interpolator(query_coord).reshape([rows, cols])
    Z_mask = np.zeros_like(Z)
    Z_mask = cv2.fillPoly(Z_mask, keypoints[tri.simplices[coplanar]].astype(np.int32), 1.)
    Z = Z_mask * Z
    # cv2.imshow('Z', Z)

    color = colored_depthmap(Z)
    color = color.astype('uint8')
    cv2.imshow('color', color)
    # image = cv2.fillPoly(image, np.array([[[100, 200], [20, 50], [40, 60]], [(70, 70), (200, 100), (100, 150)]]),
    #                      (255, 255, 255))
    # cv2.imshow('ss', image)
    # print('query time ', time.time() - t1)
    return Z, True


def simulate_reprojection_error():
    all_sum = 0
    all_coord = []
    for i in range(20):
        x = 600 * np.random.random(1000)
        y = 500 * np.random.random(1000)
        d = 10 * np.random.random(1000)

        x_mask1 = 10 < x
        x_mask2 = x < 590
        x_mask = x_mask1 & x_mask2

        y_mask1 = 10 < y
        y_mask2 = x < 490
        y_mask = y_mask1 & y_mask2

        d_mask = d > 1.
        x_y_d_mask = x_mask & y_mask & d_mask
        # print('x_y_d_mask', np.sum(x_y_d_mask))

        x = x[x_y_d_mask]
        y = y[x_y_d_mask]
        d = d[x_y_d_mask]

        coord = np.stack([x, y], axis=1)
        # print("coord", coord)
        new_coord, ray_noise, cam_coord_perturb = add_noise_on_ray(coord, d, K)
        # print(new_coord)
        error = coord - new_coord
        sum = np.linalg.norm(error, axis=1, keepdims=True)
        # print('sum', sum.transpose([1, 0]))
        # sum = error[:, 1][..., np.newaxis]
        # sum = np.abs(error[:, 0][..., np.newaxis])
        # sum = np.sum(np.abs(coord - new_coord), axis=1, keepdims=True)
        error_mask_1 = np.abs(sum) > 0.4
        # print(error_mask_1)
        # print('depth', d[error_mask_1[..., 0], ...])
        # ray_noise = np.squeeze(ray_noise)
        # print('ray_noise', ray_noise[error_mask_1[..., 0]])
        # print('coord[error_mask_1[..., 0], ...]', coord[error_mask_1[..., 0], ...])
        # print('new_coord[error_mask_1[..., 0], ...]', new_coord[error_mask_1[..., 0], ...])

        error_mask = np.abs(sum) < 8
        # print('error_mask', np.sum(error_mask))
        coord = coord[error_mask[..., 0], ...]
        new_coord = new_coord[error_mask[..., 0], ...]
        #
        sum = sum[error_mask[..., 0], ...]

        all_sum = np.append(all_sum, sum)
        all_coord = np.append(all_coord, coord)

    plt.subplot(121)
    plt.subplot(121)
    plt.subplot(121)
    plt.xlabel('Reprojection Error (px)')
    plt.ylabel('Frequency')
    plt.title('exponential Gaussian distribution')
    out = exponnorm.rvs(K=4.31, loc=0.44, scale=0.2, size=20000)
    plt.hist(out, bins=150, density=True)
    # plt.ylim(0, 1800)

    plt.subplot(122)
    plt.xlabel('Reprojection Error (px)')
    plt.ylabel('Frequency')
    plt.title('add Gaussian noise on ray')
    plt.hist(all_sum, bins=150, density=True)
    # # plt.ylim(0, 500)
    # # plt.plot(coord[:, 0], coord[:, 1], 'ro', label = 'raw')
    # # plt.plot(new_coord[:, 0], new_coord[:, 1], 'b.', label= 'new')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # image_dir = "/home/hjx/Documents/tum/image/kf_00083_00015_00.png"
    # depth_dir = "/home/hjx/Documents/tum/sparse_depth/kf_00083_00015_00.png"
    #
    # image = cv2.imread(image_dir)
    # depth = np.array(Image.open(depth_dir), dtype=np.float32) / 1000.
    # height, width = np.where(depth > 0)
    # key_points = np.stack([width, height], axis=1)
    # delaunay_triangular(key_points, key_points, image)

    scannet = ScanNet('/home/hjx/Documents/ScanNet')
    sun = SUNRGBD('/home/hjx/Documents')
    for i in range(len(scannet)):
        image, depth, K = scannet[i]
        # unproject(image, depth, K)
        key_points, key_points_depth, sparse_depth = get_keypoints(image, depth)
        delaunay_triangular(key_points, key_points, image)
        target_image = image.copy()
        # draw_keypoints(image, key_points)
        # coord_perturb = add_gausian_noise(key_points, key_points_depth, K)
        coord_perturb, depth_perturb, cam_coord_perturb = add_noise_on_ray(key_points, key_points_depth, K)
        coord_perturb = coord_perturb.astype(np.int16)
        # third_image = image.copy()
        # draw_keypoints(third_image, coord_perturb, (255, 0, 0))
        # vis_3d(cam_coord_perturb)
        inter_depth, succes = delaunay_triangular(coord_perturb, cam_coord_perturb, target_image, debug=True,
                                                  depth=key_points_depth)
        # vis_3d(cam_coord_perturb)
        if not succes:
            inter_depth = sparse_depth
            print('do not have inter')
        # scannet.save_sparse_and_inter_depth(sparse_depth, inter_depth, i)
        # cv2.imshow('source image', image)
        cv2.imshow("target_image", target_image)
        # cv2.imshow("third_image", third_image)
        cv2.waitKey(0)

    # reproject(key_points_position)
    # print(T_inv @ new_coord)
    # print(line_dis)
    # print(new_frame_position)
