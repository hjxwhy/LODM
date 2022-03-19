import cv2
import os
import sys
import sqlite3
import time
import shutil
import numpy as np
import open3d as o3d
from pyquaternion import Quaternion
import plyfile
from scipy.spatial.transform import Rotation
from tools.camera import Camera, generate_depth_map
from tools.pose import Pose


def crop_intrinsics(intr, borders):
    """
    Crop camera intrinsics matrix

    Parameters
    ----------
    intrinsics : np.array [3,3]
        Original intrinsics matrix
    borders : tuple
        Borders used for cropping (left, top, right, bottom)
    Returns
    -------
    intrinsics : np.array [3,3]
        Cropped intrinsics matrix
    """
    intr = np.copy(intr)
    intr[0, 2] -= borders[0]
    intr[1, 2] -= borders[1]
    return intr


def scale_intrinsics(K, x_scale, y_scale):
    """Scale intrinsics given x_scale and y_scale factors"""
    K[0, 0] *= x_scale
    K[1, 1] *= y_scale
    K[0, 2] = (K[0, 2] + 0.5) * x_scale - 0.5
    K[1, 2] = (K[1, 2] + 0.5) * y_scale - 0.5
    return K


def recover_database_images_and_ids(database_path):
    # Connect to the database.
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    # Recover database images and ids.
    images = {}
    cameras = {}
    cursor.execute("SELECT name, image_id, camera_id FROM images;")
    for row in cursor:
        images[row[0]] = row[1]
        cameras[row[0]] = row[2]

    # Close the connection to the database.
    cursor.close()
    connection.close()

    return images, cameras


def generate_empty_recon(images, cameras, dso_pose, empty_model_path):
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    intr = np.array(
        [0.7680921 * 1632, 1.386796621 * 1088, 0.4854619883 * 1632, 0.4905782976 * 1088,
         -0.033569, 0.034502, -0.000472, 0.000415])
    # intr = np.array([1958.400000, 816.000000, 544.000000, 0.000000])

    with open(os.path.join(empty_model_path, 'cameras.txt'), 'w') as f:
        for image_name in images:
            image_id = images[image_name]
            camera_id = cameras[image_name]
            f.write('%d %s %d %d %s\n' % (
                camera_id,
                'OPENCV',
                1632,
                1088,
                ' '.join(map(str, intr))
            ))

    with open(os.path.join(empty_model_path, 'images.txt'), 'w') as f:
        for image_name in images:
            image_id = images[image_name]
            camera_id = cameras[image_name]

            f.write('%d %s %s %d %s\n\n' % (
                image_id,
                ' '.join(map(str, dso_pose[image_name][3:])),
                ' '.join(map(str, dso_pose[image_name][:3])),
                camera_id,
                image_name
            ))

    with open(os.path.join(empty_model_path, 'points3D.txt'), 'w') as f:
        pass


def read_point3d(point3d_path, poses):
    #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    points_f = open(point3d_path, 'r')
    points_lines = points_f.readlines()
    points = []
    colors = []

    R = np.array([[0.99810882, 0.04330158, -0.04363205],
                  [0.0442426, -0.99880349, 0.02083694],
                  [-0.04267757, -0.02272793, -0.99883035]])
    t = np.array([-15.79813344, 137.74351923, 3.10881115])
    scale = 48.30724280273795

    for line in points_lines:
        line = line.strip().split()
        if line[0] == '#':
            continue
        point = np.array(line[1:4], dtype=np.float32)
        colors.append(np.array(line[4:7], dtype=np.int8))
        points.append(point)
    points = np.vstack(points)

    # points = (scale * np.matmul(R, points.T) + t[:, np.newaxis]).T

    points = np.vstack([points, poses])
    colors = np.vstack(colors) / 255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    points_f.close()


def ply2npz(ply_path, save=False):
    assert ply_path.endswith('ply'), 'please give ply file'
    plydata = plyfile.PlyData.read(ply_path)
    pc = np.array(plydata['vertex'].data.tolist())
    if save:
        np.savez(ply_path.replace('ply', 'npz'), verts=pc[:, 0:3],
                 rgb=pc[:, 6:9])
    return pc[:, 0:3]


def render_depth(poses, intrinsic, height, width, new_size=None, verts=None, pc_path='', target_path='', max_depth=200):
    assert height > 0 and width > 0, 'please give the image size'
    if verts is None and pc_path != '':
        points = np.load(pc_path)
        verts = points['verts']  # [3, N]
        rgb = points['rgb'].T  # [3, N]
    verts = verts.T

    if target_path == '':
        target_path = os.path.dirname(pc_path)
    os.makedirs(os.path.join(target_path, 'depths'), exist_ok=True)
    scale_make = False
    scale = 1

    for name, T in poses.items():
        # R = T[:3, :3]
        # t = T[:3, 3:4]

        R = Quaternion(T[3:]).transformation_matrix[:3, :3]
        t = np.array(T[:3])[:, np.newaxis]

        pc_c = np.matmul(R, verts) + t  # change the point cloud to camera coordinate
        pixel_coords = np.matmul(intrinsic, pc_c)
        d = pixel_coords[2:3, :]
        uv = pixel_coords[0:2, :] / d
        mean_d = np.mean(d)
        mask = np.logical_and.reduce(
            (uv[0, :] > 0, uv[0, :] < width,
             uv[1, :] > 0, uv[1, :] < height,
             d[0, :] < (mean_d + 0.3 * mean_d), d[0, :] > 0.5 * mean_d))
        depth = np.zeros((height, width))
        image = np.zeros((height, width, 3))

        # image[uv[1, mask].astype(np.int16), uv[0, mask].astype(np.int16), :] = rgb[:, mask].T
        if not scale_make:
            scale = max_depth / np.max(d[0, mask])
            scale_make = True
        depth[uv[1, mask].astype(np.int16), uv[0, mask].astype(np.int16)] = d[0, mask] * scale

        kernel = np.ones((5, 5), dtype=np.uint8)
        cv2.imshow('image', image.astype(np.uint8))
        depth = cv2.dilate(depth, kernel, 5)  # 1:迭代次数，也就是执行几次膨胀操作
        cv2.imshow('depth2', depth.astype(np.uint8))
        cv2.waitKey(0)

        # if (width, height) != new_size:
        #     depth = cv2.resize(depth, new_size, interpolation=cv2.INTER_NEAREST)
        #     os.makedirs(os.path.join(target_path, str(new_size)), exist_ok=True)
        #     image = cv2.imread(os.path.join(target_path, 'images', name + '.jpg'), cv2.IMREAD_UNCHANGED)
        #     image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        #     cv2.imwrite(os.path.join(target_path, str(new_size), name + '.jpg'), image)

        depth = (depth * 255).astype(np.uint16)  # larger for save
        # if len(name) < 6:
        #     name = str('{:06d}'.format(int(name)))
        cv2.imwrite(os.path.join(target_path, 'depths', name + '.png'), depth)
        print('save depth {}'.format(name))
    #
    # cv2.namedWindow('11', cv2.WINDOW_FREERATIO)
    # cv2.imshow('11', image.astype(np.uint8))
    # cv2.imshow('11', depth)
    # cv2.waitKey(30)

    ########################################################
    # this scale is not the same as the depth scale, it is multiply the scaled depth for save as uint16
    ff = open(os.path.join(target_path, 'depths', 'scale.txt'), 'w')
    ff.write(str(1 / 255))
    ff.close()
    ########################################################

    # save_dji_subsequence_poses(poses, target_path)


def write_intr_and_pose2database(database_path, intr, poses):
    # colmap generate database, intr for camera, all camera have the same intr, pose for all camera
    # this pose will be optimized to a scale free
    # now set the intr to OpenCV model, it may need to chage, but now have no time, 1D array
    # pose is dso estimate pose "world to camera", it is every important, the format is colmap need, key is the image name
    # value is x y z q_w q_x q_y q_z

    IS_PYTHON3 = sys.version_info[0] >= 3

    def array_to_blob(array):
        if IS_PYTHON3:
            return array.tostring()
        else:
            return np.getbuffer(array)

    db = sqlite3.connect(database_path)
    rows = db.execute("SELECT * FROM cameras")
    for row in rows:
        camera_id, model, width, height, params, prior = row
        # params = blob_to_array(params, np.float64)
        # params2 = np.array(
        #     [0.7680921 * 1632, 1.386796621 * 1088, 0.4854619883 * 1632, 0.4905782976 * 1088,
        #      -0.033569, 0.034502, -0.000472, 0.000415])
        blob = array_to_blob(intr)
        db.execute("UPDATE cameras set params=? where camera_id=?", (blob, camera_id))
        db.execute("UPDATE cameras set model=? where camera_id=?", (4, camera_id))

    # poses = read_dso_pose('/media/hjx/3336-6530/组会/results/results.txt')
    rows = db.execute("SELECT * FROM images")
    for row in rows:
        image_id, name, camera_id, _, _, _, _, _, _, _, = row
        name = name.split('.')[0]
        pose = poses[name]
        db.execute(
            "UPDATE images set prior_qw=?  where image_id=?",
            (float(pose[3]), image_id))
        db.execute(
            "UPDATE images set prior_qx=?  where image_id=?",
            (float(pose[4]), image_id))
        db.execute(
            "UPDATE images set prior_qy=?  where image_id=?",
            (float(pose[5]), image_id))
        db.execute(
            "UPDATE images set prior_qz=?  where image_id=?",
            (float(pose[6]), image_id))
        db.execute(
            "UPDATE images set prior_tx=?  where image_id=?",
            (float(pose[0]), image_id))
        db.execute(
            "UPDATE images set prior_ty=?  where image_id=?",
            (float(pose[1]), image_id))
        db.execute(
            "UPDATE images set prior_tz=?  where image_id=?",
            (float(pose[2]), image_id))

    db.commit()
    db.close()


def read_array(path):
    '''
    Read depth map bin file
    Args:
        path:

    Returns: depth array [H, W, C]

    '''
    with open(path, "rb") as fid:
        width, height, channels = np.genfromtxt(fid, delimiter="&", max_rows=1,
                                                usecols=(0, 1, 2), dtype=int)
        fid.seek(0)
        num_delimiter = 0
        byte = fid.read(1)
        while True:
            if byte == b"&":
                num_delimiter += 1
                if num_delimiter >= 3:
                    break
            byte = fid.read(1)
        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channels), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def read_camera(camera_path):
    '''
    Read colmap camera.txt file
    TODO(Jianxin) change to general, now fix
    Args:
        camera_path:

    Returns:

    '''
    f = open(camera_path, 'r')
    line = f.readlines()[3].strip().split()
    width = int(line[2])
    height = int(line[3])
    intrinsic = np.array([[float(line[4]), 0, float(line[6])], [0, float(line[5]), float(line[7])], [0, 0, 1]],
                         dtype=np.float32)
    f.close()
    return width, height, intrinsic


def read_poses(colmap_images_txt):
    #  colmap pose is world to camera
    with open(colmap_images_txt, 'r') as f:
        lines = f.readlines()
        # Image list with two lines of data per image:
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        #   POINTS2D[] as (X, Y, POINT3D_ID)
        camera_pose = {}  # word to camera, key is the image name
        pose_w2c = {}
        pose_c2w = {}
        for i in range(4, len(lines), 2):
            line = lines[i].strip().split()
            name = line[-1].split('.')[0]

            # q = np.array(line[1:5], dtype=np.float32)
            q = Quaternion(np.array(line[1:5], dtype=np.float32))
            t = np.array(line[5:8], dtype=np.float32)
            pose_w2c[name] = t.tolist() + np.array(line[1:5], dtype=np.float32).tolist()

            T = q.transformation_matrix
            # T[:3, 3] = t
            T[:3, 3] = -np.matmul(T[:3, :3].T, t)
            T[:3, :3] = T[:3, :3].T
            q = Quaternion(matrix=T[:3, :3]).q
            t = T[:3, 3]
            pose_c2w[name] = t.tolist() + q.tolist()  # TX, TY, TZ, QW, QX, QY, QZ,
        new_keys = sorted(pose_c2w.keys())
        new_pose_c2w = {}
        new_pose_w2c = {}
        for key in new_keys:
            new_pose_c2w[key] = pose_c2w[key]
            new_pose_w2c[key] = pose_w2c[key]
        f.close()
    return new_pose_c2w, new_pose_w2c


def visualize_depth_map(depth_map, min_depth_percentile=5, max_depth_percentile=95):
    min_depth, max_depth = np.percentile(
        depth_map, [min_depth_percentile, max_depth_percentile])
    depth_map[depth_map < min_depth] = min_depth
    depth_map[depth_map > max_depth] = max_depth

    import pylab as plt

    # Visualize the depth map.
    plt.figure()
    plt.imshow(depth_map)
    plt.title("depth map")

    # # Visualize the normal map.
    # plt.figure()
    # plt.imshow(normal_map)
    # plt.title("normal map")

    plt.show()


def vis_points(points3D, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3D)
    if colors is not None:
        colors = colors / 256
        pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])
    # points_f.close()


def dilate_depth_map(depth_map):
    kernel = np.ones((3, 3), dtype=np.uint8)
    depth_map_dilate = cv2.dilate(depth_map, kernel, 2)  # 1:迭代次数，也就是执行几次膨胀操作
    # just change the source depth_map zero value area, other keep the same source
    depth_map[depth_map == 0] = depth_map_dilate[depth_map == 0]
    return depth_map


def read_image_depth(img_path, depth_path):
    img = cv2.imread(img_path)
    depth_map = read_array(depth_path)
    depth_map = cv2.resize(depth_map, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return img, depth_map


def read_pose(colmap_pose_w2c, img_path, img_next_path):
    pose_i = Pose(wxyz=np.array(colmap_pose_w2c[img_path.split('/')[-1].split('.')[0]][3:]),
                  tvec=np.array(colmap_pose_w2c[img_path.split('/')[-1].split('.')[0]][:3]))
    pose_i_next = Pose(wxyz=np.array(colmap_pose_w2c[img_next_path.split('/')[-1].split('.')[0]][3:]),
                       tvec=np.array(colmap_pose_w2c[img_next_path.split('/')[-1].split('.')[0]][:3]))
    return pose_i, pose_i_next


def generate_cam(intrinsic, pose_i, pose_next, align_pose=None):
    cam_i = Camera(K=intrinsic, p_cw=pose_i, align_pose=align_pose)
    cam_next = Camera(K=intrinsic, p_cw=pose_next, align_pose=align_pose)
    return cam_i, cam_next


def generate_scaled_depth(depth_map, img, img_next, cam_i, cam_next, img_height, img_width):
    pix_coords = cam_i.reconstruct(depth_map)
    warp_image = cam_next.sample(pix_coords, img_next, img_height, img_width)
    diff = np.mean(np.abs(img - warp_image), axis=-1)

    # have another check is warp_image > 0, because it has none overlap area, it just for visualize, at training, it
    # will center crop
    mask = np.logical_and(diff > 10, warp_image[:, :, 0] > 0)
    depth_map[mask] = 0
    depth_map = dilate_depth_map(np.copy(depth_map))

    # as my test, this can move some outlier without using point cloud remove outlier, it is time-consuming
    # just remove ground outlier
    max_depth = np.percentile(depth_map, 95)
    depth_map[depth_map > max_depth] = 0

    # the depth_map still scale free, reconstruct to scale aware point and project
    Xw = cam_i.reconstruct(depth_map)
    depth_map_scale = generate_depth_map(cam_i, Xw, (img_height, img_width))
    depth_map_scale = dilate_depth_map(np.copy(depth_map_scale))
    return depth_map_scale, Xw


def scale_pose(colmap_pose_c2w, align_pose):
    scale_poses = {}  # world to camera
    R = align_pose.rotation_matrix
    t = align_pose.tvec
    scale = align_pose.scale
    for key, item in colmap_pose_c2w.items():
        #######################################
        p_t = np.array(item[:3])
        p_q = np.array(item[3:])  # (w x y z)
        q = Quaternion(p_q)
        # scale colmap pose
        R_s = np.matmul(R, q.transformation_matrix[:3, :3])
        t_s = scale * np.matmul(R, p_t) + t

        scale_poses[key] = t_s.tolist() + Quaternion(matrix=R_s).q.tolist()
    return scale_poses


def save_scaled_pose(scaled_pose_c2w, path):
    # dji don't have gt, so dso and gt is the same from colmap, the poses have aligned with scale aware dso pose
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    dir_name = path
    # pose_gt_file = open(os.path.join(dir_name, 'poses_gt_front.txt'), 'w')
    pose_dso_file = open(os.path.join(dir_name, 'poses_dso.txt'), 'w')
    for name, pose in scaled_pose_c2w.items():
        name = str('{:06d}'.format(int(name))) + ' '
        T = Pose(wxyz=np.array(pose[3:]), tvec=np.array(pose[:3])).matrix
        pose_dso_file.write(name)
        pose_dso_file.writelines(' '.join(map(str, T.flatten())))
        pose_dso_file.write('\n')
        # pose_gt_file.write(name)
        # pose_gt_file.writelines(' '.join(map(str, T.flatten())))
        # pose_gt_file.write('\n')
    pose_dso_file.close()
    # pose_gt_file.close()


def generate_tuples(pose_dso_file, scale=1):
    if not os.path.isfile(pose_dso_file):
        dir_name = pose_dso_file
        pose_dso_file = os.path.join(pose_dso_file, 'poses_dso.txt')
    else:
        dir_name = os.path.dirname(pose_dso_file)
    pose_dso = open(pose_dso_file, 'r')
    pose_lines = pose_dso.readlines()

    w8 = open(os.path.join(dir_name, 'tuples_dso_optimization_windows.txt'), 'w')  # save for 8 windows
    last3 = open(os.path.join(dir_name, 'tuples_dso_optimization_windows_last3.txt'), 'w')  # save last 3 windows
    for i in range(len(pose_lines) - 8):
        w8.write('8 ')
        # w8.write(' ')
        last3.write('3 ')
        # last3.write(' ')
        for j in range(8):
            line = pose_lines[i + j]
            if j > 4:
                last3.write(line.split()[0] + ' ')
                # last3.write(' ')
            w8.write(line.split()[0] + ' ')
            # w8.write(' ')
        w8.write(str(scale))
        w8.write('\n')
        last3.write(str(scale))
        last3.write('\n')
    w8.close()
    last3.close()


def save_pose_as_tum(poses, target_path, skip_rate=10, bias=1):
    """ save pose file and gps file to tum format, and can be used evo for calculate scale
    tum format: timestamp x y z q_x q_y q_z q_w
    """
    f = open(target_path, 'w')
    for key, item in poses.items():
        name = (int(key) - bias) // skip_rate  # this number is different for diff dataset
        f.write(str(name) + ' ')
        f.writelines(' '.join(map(str, item[:3])))
        f.write(' ')
        q = item[4:] + [item[3]]  # change w x y z to x y z w
        f.writelines(' '.join(map(str, q)))
        f.write('\n')
    f.close()


def read_dso_result(dso_result_path, skip_rate=10, bias=1):
    # dso pose format: id x y z q_w q_x q_y q_z
    # the id to image name have to multi scale and bias

    dso = open(dso_result_path, 'r')
    dso_lines = dso.readlines()
    pose_c2w = {}
    pose_w2c = {}
    for line in dso_lines:
        line = line.strip().split()
        # idx = '{:06d}'.format(int(line[0]) * skip_rate + bias)
        idx = '{:06d}'.format(int(line[0]))
        name = str(idx)  # + '.jpg'
        xyz = np.array(line[1:4], dtype=np.float32)
        wxyz = np.array([line[-1]] + line[4:-1], dtype=np.float32)

        r = Quaternion(wxyz).rotation_matrix
        r_T = r.T
        t = -np.matmul(r_T, xyz)
        q = Quaternion(matrix=r_T).q
        pose_c2w[name] = xyz.tolist() + wxyz.tolist()  # the save q format is (w x y z)
        pose_w2c[name] = t.tolist() + q.tolist()  # the save q format is (w x y z)
    return pose_c2w, pose_w2c


def save_colmap_dso_2tum(colmap_image_path, dso_result_path, save_path, skip_rate=10, bias=1):
    colmap_pose_c2w, colmap_pose_w2c = read_poses(colmap_image_path)
    dso_pose_c2w, dso_pose_w2c = read_dso_result(dso_result_path, skip_rate=skip_rate, bias=bias)
    save_pose_as_tum(colmap_pose_c2w, os.path.join(save_path, 'col.txt'), skip_rate=skip_rate, bias=bias)
    save_pose_as_tum(dso_pose_c2w, os.path.join(save_path, 'dso.txt'), skip_rate=skip_rate, bias=bias)
    print('save all')


def process_depth(image_path, camera_path, colmap_image_path, scene_path, align_pose):
    images = os.listdir(image_path)
    images.sort()
    img_width, img_height, intrinsic = read_camera(camera_path)

    colmap_pose_c2w, colmap_pose_w2c = read_poses(colmap_image_path)

    all_point = []
    n = 0
    colmap_pose_c2w_scaled = scale_pose(colmap_pose_c2w, align_pose)
    ## save scaled colmap pose
    save_scaled_pose(colmap_pose_c2w_scaled, scene_path)
    generate_tuples(scene_path)

    os.makedirs(os.path.join(scene_path, 'depths'), exist_ok=True)

    for i in range(0, len(images)):
        if i == len(images) - 1:
            image = os.path.join(image_path, images[i])
            image_next = os.path.join(image_path, images[i - 1])
            depth = os.path.join(depth_path, '{}.geometric.bin'.format(images[i]))
            depth_next = os.path.join(depth_path, '{}.geometric.bin'.format(images[i - 1]))
        else:
            image = os.path.join(image_path, images[i])
            image_next = os.path.join(image_path, images[i + 1])
            depth = os.path.join(depth_path, '{}.geometric.bin'.format(images[i]))
            depth_next = os.path.join(depth_path, '{}.geometric.bin'.format(images[i + 1]))

        img, depth_map = read_image_depth(image, depth)
        img_next, depth_map_next = read_image_depth(image_next, depth_next)

        pose_i, pose_next = read_pose(colmap_pose_w2c, image, image_next)

        cam_i, cam_next = generate_cam(intrinsic, pose_i, pose_next, align_pose)

        depth_map_scale, Xw = generate_scaled_depth(depth_map, img, img_next, cam_i, cam_next, img_height, img_width)
        print(np.max(depth_map_scale))

        # save depth
        depth_map_scale = (depth_map_scale * 255).astype(np.uint16)  # larger for save
        name = str('{:06d}'.format(int(images[i].split('.')[0])))
        cv2.imwrite(os.path.join(scene_path, 'depths', name + '.png'), depth_map_scale)
        print('save depth {}'.format(name))
        # all_point.append(Xw)
        # n += 1
        # if n == 10:
        #     break
    # all_point = np.vstack(all_point)
    # print(all_point.shape)
    # vis_points(all_point)
    ########################################################
    # this scale is not the same as the depth scale, it is multiply the scaled depth for save as uint16
    ff = open(os.path.join(scene_path, 'depths', 'scale.txt'), 'w')
    ff.write(str(1 / 255))
    ff.close()
    ########################################################


def fix_database(database_path, dso_result_path, intr, skip_rate=10, bias=1):
    pose_c2w, pose_w2c = read_dso_result(dso_result_path, skip_rate=skip_rate, bias=bias)
    ## camera model is fix as OpenCV model
    write_intr_and_pose2database(database_path, intr.flatten(), pose_w2c)
    print('finish change')


def pickup_dso_frame(images_path, dso_result, result_path='', skip_rate=10, bias=1):
    image_files = os.listdir(images_path)
    if result_path == '':
        result_path = os.path.join(os.path.dirname(images_path), 'dso_kf')
    os.makedirs(result_path, exist_ok=True)
    dso = open(dso_result, 'r')
    dso_lines = dso.readlines()
    for line in dso_lines:
        line = line.strip().split()
        idx = '{:05d}'.format(int(line[0]) * skip_rate + bias)
        name = str(idx) + '.jpg'
        target_name = str('{:06d}'.format(int(line[0]) * skip_rate + bias)) + '.jpg'
        if name not in image_files:
            print('can not find {}'.format(name))
        shutil.copy(os.path.join(images_path, name), os.path.join(result_path, target_name))
    print('pick finish')


def save_intr(scene_path, intr, source_size=(1632, 1088), crop=(1076, 808)):
    f = open(os.path.join(scene_path, 'camera.txt'), 'w')
    K = intr[:4]
    f.write('Pinhole ')
    f.writelines(' '.join(map(str, K)))
    f.write('\n')
    f.writelines(' '.join(map(str, source_size)))
    f.write('\n')
    if len(crop) > 0:
        f.write('crop \n')
        f.writelines(' '.join(map(str, crop)))
    print('save intr to {}'.format(scene_path))


def save_dso_sparse_conf(window_path, target_dir, real_id, rate=10, skip=1, size=(1088, 1632)):
    wins = os.listdir(window_path)
    wins = [int(win.split('.')[0]) for win in wins]
    wins.sort()
    wins = [str(win) + '.txt' for win in wins]
    os.makedirs(os.path.join(target_dir, 'sparse_tuple'), exist_ok=True)
    dso_tuple = open(os.path.join(target_dir, 'tuples_dso_optimization_windows.txt'), 'w')
    target_image_name = os.listdir(os.path.join(target_dir, 'imagesRgb_resize'))
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
                    old_id = str('{:06d}'.format(int(real_id[int(line[0])]) * rate + skip))
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


def generate_depth_map(dso_result_path, image_path, depth_path, K):
    dso_pose = open(dso_result_path, 'r')
    pose_lines = dso_pose.readlines()
    points_list = []
    n = 0
    kk = 0

    gt_poses = open('/media/hjx/Jianxin_Huang1052/AirSim/run-DSO/yingrenshi2/poses_gt.txt', 'r')
    gt_poses_lines = gt_poses.readlines()
    poses_gt = {}
    for gt_line in gt_poses_lines:
        gt_line = gt_line.split()
        poses_gt[gt_line[0]] = np.array(gt_line[1:], dtype=np.float32).reshape([4, 4])

    for line in pose_lines:
    # for k in poses_gt.keys():
        if n > 0:
            n -= 1
            continue
        n = 30
        line = line.split()
        image_id = '{:06d}'.format(int(line[0]))
        xyz = np.array([line[1], line[2], line[3]], dtype=np.float32)
        q = np.array([line[-1]]+line[4:-1], dtype=np.float32)
        # q = np.array(line[4:], dtype=np.float32)
        p = poses_gt[image_id]
        p = poses_gt[image_id]
        # x = p[0, 3]
        # y = p[1, 3]
        # p[0, 3] = y
        p[1, 3] = -p[1, 3]
        pose = Pose.from_matrix(p)
        # pose = Pose(wxyz=q, tvec=xyz)
        # intr = Camera.scale_intrinsics(K, 640/1632, 480/1088)
        camera = Camera(K, p_cw=pose.inverse())
        depth = cv2.imread(os.path.join(depth_path, str(image_id) + '.png'), -1)
        # depth = cv2.resize(depth, (640, 480))
        depth = depth.astype(np.float32) / 255
        print(np.max(depth))
        print(np.min(depth))
        # image = cv2.imread(os.path.join(image_path, str(image_id) + '.png'), -1)
        points = camera.reconstruct(depth)
        points_list.append(points)
        kk += 1
        if kk > 10:
            break
    map = np.vstack(points_list)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(map)
    o3d.io.write_point_cloud('/media/hjx/Jianxin_Huang1052/AirSim/run-DSO/yingrenshi2/results/dense_map_gt.pcd', pcd)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    dso_result = '/media/hjx/Jianxin_Huang1052/dataset/DJI/0315/results/results.txt'
    save_path = '/media/hjx/dataset/DJI_colmap/rec-for-eval/0315/dense/tum_format'
    source_images = '/media/hjx/Jianxin_Huang1052/dataset/DJI/0477/imagesRgb_resize'
    scene_path = '/home/hjx/Documents/dji_data/DJI_gen2/0477'
    result_img_path = os.path.join(scene_path, 'images')

    database_path = '/media/hjx/dataset/DJI_colmap/rec-for-eval/0315/database.db'
    camera_path = '/home/hjx/Documents/dji_data/0477_colmap/cameras.txt'
    colmap_image_path = "/media/hjx/dataset/DJI_colmap/rec-for-eval/0315/dense/sparse/images.txt"
    depth_path = '/home/hjx/Documents/dji_data/0477_colmap/stereo/depth_maps'
    image_path = result_img_path

    skip_rate, bias = 8, 1

    # zjut
    intr = np.array(
        [0.7680921 * 1632, 1.386796621 * 1088, 0.4854619883 * 1632, 0.4905782976 * 1088,
         -0.033569, 0.034502, -0.000472, 0.000415])  # this camera intrinsic is running dso intrinsic
    # zjg
    # intr = np.array(
    #     [1.0530921 * 1632, 1.956796621 * 1088, 0.4854619883 * 1632, 0.4905782976 * 1088, - 0.033569, 0.034502,
    #      - 0.000472, 0.000415])  # this camera intrinsic is running dso intrinsic

    # new intr
    # intr = np.array(
    #     [1.123835784313725 * 1632, 1.997702205882353 * 1088, 0.5028463235294118 * 1632, 0.4886794117647059 * 1088,
    #      0.0014, -0.0507, 0, 0])

    # step 1
    # save_intr(scene_path, intr, source_size=(1632, 1088), crop=(1076, 808))

    # ## need to change skip rate and bias for different dataset
    # pickup_dso_frame(source_images, dso_result, result_img_path, skip_rate, bias)
    #

    # step 2
    # # ## need to change skip rate and bias for different dataset
    # fix_database(database_path, dso_result, intr, skip_rate, bias)

    # step 3
    # ## need to change skip rate and bias for different dataset
    # save_colmap_dso_2tum(colmap_image_path, dso_result, save_path, skip_rate, bias)

    # step 4
    # R = np.array([[-0.34605398, -0.93783079, -0.02683372],
    #               [-0.93745732, 0.34448431, 0.05004331],
    #               [-0.03768836, 0.04247316, -0.99838651]]
    #              )
    # t = np.array([119.12868582, 361.81692723,   2.09751106])
    # scale = 58.68862799799203
    # align_pose = Pose(wxyz=Quaternion(matrix=R), tvec=t, scale=scale)
    # process_depth(image_path, camera_path, colmap_image_path, scene_path, align_pose)

    K = np.array([[865.1232, 0, 816], [0, 863.2192, 544], [0, 0, 1]])
    image_path = '/media/hjx/Jianxin_Huang1052/AirSim/run-DSO/yingrenshi2/imagesRgb'
    depth_path = '/media/hjx/Jianxin_Huang1052/AirSim/run-DSO/yingrenshi2/depths'
    dso_result = '/media/hjx/3336-6530/组会/dso_result/yingrenshi2/results.txt'
    generate_depth_map(dso_result, image_path, depth_path, K)
