import numpy as np
from pyquaternion import Quaternion
import cv2
import os
import shutil
import glob


def rename_files(root_dir, format='{:06d}', ext='png'):
    scenes = os.listdir(root_dir)
    for scene in scenes:
        scene_dir = os.path.join(root_dir, scene)
        files = os.listdir(os.path.join(scene_dir, 'images'))
        files.sort()
        for i, file in enumerate(files):
            # name = str(format.format(i)) + '.' + ext
            name = file.split('.')[0]
            if len(name) < 6:
                name = str(format.format(int(name))) + '.' + ext

            os.rename(os.path.join(scene_dir, 'images', file), os.path.join(scene_dir, 'images', str(name)))
            os.rename(os.path.join(scene_dir, 'depths', file), os.path.join(scene_dir, 'depths', str(name)))
            print('rename {}'.format(name))


def generate_intrinsic(scene_file, height=480, width=640):
    K = np.array([[0.5 * 640, 0, 0.5 * 640], [0, 0.5 * 480, 0.5 * 480], [0, 0, 1]])  # for simulation data
    inc = open(os.path.join(scene_file, 'camera.txt'), 'w')
    inc.write('Pinhole')
    inc.write(' ')
    inc.write(str(K[0, 0]))
    inc.write(' ')
    inc.write(str(K[1, 1]))
    inc.write(' ')
    inc.write(str(K[0, 2]))
    inc.write(' ')
    inc.write(str(K[1, 2]))
    inc.write(' ')
    inc.write('0')
    inc.write('\n')
    inc.write(str(width))
    inc.write(' ')
    inc.write(str(height))
    inc.write('\n')
    inc.write('"crop" / "full" / "none" / "fx fy cx cy 0"')
    inc.write('\n')
    inc.write('out_width out_height')
    inc.close()


def generate_sim_gt_pose(sensor_msg_file, image_file, format='{:06d}', ext='png'):
    pose_list = {}
    with open(sensor_msg_file, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            time_stamp = line[10:line.find('POS')].split()[0][0:-1]
            name = str(format.format(i))
            split_f = sensor_msg_file.split('/')[-2]

            xyz_begin = line.find('xyz')
            xyz_end = line.find('orien')
            xyz = np.array(line[xyz_begin + 4: xyz_end].strip().split(), dtype=np.float32)
            if split_f == 'Africa2400':
                xyz[2] = xyz[2] - 30

            orien_begin = line.find('QZ')
            orien_end = line.find('IMU_GPS')
            orien = np.array(line[orien_begin + 3:orien_end].strip().split(), dtype=np.float32)  # QW,QX,QY,QZ
            q = Quaternion(orien)
            T = q.transformation_matrix
            T[:3, 3] = xyz
            pose_list[name] = T
        f.close()

    # generate cva mvs dataset
    dir_name = os.path.dirname(sensor_msg_file)
    pose_gt_file = open(os.path.join(dir_name, 'poses_gt.txt'), 'w')
    pose_dso_file = open(os.path.join(dir_name, 'poses_dso.txt'), 'w')

    # generate sim dso key frame for every 3-5 frames
    images_name = os.listdir(image_file)
    n = 0
    for key in pose_list.keys():
        if not (key + '.' + ext) in images_name:
            continue
        trans = pose_list[key]
        pose_gt_file.write(key)
        pose_gt_file.write(' ')
        for i in range(4):
            for j in range(4):
                pose_gt_file.write(str(trans[i, j]))
                pose_gt_file.write(' ')
        pose_gt_file.write('\n')
        if n > 0:
            n -= 1
            continue
        n = np.random.randint(3, 6, 1)
        pose_dso_file.write(key)
        pose_dso_file.write(' ')
        for i in range(4):
            for j in range(4):
                pose_dso_file.write(str(trans[i, j]))
                pose_dso_file.write(' ')
        pose_dso_file.write('\n')
    pose_gt_file.close()
    pose_dso_file.close()


def generate_tuples(pose_dso_file, scale=1):
    if not os.path.isfile(pose_dso_file):
        pose_dso_file = os.path.join(pose_dso_file, 'poses_dso.txt')
    pose_dso = open(pose_dso_file, 'r')
    pose_lines = pose_dso.readlines()

    dir_name = os.path.dirname(pose_dso_file)
    w8 = open(os.path.join(dir_name, 'tuples_dso_optimization_windows.txt'), 'w')  # save for 8 windows
    last3 = open(os.path.join(dir_name, 'tuples_dso_optimization_windows_last3.txt'), 'w')  # save last 3 windows
    for i in range(len(pose_lines) - 8):
        w8.write('8')
        w8.write(' ')
        last3.write('3')
        last3.write(' ')
        for j in range(8):
            line = pose_lines[i + j]
            if j > 4:
                last3.write(line.split()[0])
                last3.write(' ')
            w8.write(line.split()[0])
            w8.write(' ')
        w8.write(str(scale))
        w8.write('\n')
        last3.write(str(scale))
        last3.write('\n')
    w8.close()
    last3.close()


def sample_dji_datasets(source_path, target_path, frequency=1):
    assert len(target_path) > 0, 'please give a target file path'
    os.makedirs(target_path, exist_ok=True)
    images = os.listdir(source_path)
    images.sort()
    for i in range(2400, 3900, frequency):
        image = images[i]
        source_name = image
        name = image.split('.')[0]
        if len(name) != 6:
            name = '{:06d}'.format(int(name))
            image = str(name) + '.jpg'
        s = os.path.join(source_path, source_name)
        t = os.path.join(target_path, image)
        shutil.copy(s, t)
    print('sample ok')


def save_dji_subsequence_poses(pose_list, path):
    # dji don't have gt, so dso and gt is the same from colmap
    if not os.path.isdir(path):
        path = os.path.dirname(path)
    dir_name = path
    pose_gt_file = open(os.path.join(dir_name, 'poses_gt.txt'), 'w')
    pose_dso_file = open(os.path.join(dir_name, 'poses_dso.txt'), 'w')

    for key in pose_list.keys():
        trans = pose_list[key]

        if len(key) < 6:
            key = str('{:06d}'.format(int(key)))

        pose_gt_file.write(key)
        pose_gt_file.write(' ')
        for i in range(4):
            for j in range(4):
                pose_gt_file.write(str(trans[i, j]))
                pose_gt_file.write(' ')
        pose_gt_file.write('\n')
        # it is duplicate, but more clear, so I not change now
        pose_dso_file.write(key)
        pose_dso_file.write(' ')
        for i in range(4):
            for j in range(4):
                pose_dso_file.write(str(trans[i, j]))
                pose_dso_file.write(' ')
        pose_dso_file.write('\n')
    pose_gt_file.close()
    pose_dso_file.close()


def read_poses(colmap_images_txt):
    with open(colmap_images_txt, 'r') as f:
        lines = f.readlines()
        # Image list with two lines of data per image:
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        #   POINTS2D[] as (X, Y, POINT3D_ID)
        camera_pose = {}  # word to camera, key is the image name
        for i in range(4, len(lines), 2):
            line = lines[i].strip().split()
            q = Quaternion(np.array(line[1:5], dtype=np.float32))
            t = np.array(line[5:8], dtype=np.float32)
            T = q.transformation_matrix
            T[:3, 3] = t
            name = line[-1].split('.')[0]
            camera_pose[name] = T
        new_keys = sorted(camera_pose.keys())
        new_pose = {}
        for key in new_keys:
            new_pose[key] = camera_pose[key]
        f.close()
    return new_pose


def save_pose_as_tum(pose_file, gps_file, target_dir):
    """ save pose file and gps file to tum format, and can be used evo for calculate scale
    for simple, the rotation all give (0, 0, 0, 1)(quaternion)
    tum format: timestamp x y z q_x q_y q_z q_w
    """
    pose_dso_f = open(pose_file, 'r')
    pose_gps_f = open(gps_file, 'r')
    # poses_dso = read_poses(pose_file)
    poses_gps_lines = pose_gps_f.readlines()
    poses_dso_lines = pose_dso_f.readlines()
    poses_gps = {}
    poses_dso = {}
    for line in poses_gps_lines:
        line = line.strip().split()
        poses_gps[line[0]] = line[1:]

    for i in range(4, len(poses_dso_lines), 2):
        line = poses_dso_lines[i].strip().split()
        # t = np.array(line[5:8], dtype=np.float32)
        name = line[-1].split('.')[0]
        poses_dso[name] = line[5:8]

    dso_tum = open(os.path.join(target_dir, 'dso_tum.txt'), 'w')
    gps_tum = open(os.path.join(target_dir, 'gps_tum.txt'), 'w')
    q = ' '.join(['0', '0', '0', '1'])
    for pose_dso in poses_dso.keys():
        if pose_dso in poses_gps:
            p_d = poses_dso[pose_dso]
            p_g = poses_gps[pose_dso]
            p_d = ' '.join(p_d)
            p_g = ' '.join(p_g)

            dso_tum.write(pose_dso)
            dso_tum.write(' ')
            dso_tum.writelines(p_d)
            dso_tum.write(' ')
            dso_tum.writelines(q)
            dso_tum.write('\n')

            gps_tum.write(pose_dso)
            gps_tum.write(' ')
            gps_tum.writelines(p_g)
            gps_tum.write(' ')
            gps_tum.writelines(q)
            gps_tum.write('\n')

        else:
            print('pose {} don\'t have gps value'.format(pose_dso))


def process_airsim(images_path, msg_path, target_path, format='{:05d}', z_bias=0):
    images = os.listdir(images_path)
    images.sort()
    msg = open(msg_path, 'r')
    msg_lines = msg.readlines()
    pose_list = {}
    gps_list_noise = {}
    gps_list = {}

    dir_name = os.path.dirname(msg_path)
    pose_gt_file = open(os.path.join(target_path, 'poses_gt.txt'), 'w')
    gps_file = open(os.path.join(target_path, 'GPS_gt.txt'), 'w')
    gps_noise_file = open(os.path.join(target_path, 'GPS.txt'), 'w')

    depth_dir = os.path.join(dir_name, 'picture', 'DepthVis')
    os.makedirs(os.path.join(target_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(target_path, 'depths'), exist_ok=True)

    for i, line in enumerate(msg_lines):
        line = line.strip()
        time_stamp = line[10:line.find('POS')].split()[0][0:-1]
        if time_stamp + '.png' not in images:
            continue
        name = str(format.format(i))
        xyz_begin = line.find('xyz')
        xyz_end = line.find('orien')
        xyz = np.array(line[xyz_begin + 4: xyz_end].strip().split(), dtype=np.float32)
        xyz[2] -= z_bias

        orien_begin = line.find('QZ')
        orien_end = line.find('IMU_GPS')
        orien = np.array(line[orien_begin + 3:orien_end].strip().split(), dtype=np.float32)  # QW,QX,QY,QZ
        q = Quaternion(orien)
        T = q.transformation_matrix
        T[:3, 3] = xyz
        pose_list[name] = T

        gps = line.split('\t')[-1].strip().split(',')
        ggg = []
        gggg = []
        for i, gg in enumerate(gps):
            gggg.append(float(gg.split(':')[-1].strip(' ')))
            if i == 0:
                # print(float(gg.split(':')[-1].strip(' ')) * np.random.randn(1).item() * 1.5 * 0.00001141)
                ggg.append(float(gg.split(':')[-1].strip(' ')) + np.random.randn(1).item() * 0.00001141)
                # ggg.append(float(gg.split(':')[-1].strip(' ')) + np.abs(np.random.exponential(1)).item() * 1. * 0.00001141)
            if i == 1:
                ggg.append(float(gg.split(':')[-1].strip(' ')) + np.random.randn(1).item() * 0.00000899)
            elif i == 2:
                # print(float(gg.split(':')[-1].strip(' ')))

                ggg.append(float(gg.split(':')[-1].strip(' '))) #+ np.abs(np.random.exponential(1)).item())
        # gps_list[name] = np.array(ggg)
        gps_list_noise[name] = ggg
        gps_list[name] = gggg
    # print(gps_list)


        # img = cv2.imread(os.path.join(images_path, time_stamp + '.png'), -1)
        # img = cv2.resize(img, (1632, 1088))
        # cv2.imwrite(os.path.join(target_path, 'images', name + '.png'), img)
        #
        # img = cv2.imread(os.path.join(depth_dir, time_stamp + '.png'), -1)
        # img = cv2.resize(img, (1632, 1088))
        # cv2.imwrite(os.path.join(target_path, 'depths', name + '.png'), img)

        shutil.copy(os.path.join(images_path, time_stamp + '.png'), os.path.join(target_path, 'images', name + '.png'))
        shutil.copy(os.path.join(depth_dir, time_stamp + '.png'), os.path.join(target_path, 'depths', name + '.png'))

    for key in pose_list.keys():
        pose = pose_list[key]
        gps = gps_list[key]
        gps_noise = gps_list_noise[key]
        pose_gt_file.write(str(key) + ' ')
        pose_gt_file.writelines(' '.join(map(str, pose.flatten())))
        pose_gt_file.write('\n')

        gps_file.write(str(key) + ' ')
        gps_file.writelines(' '.join(map(str, gps)))
        gps_file.write('\n')

        gps_noise_file.write(str(key) + ' ')
        gps_noise_file.writelines(' '.join(map(str, gps_noise)))
        gps_noise_file.write('\n')


if __name__ == '__main__':
    ########################################
    # generate mvs training txt
    data_dir = '/home/hjx/Documents/airsim_data'
    scenes = os.listdir(data_dir)
    rename_files(data_dir)
    for scene in scenes:
        scene_path = os.path.join(data_dir, scene)
        images_path = os.path.join(scene_path, 'images')
        msg_file = glob.glob(scene_path + '/Drone1*.txt')[0]
        generate_intrinsic(scene_path)
        generate_sim_gt_pose(msg_file, images_path)
        generate_tuples(os.path.join(scene_path, 'poses_dso.txt'))

    #########################################
    # images_path = '/home/hjx/Documents/dji_data/imagesRgb'
    # target_path = '/home/hjx/Documents/dji_data/zjut_3/images'
    # sample_dji_datasets(images_path, target_path, 10)

    # rename_files('/media/hjx/Sakura/UE4图片采集/eval-data', ext='png')
    # gps_file = '/media/hjx/3336-6530/组会/11111/GPS.txt'
    # dso_file = '/media/hjx/3336-6530/组会/11111/images.txt'
    # save_pose_as_tum(dso_file, gps_file, '/media/hjx/3336-6530/组会/11111')

    # msg_file = '/media/hjx/Sakura/UE4图片采集/yingrenshi/Drone1_ImuSensor1.txt'
    # image_path = '/media/hjx/Sakura/UE4图片采集/yingrenshi/picture/Scene'
    # target_path = '/media/hjx/Sakura/UE4图片采集/yingrenshi_process'
    # process_airsim(image_path, msg_file,target_path, z_bias=0)
    # depth_path = '/media/hjx/dataset/sim/Sci_Art/depths'
    # images = os.listdir(depth_path)
    # for d in images:
    #     depth = cv2.imread(os.path.join(depth_path, d), -1)
    #     depth = np.array(depth).astype(np.float32) / 255
    #     print(np.max(depth))
    # rename_files('/home/hjx/Downloads/icl/traj0_frei_png/11111')