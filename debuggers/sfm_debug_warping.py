import os
import json
import numpy as np
import shutil
import cv2
import sys
from pyquaternion import Quaternion
import argparse
import multiprocessing

# compute the homography directly from the 3*4 projection matrix
# plane_vec is a 4 by 1 vector
# return homography matrix that warps the src image to the reference image
def compute_homography(ref_P, src_P, plane_vec):
    plane_normal = plane_vec[:3, :]
    plane_constant = plane_vec[3, 0]

    # first rewrite x=P[X; 1] as x=SX
    ref_S = (ref_P[:3, :3] + np.dot(ref_P[:3, 3:4], plane_normal.T) / plane_constant);
    src_S = (src_P[:3, :3] + np.dot(src_P[:3, 3:4], plane_normal.T) / plane_constant);

    # H = np.dot(src_S, np.linalg.inv(ref_S))
    H = np.dot(ref_S, np.linalg.inv(src_S))

    H = H / np.max(np.abs(H))   # increase numeric stability
    return H

def create_warped_images_worker(sweep_plane, camera_mat_dict, image_dir, ref_img_name, src_img_names, out_subdir_dict, avg_img_out_dir):
    i, plane_vec = sweep_plane
    ref_im = np.float32(cv2.imread(os.path.join(image_dir, ref_img_name))) / 255.0
    cnt = 1
    for img_name in src_img_names:
        print('plane {}, ({},{},{},{}) warping {} to {}'.format(i, plane_vec[0, 0], plane_vec[1, 0], plane_vec[2, 0], plane_vec[3, 0],
                                                                        img_name, ref_img_name))
        # compute homography
        H = compute_homography(camera_mat_dict[ref_img_name], camera_mat_dict[img_name], plane_vec)
        # debug
        # print('H: {}'.format(H))

        # warp image
        im = np.float32(cv2.imread(os.path.join(image_dir, img_name))) / 255.0
        ref_h, ref_w = ref_im.shape[:2]

        warped_im = cv2.warpPerspective(im, H, (ref_w, ref_h), borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(os.path.join(out_subdir_dict[img_name], 'warped_plane_{:04}.jpg'.format(i)), np.uint8(warped_im * 255.0))

        ref_im += warped_im
        cnt += 1
    ref_im /= cnt
    cv2.imwrite(os.path.join(avg_img_out_dir, 'avg_plane_{:04}.jpg'.format(i)), np.uint8(ref_im * 255.0))


def create_warped_images(sfm_perspective_dir, ref_img_id, z_min, z_max, num_planes, normal, out_dir, src_img_ids=[], max_processes=None):
    # for each source image, it would create a folder to save the warped images for all the sweeping planes
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)

    # load camera matrices
    with open(os.path.join(sfm_perspective_dir, 'init_ba_camera_dict.json')) as fp:
        camera_dict = json.load(fp)

    if len(src_img_ids) == 0:
        useful_img_ids = []
        for item in camera_dict.keys():
            idx = item.find('_')
            id = int(item[:idx])
            useful_img_ids.append(id)
    else:
        if ref_img_id in src_img_ids:
            raise Exception('ref_img_id should not appear in src_img_ids')
        else:
            useful_img_ids = [ref_img_id, ] + src_img_ids

    img_id2name = {}
    img_name2id = {}
    camera_size_dict = {}
    camera_mat_dict = {}
    for item in camera_dict.keys():
        idx = item.find('_')
        id = int(item[:idx])
        if id not in useful_img_ids:
            continue

        img_id2name[id] = item
        img_name2id[item] = id

        params = camera_dict[item]
        w, h, fx, fy, cx, cy, s = params[:7]
        camera_size_dict[item] = (w, h)

        qvec = params[7:11]
        tvec = params[11:14]
        K = np.array([[fx, s, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]])
        R = Quaternion(qvec[0], qvec[1], qvec[2], qvec[3]).rotation_matrix
        t = np.array(tvec).reshape((3, 1))
        P = np.dot(K, np.hstack((R, t)))
        P = P / np.max(np.abs(P))   # increase numeric stability
        camera_mat_dict[item] = P

    ref_img_name = img_id2name[ref_img_id]

    out_subdir_dict = {}
    src_img_names = []
    for item in sorted(img_name2id.keys()):
        if img_name2id[item] == ref_img_id:
            continue
        src_img_names.append(item)
        # create output folders; one for each source image
        subdir = os.path.join(out_dir, item[:-4])   # remove '.png' suffix
        if os.path.exists(subdir):  # remove existing subdir
            shutil.rmtree(subdir)
        os.mkdir(subdir)
        out_subdir_dict[item] = subdir

    # loop over all the sweep planes
    z_sequence = np.linspace(z_min, z_max, num_planes)
    sweep_plane_sequence = []
    for i in range(len(z_sequence)):
        plane_vec = np.array([normal[0], normal[1], normal[2], z_sequence[i]]).reshape(4, 1) 
        sweep_plane_sequence.append((i, plane_vec))

    image_dir = os.path.join(sfm_perspective_dir, 'images')
    print('Start warping all the images to the same reference view: {}...'.format(ref_img_name))
    shutil.copy2(os.path.join(image_dir, ref_img_name), out_dir)

    # ref_im = cv2.imread(os.path.join(image_dir, ref_img_name))
    # cv2.imwrite(os.path.join(out_dir, ref_img_name[:-4]+'.jpg'), ref_im)
    
    avg_img_out_dir = os.path.join(out_dir, 'avg_img')
    if os.path.exists(avg_img_out_dir):
        shutil.rmtree(avg_img_out_dir)
    os.mkdir(avg_img_out_dir)

    # debug
    # create_warped_images_worker(sweep_plane_sequence[0], camera_mat_dict, image_dir, ref_img_name, src_img_names, out_subdir_dict, avg_img_out_dir)

    if max_processes is None:
        max_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(max_processes)

    results = []
    for sweep_plane in sweep_plane_sequence:
        r = pool.apply_async(create_warped_images_worker, 
                            (sweep_plane, camera_mat_dict, image_dir, ref_img_name, src_img_names, out_subdir_dict, avg_img_out_dir))
        results.append(r)

    [r.wait() for r in results]     # sync

    # create an average video
    cmd = 'ffmpeg -y -framerate 25 -i  {} \
            -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
            {}'.format(os.path.join(avg_img_out_dir, 'avg_plane_%04d.jpg'),
                       os.path.join(out_dir, 'avg_img.mp4'))
    os.system(cmd)

def debug():
    # {work_dir}/colmap/sfm_perspective
    sfm_perspective_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville/colmap/sfm_perspective'
    out_dir = '/data2/kz298/core3d_result/aoi-d4-jacksonville/debug_warping'

    # warp the other images to this reference image frame
    ref_img_id = 5
    # src_img_ids = [6, 7]
    # src_img_ids = [0, 1]
    src_img_ids = []      # if set to empty list, then all the other images except the reference image will be warped

    # height range
    z_min = 20  # meters
    z_max = 200 
    # number of sweeping planes and normal direction
    # a plane with height z is written as n^x-z=0
    num_planes = 360
    normal = (0, 0, 1)

    create_warped_images(sfm_perspective_dir, ref_img_id, z_min, z_max, num_planes, normal, out_dir, src_img_ids)

def deploy():
    parser = argparse.ArgumentParser(description='Create Warped Images')
    parser.add_argument('--sfm_perspective_dir', 
                        help='path to the satellite_stereo sfm_perspective directory')
    parser.add_argument('--ref_img_id', type=int,
                        help='image index shown in the image name of satellite_stereo')
    parser.add_argument('--src_img_ids', type=str, default='',
                        help='list of source image indices split by comma; default value is all the images except the reference image')
    parser.add_argument('--z_min', type=float,
                        help='z_min for the sweeping plane; unit is meter')
    parser.add_argument('--z_max', type=float,
                        help='z_max for the sweeping plane; unit is meter')
    parser.add_argument('--num_planes', type=int,
                        help='number of sweeping planes to be used')
    parser.add_argument('--out_dir', 
                        help='output directory to save the warped images; should be empty')
    parser.add_argument('--max_processes', type=int, default=-1,
                        help='maximum number of processes to be launched; default value is the number of cpu cores')
    args = parser.parse_args()
    
    if args.max_processes <= 0:
        args.max_processes = multiprocessing.cpu_count()

    if not args.src_img_ids:
        args.src_img_ids = []   # default is all the images
    else:
        tmp = args.src_img_ids
        args.src_img_ids = [int(x) for x in tmp.split(',')]

    normal = (0, 0, 1)
    print(args)

    create_warped_images(args.sfm_perspective_dir, args.ref_img_id, 
                         args.z_min, args.z_max, args.num_planes, normal, args.out_dir, args.src_img_ids, args.max_processes)


if __name__ == '__main__':
    debug()
    # deploy()
