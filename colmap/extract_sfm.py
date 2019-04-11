from colmap.read_model import read_model
import numpy as np
import json
import os


# read all tracks
# each track is dict
def read_tracks(colmap_images, colmap_points3D, return_track_ids=False):
    all_tracks = []

    track_ids = None
    if return_track_ids:
        track_ids = []

    for point3D_id in colmap_points3D:
        point3D = colmap_points3D[point3D_id]

        image_ids = point3D.image_ids
        point2D_idxs = point3D.point2D_idxs

        cur_track = {}
        cur_track['xyz'] = (point3D.xyz[0], point3D.xyz[1], point3D.xyz[2])
        cur_track['err'] = point3D.error

        cur_track_len = len(image_ids)
        assert (cur_track_len == len(point2D_idxs))
        pixels = []
        for i in range(cur_track_len):
            image = colmap_images[image_ids[i]]
            img_name = image.name

            point2D_idx = point2D_idxs[i]
            point2D = image.xys[point2D_idx]
            assert (image.point3D_ids[point2D_idx] == point3D_id)

            pixels.append((img_name, point2D[0], point2D[1]))

        # sort pixels by the img_name
        pixels = sorted(pixels, key=lambda x: x[0])

        cur_track['pixels'] = pixels
        all_tracks.append(cur_track)

        if return_track_ids:
            track_ids.append(point3D_id)

    if return_track_ids:
        return all_tracks, track_ids
    else:
        return all_tracks


def read_camera_dict(colmap_cameras, colmap_images):
    camera_dict = {}
    for image_id in colmap_images:
        image = colmap_images[image_id]

        img_name = image.name
        cam = colmap_cameras[image.camera_id]

        img_size = (cam.width, cam.height)
        params = tuple(cam.params)
        qvec = tuple(image.qvec)
        tvec = tuple(image.tvec)

        # w, h, fx, fy, cx, cy, s, qvec, tvec
        camera_dict[img_name] = img_size + params + qvec + tvec

    return camera_dict


def extract_camera_dict(sparse_dir, ext='.txt'):
    colmap_cameras, colmap_images, _ = read_model(sparse_dir, ext)

    camera_dict = read_camera_dict(colmap_cameras, colmap_images)

    return camera_dict


def extract_all_tracks(sparse_dir, ext='.txt'):
    _, colmap_images, colmap_points3D = read_model(sparse_dir, ext)
    all_tracks = read_tracks(colmap_images, colmap_points3D)

    return all_tracks


def write_all_tracks(all_tracks, xyz_file, track_file):
    points = []
    tracks = []
    for track in all_tracks:
        points.append(track['xyz'] + (track['err'],))
        tracks.append(track['pixels'])
    points = np.array(points)

    np.savetxt(xyz_file, points, header='# format: x, y, z, reproj_err')

    with open(track_file, 'w') as fp:
        json.dump(tracks, fp)


def extract_all_tracks_to_file(sparse_dir, xyz_file, track_file):
    all_tracks = extract_all_tracks(sparse_dir)
    write_all_tracks(all_tracks, xyz_file, track_file)


def extract_all_to_files(sparse_dir, camera_dict_file, xyz_file, track_file, ext='.txt', track_ids_file=None):
    colmap_cameras, colmap_images, colmap_points3D = read_model(sparse_dir, ext)

    camera_dict = read_camera_dict(colmap_cameras, colmap_images)
    with open(camera_dict_file, 'w') as fp:
        json.dump(camera_dict, fp, indent=2, sort_keys=True)

    if track_ids_file is not None:
        all_tracks, track_ids = read_tracks(colmap_images, colmap_points3D, return_track_ids=True)
        with open(track_ids_file, 'w') as fp:
            json.dump(track_ids, fp)
    else:
        all_tracks = read_tracks(colmap_images, colmap_points3D)
    write_all_tracks(all_tracks, xyz_file, track_file)


def extract_all_to_dir(sparse_dir, out_dir, ext='.txt'):
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    camera_dict_file = os.path.join(out_dir, 'kai_cameras.json')
    xyz_file = os.path.join(out_dir, 'kai_coordinates.txt')
    track_file = os.path.join(out_dir, 'kai_tracks.json')
    track_ids_file = os.path.join(out_dir, 'kai_track_ids.json')
    extract_all_to_files(sparse_dir, camera_dict_file, xyz_file, track_file, ext, track_ids_file)

    return camera_dict_file, xyz_file, track_file


#
# def extract_sfm(sparse_dir, ext='.txt'):
#     colmap_cameras, colmap_images, colmap_points3D = read_model(sparse_dir, ext)
#
#     camera_params = read_camera_params(colmap_cameras, colmap_images)
#     all_tracks = read_tracks(colmap_images, colmap_points3D)
#
#     return camera_params, all_tracks


if __name__ == '__main__':
    sparse_dir = '/data2/kz298/mvs3dm_result/MasterProvisional2/colmap/sfm_perspective/init_triangulate'
    out_dir = '/data2/kz298/tmp'
    extract_all_to_dir(sparse_dir, out_dir)
