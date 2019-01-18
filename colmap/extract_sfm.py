from colmap.read_model import read_model

# read all tracks
# each track is dict

def read_tracks(colmap_images, colmap_points3D):
    all_tracks = []

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

    return all_tracks

def read_camera_params(colmap_cameras, colmap_images):
    camera_params = {}

    for image_id in colmap_images:
        image = colmap_images[image_id]

        img_name = image.name
        cam = colmap_cameras[image.camera_id]

        img_size = (cam.width, cam.height)

        camera_params[img_name] = (img_size, cam.params, image.qvec, image.tvec)

    return camera_params

def extract_sfm(sparse_dir, ext='.txt'):
    colmap_cameras, colmap_images, colmap_points3D = read_model(sparse_dir, ext)

    camera_params = read_camera_params(colmap_cameras, colmap_images)
    all_tracks = read_tracks(colmap_images, colmap_points3D)

    return camera_params, all_tracks
