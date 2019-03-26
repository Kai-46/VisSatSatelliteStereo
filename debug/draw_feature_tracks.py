import cv2
import json
import os
import shutil


def draw_track(index, track_file, image_dir, out_dir):
    with open(track_file) as fp:
        tracks = json.load(fp)

    track_to_draw = tracks[index]

    for feature in track_to_draw:
        img_name, col, row = feature
        f = cv2.KeyPoint()
        f.pt = (col, row)
        # Dummy size
        f.size = 50
        f.angle = 0
        f.response = 10

        img = cv2.imread(os.path.join(image_dir, img_name))
        GREEN = (0, 255, 0)
        cv2.drawKeypoints(img, [f,], img, color=GREEN)

        cv2.imwrite(os.path.join(out_dir, img_name), img)


if __name__ == '__main__':
    index = 0
    colmap_dir = '/data2/kz298/mvs3dm_result/MasterSequesteredPark/colmap'
    track_file = os.path.join(colmap_dir, 'sfm_perspective/init_triangulate_inspect/kai_tracks.json')
    image_dir = os.path.join(colmap_dir, 'sfm_perspective/images')
    out_dir = os.path.join(colmap_dir, 'sfm_perspective/init_triangulate_inspect/tracks')
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.mkdir(out_dir)
    draw_track(index, track_file, image_dir, out_dir)
