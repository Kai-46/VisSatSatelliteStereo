# get a stats of the sparse reconstruction

import lib.read_model
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import numpy as np

class SparseModel(object):
    def __init__(self, sparse_dir, ext):
        self.sparse_dir = os.path.abspath(sparse_dir)

        self.cameras, self.images, self.points3D = lib.read_model.read_model(self.sparse_dir, ext)

        self.img_cnt = len(self.images.keys())

    def inspect_image_key_points(self):
        img_names = []
        img_widths = []
        img_heights = []

        key_point_cnt = []
        for image_id in self.images:
            image = self.images[image_id]
            img_names.append(image.name)
            key_point_cnt.append(len(image.xys))

            cam = self.cameras[image.camera_id]
            img_widths.append(cam.width)
            img_heights.append(cam.height)

        plt.clf()
        plt.bar(range(0, self.img_cnt), key_point_cnt)
        plt.xticks(ticks=range(0, self.img_cnt), labels=img_names, rotation=90)
        plt.ylabel('# of sift features')
        plt.grid(True)
        plt.title('total # of images: {}'.format(self.img_cnt))
        plt.tight_layout()

        plt.savefig(os.path.join(self.sparse_dir, 'inspect_key_points.png'))
        #plt.show()

        plt.clf()
        plt.plot(range(0, self.img_cnt), img_widths, 'b-o', label='width')
        #plt.legend('width')
        plt.plot(range(0, self.img_cnt), img_heights, 'r-+', label='height')
        #plt.legend('height')
        plt.legend()
        #plt.legend('width', 'height')
        plt.xticks(ticks=range(0, self.img_cnt), labels=img_names, rotation=90)
        plt.ylabel('# of pixels')
        plt.grid(True)
        plt.title('total # of images: {}'.format(self.img_cnt))
        plt.tight_layout()
        plt.savefig(os.path.join(self.sparse_dir, 'inspect_image_size.png'))
        #return (img_names, key_point_cnt)

    def stats_points3D(self):
        for point3D_id in self.points3D:
            print(self.points3D[point3D_id].id)
            print(self.points3D[point3D_id].xyz)
            print(self.points3D[point3D_id].error)
            print(len(self.points3D[point3D_id].image_ids))
            print(len(self.points3D[point3D_id].point2D_idxs))
            print('\n')

    def inspect_feature_tracks(self):
        all_tracks = []
        all_points_id = []
        all_points_xyz = []
        all_points_err = []

        for point3D_id in self.points3D:
            point3D = self.points3D[point3D_id]
            image_ids = point3D.image_ids
            point2D_idxs = point3D.point2D_idxs

            all_points_id.append(point3D.id)
            all_points_xyz.append((point3D.xyz[0], point3D.xyz[1], point3D.xyz[2]))
            all_points_err.append(float(point3D.error))

            cur_track_len = len(image_ids)
            assert (cur_track_len == len(point2D_idxs))
            cur_track = []
            for i in range(cur_track_len):
                image = self.images[image_ids[i]]
                img_name = image.name
                point2D_idx = point2D_idxs[i]
                point2D = image.xys[point2D_idx]
                assert (image.point3D_ids[point2D_idx] == point3D_id)
                cur_track.append((img_name, point2D[0], point2D[1]))
            all_tracks.append(cur_track)

        print('number of feature tracks: {}'.format(len(all_tracks)))
        print('first feature track: {}'.format(all_tracks[0]))

        with open(os.path.join(self.sparse_dir, 'inspect_points_id.json'), 'w') as fp:
            json.dump(all_points_id, fp, indent=2)

        with open(os.path.join(self.sparse_dir, 'inspect_points_tracks.json'), 'w') as fp:
            json.dump(all_tracks, fp, indent=2)

        with open(os.path.join(self.sparse_dir, 'inspect_points_xyz.json'), 'w') as fp:
            json.dump(all_points_xyz, fp, indent=2)

        with open(os.path.join(self.sparse_dir, 'inspect_points_err.json'), 'w') as fp:
            json.dump(all_points_err, fp, indent=2)

        # check distribution of track_len
        track_len = [len(x) for x in all_tracks]
        max_track_len = max(track_len)
        plt.clf()
        plt.hist(track_len, bins=np.arange(0.5, max_track_len + 1.5, 1))
        plt.xticks(range(1, max_track_len+1))
        plt.ylabel('# of tracks')
        plt.xlabel('track length')
        plt.title('total # of images: {}\ntotal # of tracks: {}\nmin track length: {}, max track length: {}'
                  .format(self.img_cnt, len(all_tracks), min(track_len), max_track_len))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.sparse_dir, 'inspect_track_len.png'))

        #plt.show()

        # check reprojection error
        plt.clf()
        plt.figure(figsize=(14, 5), dpi=80)
        plt.hist(all_points_err, bins=20, density=True, cumulative=True)
        max_points_err = max(all_points_err)
        plt.xticks(np.arange(0, max_points_err+0.01, 0.1))
        plt.yticks(np.arange(0, 1.01, 0.1))
        plt.xlabel('reprojection error (# pixels)')
        plt.ylabel('cdf')
        plt.title('total # of images: {}\ntotal # of sparse 3D points: {}\nreproj. err. (pixels): min {:.6f}, mean {:.6f}, median {:.6f}, max {:.6f}'
                  .format(self.img_cnt, len(all_points_err), min(all_points_err), np.mean(all_points_err), np.median(all_points_err), max_points_err))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.sparse_dir, 'inspect_reproj_err.png'))
        #plt.show()

        #return all_tracks



def test():
    # sparse_dir = '/data2/kz298/bak/data_aoi-d1-wpafb_pinhole/sparse_ba'
    # sparse_dir = '/data2/kz298/bak/data_aoi-d2-wpafb_pinhole/sparse_ba'
    # sparse_dir = '/data2/kz298/bak/data_aoi-d3-ucsd_pinhole/sparse_ba'
    sparse_dir = '/data2/kz298/bak/data_aoi-d4-jacksonville_pinhole/sparse_ba'

    ext = '.bin'

    sparse_model = SparseModel(sparse_dir, ext)

    sparse_model.inspect_feature_tracks()
    sparse_model.inspect_image_key_points()

    #tracks = sparse_model.feature_tracks()
    #sparse_model.image_key_points()

    # with open(os.path.join(sparse_dir, 'tracks.json'), 'w') as fp:
    #     json.dump(tracks, fp, indent=2)


if __name__ == '__main__':
    test()


