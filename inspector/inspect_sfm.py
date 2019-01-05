from colmap.read_model import read_model
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np
import quaternion


class InspectSparseModel(object):
    def __init__(self, sparse_dir, out_dir, ext='.txt'):
        self.sparse_dir = os.path.abspath(sparse_dir)
        self.out_dir = os.path.abspath(out_dir)

        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        self.cameras, self.images, self.points3D = read_model(self.sparse_dir, ext)

        self.img_cnt = len(self.images.keys())

    def inspect(self):
        self.inspect_feature_tracks()
        self.inspect_image_key_points()

    def inspect_image_key_points(self):
        img_id2name = []
        img_names = []
        img_widths = []
        img_heights = []

        key_point_cnt = []
        for image_id in self.images:
            image = self.images[image_id]

            img_id2name.append((image_id, image.name))
            img_names.append(image.name)
            key_point_cnt.append(len(image.xys))

            cam = self.cameras[image.camera_id]
            img_widths.append(cam.width)
            img_heights.append(cam.height)

        with open(os.path.join(self.out_dir, 'inspect_img_id2name.jpg'), 'w') as fp:
            json.dump(img_id2name, fp)

        plt.figure()
        plt.bar(range(0, self.img_cnt), key_point_cnt)
        plt.xticks(ticks=range(0, self.img_cnt), labels=img_names, rotation=90)
        plt.ylabel('# of sift features')
        plt.grid(True)
        plt.title('total # of images: {}'.format(self.img_cnt))
        plt.tight_layout()

        plt.savefig(os.path.join(self.out_dir, 'inspect_key_points.jpg'))
        plt.close()
        #plt.show()

        plt.figure()
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
        plt.savefig(os.path.join(self.out_dir, 'inspect_image_size.jpg'))
        plt.close()

    # this method is documented in colmap src/mvs/model.cc
    def inspect_depth_range(self):
        depth_range = {}
        for img_id in self.images:
            img_name = self.images[img_id].name
            depth_range[img_name] = []

        for point3D_id in self.points3D:
            point3D = self.points3D[point3D_id]
            x = point3D.xyz.reshape((3, 1))
            for img_id in point3D.image_ids:
                img_name = self.images[img_id].name
                qvec = self.images[img_id].qvec
                tvec = self.images[img_id].tvec.reshape((3, 1))
                R = quaternion.as_rotation_matrix(np.quaternion(qvec[0], qvec[1], qvec[2], qvec[3]))
                x = np.dot(R,x) + tvec
                depth = x[2, 0]
                if depth > 0:
                    depth_range[img_name].append(depth)

        for img_name in depth_range:
            if depth_range[img_name]:
                tmp = sorted(depth_range[img_name])
                cnt = len(tmp)
                min_depth = tmp[int(0.01 * cnt)] * (1 - 0.25)
                max_depth = tmp[int(0.99 * cnt)] * (1 + 0.25)
                depth_range[img_name] = (min_depth, max_depth)
            else:
                depth_range[img_name] = (0, 0)

        with open(os.path.join(self.out_dir, 'inspect_depth_range.json'), 'w') as fp:
            json.dump(depth_range, fp)


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
        # print('first feature track: {}'.format(all_tracks[0]))

        with open(os.path.join(self.out_dir, 'inspect_points_id.json'), 'w') as fp:
            json.dump(all_points_id, fp, indent=2)

        with open(os.path.join(self.out_dir, 'inspect_points_tracks.json'), 'w') as fp:
            json.dump(all_tracks, fp, indent=2)

        with open(os.path.join(self.out_dir, 'inspect_points_xyz.json'), 'w') as fp:
            json.dump(all_points_xyz, fp, indent=2)

        with open(os.path.join(self.out_dir, 'inspect_points_err.json'), 'w') as fp:
            json.dump(all_points_err, fp, indent=2)

        # check distribution of track_len
        plt.figure(figsize=(14, 5), dpi=80)
        track_len = [len(x) for x in all_tracks]
        max_track_len = max(track_len)
        plt.hist(track_len, bins=np.arange(0.5, max_track_len + 1.5, 1))
        plt.xticks(range(1, max_track_len+1))
        plt.ylabel('# of tracks')
        plt.xlabel('track length')
        plt.title('total # of images: {}\ntotal # of tracks: {}\ntrack length, min: {}, mean: {:.6f}, median: {}, max: {}'
                  .format(self.img_cnt, len(all_tracks), min(track_len), np.mean(track_len), np.median(track_len), max_track_len))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'inspect_track_len.jpg'))
        plt.close()
        #plt.show()

        # check reprojection error
        plt.figure(figsize=(14, 5), dpi=80)
        plt.hist(all_points_err, bins=50, density=True, cumulative=False)
        max_points_err = max(all_points_err)
        plt.xticks(np.arange(0, max_points_err+0.01, 0.2))
        plt.yticks(np.arange(0, 1.01, 0.1))
        plt.xlabel('reprojection error (# pixels)')
        plt.ylabel('pdf')
        plt.title('total # of images: {}\ntotal # of sparse 3D points: {}\nreproj. err. (pixels): min {:.6f}, mean {:.6f}, median {:.6f}, max {:.6f}'
                  .format(self.img_cnt, len(all_points_err), min(all_points_err), np.mean(all_points_err), np.median(all_points_err), max_points_err))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'inspect_reproj_err.jpg'))
        plt.close()
        #plt.show()


def test():
    # sparse_dir = '/data2/kz298/bak/data_aoi-d1-wpafb_pinhole/sparse_ba'
    # sparse_dir = '/data2/kz298/bak/data_aoi-d2-wpafb_pinhole/sparse_ba'
    # sparse_dir = '/data2/kz298/bak/data_aoi-d3-ucsd_pinhole/sparse_ba'
    # sparse_dir = '/data2/kz298/bak/data_aoi-d4-jacksonville_pinhole/sparse_ba'


    # sparse_dir = '/data2/kz298/core3d_result/aoi-d1-wpafb/colmap/sparse_ba/'
    # out_dir = '/data2/kz298/core3d_result/aoi-d1-wpafb/colmap/inspect/'

    #sparse_dir = '/data2/kz298/core3d_result/aoi-d2-wpafb/colmap/sparse_ba/'
    #out_dir = '/data2/kz298/core3d_result/aoi-d2-wpafb/colmap/inspect/'

    # sparse_dir = '/data2/kz298/core3d_result/aoi-d3-ucsd/colmap/sparse_ba/'
    # out_dir = '/data2/kz298/core3d_result/aoi-d3-ucsd/colmap/inspect/'

    # sparse_dir = '/data2/kz298/core3d_result/aoi-d1-wpafb/colmap/sparse/'
    # out_dir = '/data2/kz298/core3d_result/aoi-d1-wpafb/colmap/inspect/'

    sparse_dir = '/data2/kz298/core3d_result/aoi-d3-ucsd/colmap/sparse/'
    out_dir = '/data2/kz298/core3d_result/aoi-d3-ucsd/colmap/inspect/'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # sparse_dir = '/data2/kz298/core3d_aoi/aoi-d4-jacksonville-overlap/colmap/sparse_ba/'
    # out_dir = '/data2/kz298/core3d_aoi/aoi-d4-jacksonville-overlap/colmap/inspect/sparse/'
    sparse_inspector = InspectSparseModel(sparse_dir, out_dir)
    sparse_inspector.inspect()
    sparse_inspector.inspect_depth_range()

if __name__ == '__main__':
    test()