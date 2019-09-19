import os

top_dir = '/data2/kz298/core3d_result'
csv_to_write = '/data2/kz298/core3d_result/results.csv'

metric_items = ['best_dx (cells)', 'best_dy (cells)', 'best_dz (meters)',
                'best_median_err (meters)', 'completeness (<1 meter)', 'rmse (meters)']

aggregate_3d_results = []
aggregate_3d_results.append('AOI name, ' + ', '.join(metric_items) + '\n')

for subdir in sorted(os.listdir(top_dir)):
    if 'aoi-d' not in subdir:
        continue
    print('collecting {}'.format(subdir))
    aoi_name = subdir
    subdir = os.path.join(top_dir, subdir)

    eval_file = os.path.join(subdir, 'mvs_results/aggregate_3d/evaluate/offset.txt')
    if not os.path.exists(eval_file):
        continue

    with open(eval_file) as fp:
        all_lines = fp.readlines()
        cnt = 0
        values = []
        for line in all_lines[2:]:  # skip the first two lines
            line = line.strip()
            if line:
                line = line.split(':')
                assert(metric_items[cnt] in line[0])
                values.append(line[1])
                cnt += 1
        assert(cnt == len(metric_items))
        aggregate_3d_results.append(aoi_name + ', ' + ', '.join(values) + '\n')


aggregate_2p5d_results = []
aggregate_2p5d_results.append('AOI name, ' + ', '.join(metric_items) + '\n')
for subdir in sorted(os.listdir(top_dir)):
    if 'aoi-d' not in subdir:
        continue
    print('collecting {}'.format(subdir))
    aoi_name = subdir
    subdir = os.path.join(top_dir, subdir)

    eval_file = os.path.join(subdir, 'mvs_results/aggregate_2p5d/evaluate/offset.txt')
    if not os.path.exists(eval_file):
        continue

    with open(eval_file) as fp:
        all_lines = fp.readlines()
        cnt = 0
        values = []
        for line in all_lines[2:]:  # skip the first two lines
            line = line.strip()
            if line:
                line = line.split(':')
                assert(metric_items[cnt] in line[0])
                values.append(line[1])
                cnt += 1
        assert(cnt == len(metric_items))
        aggregate_2p5d_results.append(aoi_name + ', ' + ', '.join(values) + '\n')

with open(csv_to_write, 'w') as fp:
    fp.write('3D Aggregation\n')
    fp.write(''.join(aggregate_3d_results))

    fp.write('\n\n')
    fp.write('2.5D Aggregation\n')
    fp.write(''.join(aggregate_2p5d_results))

