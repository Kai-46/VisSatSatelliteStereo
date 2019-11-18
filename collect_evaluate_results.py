#  ===============================================================================================================
#  Copyright (c) 2019, Cornell University. All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that
#  the following conditions are met:
#
#      * Redistributions of source code must retain the above copyright otice, this list of conditions and
#        the following disclaimer.
#
#      * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and
#        the following disclaimer in the documentation and/or other materials provided with the distribution.
#
#      * Neither the name of Cornell University nor the names of its contributors may be used to endorse or
#        promote products derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
#  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
#  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
#  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
#  OF SUCH DAMAGE.
#
#  Author: Kai Zhang (kz298@cornell.edu)
#
#  The research is based upon work supported by the Office of the Director of National Intelligence (ODNI),
#  Intelligence Advanced Research Projects Activity (IARPA), via DOI/IBC Contract Number D17PC00287.
#  The U.S. Government is authorized to reproduce and distribute copies of this work for Governmental purposes.
#  ===============================================================================================================


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

