import os
import csv
import json
from lib.logger import GlobalLogger
import logging
import numpy as np


if __name__ == '__main__':
    # work_dirs = ['/data2/kz298/mvs3dm_result/Explorer',
    #             '/data2/kz298/mvs3dm_result/MasterProvisional1',
    #             '/data2/kz298/mvs3dm_result/MasterProvisional2',
    #             '/data2/kz298/mvs3dm_result/MasterProvisional3',
    #             '/data2/kz298/mvs3dm_result/MasterSequestered1',
    #             '/data2/kz298/mvs3dm_result/MasterSequestered2',
    #             '/data2/kz298/mvs3dm_result/MasterSequestered3',
    #             '/data2/kz298/mvs3dm_result/MasterSequesteredPark']
    work_dirs = ['/data2/kz298/mvs3dm_result/MasterSequestered1',
                '/data2/kz298/mvs3dm_result/MasterSequestered2',
                '/data2/kz298/mvs3dm_result/MasterSequestered3',
                '/data2/kz298/mvs3dm_result/MasterSequesteredPark']

    logger = GlobalLogger()
    logger.turn_on_terminal()

    with open('/data2/kz298/mvs3dm_result/all_result.csv', 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow(['aoi_name', 'completeness', 'median_error', 'rmse'])
        completeness = []
        median_error = []
        rmse = []
        for work_dir in work_dirs:
            with open(os.path.join(work_dir, 'evaluation/eval_point_cloud.ply_results.json')) as fp:
                result = json.load(fp)['entries']
            aoi_name = work_dir[work_dir.rfind('/')+1:]

            logging.info('aoi_name: {}, completeness: {}, median_error: {}, rmse: {}'
                         .format(aoi_name, result['completeness'], result['error_median'], result['error_rms']))
            csv_writer.writerow([aoi_name, result['completeness'], result['error_median'], result['error_rms']])

            completeness.append(result['completeness'])
            median_error.append(result['error_median'])
            rmse.append(result['error_rms'])

        csv_writer.writerow(['Avg.',  np.mean(completeness), np.mean(median_error), np.mean(rmse)])
        logging.info('Avg. completeness: {}, median_error: {}, rmse: {}'.format( np.mean(completeness), np.mean(median_error), np.mean(rmse)))