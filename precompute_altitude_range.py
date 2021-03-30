from lib.parse_meta import parse_meta
import glob
import os


clean_dir = '/phoenix/S7/IARPA-SMART/datadump-1/KR-Pyeongchan/mvs_results/clean_data/'

min_alts = []
max_alts = []
for xml_file in glob.glob(os.path.join(clean_dir, '*.XML')):
    try:
        meta_dict = parse_meta(xml_file)
        rpc = meta_dict['rpc']
        min_val =  -rpc['altScale'] + rpc['altOff']
        max_val = rpc['altScale'] + rpc['altOff']
        
        print(min_val, max_val)
        min_alts.append(min_val)
        max_alts.append(max_val)
    except:
        print(xml_file)


print('final:', max(min_alts), min(max_alts))

