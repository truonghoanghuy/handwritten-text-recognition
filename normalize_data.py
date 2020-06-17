import sys
import os
from numpy import load
import cv2
import yaml

import run_hwr
from e2e import e2e_postprocessing


def get_pick_and_filter(out, config):
    out = e2e_postprocessing.postprocess(out,
                                          pick_line_imgs=True,
                                          sol_threshold=config['post_processing']['sol_threshold'],
                                          lf_nms_params={
                                              "overlap_range": config['post_processing']['lf_nms_range'],
                                              "overlap_threshold": config['post_processing']['lf_nms_threshold']
                                          }
                                          )
    return out


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Your arguments is incorrect!')
        exit()

    input_path = sys.argv[1]

    config_path = sys.argv[2]
    with open(config_path) as f:
        config = yaml.load(f)

    output_path = sys.argv[3]
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    print(f'Start normalizing all image files in \"{input_path}\" ...', end='\n\n')
    run_hwr.process(input_path, config_path, output_path)
    print('\nDone.')
    print(f'Finished normalizing all image files saved in \"{output_path}\"')

    print()
    print(f'Start saving back all file images normalized to data directory \"{input_path}\" ...')
    for root, _, files in os.walk(output_path):
        print(f'Total {len(files)} file(s) will be processed')
        for file in files:
            if not file.endswith('.npz'):
                continue
            data = load(os.path.join(root, file))
            data = dict(data)
            data = get_pick_and_filter(data, config)

            line_img = data['line_imgs']
            if len(line_img) != 1:
                print(f'Image \"{file}\" is not normalized correctly. Detected {len(line_img)} line(s) in image.')
                if len(line_img) == 0:
                    continue

            img = line_img[0]
            img_path = os.path.join(os.path.normpath(str(data['image_path'])))
            cv2.imwrite(img_path, img)

    print('Done.')