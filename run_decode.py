import argparse
import codecs
import json
import os
import cv2
import numpy as np
import torch
import yaml

from e2e import e2e_postprocessing
from e2e import visualization
from utils import PAGE_xml
from utils.continuous_state import init_model
from utils.paragraph_processing import softmax, combine_lines_into_paragraph
from utils import string_utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    parser.add_argument('npz_folder')
    parser.add_argument('--in_xml_folder')
    parser.add_argument('--out_xml_folder')
    parser.add_argument('--best_path', action='store_true')  # use best path decoding, default is beam search decoding
    parser.add_argument('--combine', action='store_true')  # combine all lines in paragraph before decoding
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--roi', action='store_true')
    args = parser.parse_args()

    config_path = args.config_path
    npz_folder = args.npz_folder
    in_xml_folder = args.in_xml_folder
    out_xml_folder = args.out_xml_folder

    in_xml_files = {}
    if in_xml_folder and out_xml_folder:
        for root, folders, files in os.walk(in_xml_folder):
            for f in files:
                if f.endswith(".xml"):
                    basename = os.path.basename(f).replace(".xml", "")
                    in_xml_files[basename] = os.path.join(root, f)

    use_best_path = args.best_path
    combine_lines = args.combine
    use_aug = args.aug
    use_roi = args.roi

    if not use_best_path:
        from hw_vn.beam_search_with_lm import beam_search_with_lm

    with open(config_path) as f:
        config = yaml.load(f)

    npz_paths = []
    for root, folder, files in os.walk(npz_folder):
        for f in files:
            if f.lower().endswith(".npz"):
                npz_paths.append(os.path.join(root, f))

    char_set_path = config['network']['hw']['char_set_path']

    with open(char_set_path, encoding='utf8') as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k, v in iter(char_set['idx_to_char'].items()):
        idx_to_char[int(k)] = v

    char_to_idx = char_set['char_to_idx']

    decoder, hw = None, None

    if use_aug:
        model_mode = "best_overall"
        _, _, hw = init_model(config, hw_dir=model_mode, only_load="hw")
        hw.eval()

    for npz_path in sorted(npz_paths):
        out = np.load(npz_path)
        out = dict(out)

        image_path = str(out['image_path'])
        print(image_path)
        org_img = cv2.imread(image_path)

        # Postprocessing Steps
        out['idx'] = np.arange(out['sol'].shape[0])
        out = e2e_postprocessing.trim_ends(out)

        e2e_postprocessing.filter_on_pick(out, e2e_postprocessing.select_non_empty_string(out))
        out = e2e_postprocessing.postprocess(out,
                                             sol_threshold=config['post_processing']['sol_threshold'],
                                             lf_nms_params={
                                                 "overlap_range": config['post_processing']['lf_nms_range'],
                                                 "overlap_threshold": config['post_processing']['lf_nms_threshold']
                                             }
                                             # },
                                             # lf_nms_2_params={
                                             #     "overlap_threshold": 0.5
                                             # }
                                             )
        order = e2e_postprocessing.read_order(out)
        e2e_postprocessing.filter_on_pick(out, order)

        output_strings = []

        # Decoding network output
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        space_idx = char_to_idx[' '] - 1
        len_decoder = len(char_to_idx) + 1

        if use_best_path:
            if combine_lines:
                out['hw'] = combine_lines(out['hw'], space_idx, len_decoder)
                pred, raw_pred = string_utils.naive_decode(out['hw'])
                pred_str = string_utils.label2str_single(pred, idx_to_char, False)
                output_strings.append(pred_str)
            output_strings, decoded_raw_hw = e2e_postprocessing.decode_handwriting(out, idx_to_char)
        else:
            paragraph = []
            for line in out['hw']:
                temp = np.copy(line[:, 0])
                for i in range(0, len(line[0]) - 1):
                    line[:, i] = line[:, i + 1]
                line[:, len(line[0]) - 1] = temp
                softmax_line = softmax(line)
                paragraph.append(softmax_line)

            if combine_lines:
                paragraph = combine_lines_into_paragraph(paragraph, space_idx, len_decoder)
                output_strings = [beam_search_with_lm(paragraph)]
            else:
                for line in paragraph:
                    s = beam_search_with_lm(line)
                    output_strings.append(s)

        draw_img = visualization.draw_output(out, org_img)
        cv2.imwrite(npz_path + ".png", draw_img)

        # Save results
        label_string = "_"
        if use_best_path:
            label_string += "bestpath_"
        else:
            label_string += 'lm_'
        if use_aug:
            label_string += "aug_"
        filepath = npz_path + label_string + ".txt"

        with codecs.open(filepath, 'w', encoding='utf-8') as f:
            f.write(u'\n'.join(output_strings))

        key = os.path.basename(image_path)[:-len(".jpg")]
        if in_xml_folder:
            if use_roi:

                key, region_id = key.split("_", 1)
                region_id = region_id.split(".")[0]

                if key in in_xml_files:
                    in_xml_file = in_xml_files[key]
                    out_xml_file = os.path.join(out_xml_folder, os.path.basename(in_xml_file))
                    PAGE_xml.create_output_xml_roi(in_xml_file, out, output_strings, out_xml_file, region_id)
                    in_xml_files[key] = out_xml_file  # after first, add to current xml
                else:
                    print("Couldn't find xml file for ", key)
            else:
                if key in in_xml_files:
                    in_xml_file = in_xml_files[key]
                    out_xml_file = os.path.join(out_xml_folder, os.path.basename(in_xml_file))
                    PAGE_xml.create_output_xml(in_xml_file, out, output_strings, out_xml_file)
                else:
                    print("Couldn't find xml file for ", key)
