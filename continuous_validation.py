import copy
import itertools
import json
import operator
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from e2e import alignment_dataset, e2e_postprocessing
from e2e import validation_utils
from e2e.alignment_dataset import AlignmentDataset
from e2e.e2e_model import E2EModel
from utils import error_rates, printer, continuous_state
from utils.continuous_training_util import ContinuousTrainingUtil
from utils.dataset_parse import load_file_list
from utils.dataset_wrapper import DatasetWrapper


def alignment_step(config, idx_to_char, loader, is_validation_set=True, baseline=False):
    network_config = config['network']
    checkpoint_dir = config['training']['snapshot']['best_validation']
    if is_validation_set:
        json_dir = config['training']['validation_set']['json_folder']
    else:
        json_dir = config['training']['training_set']['json_folder']
    post_processing_config = config['training']['alignment']['validation_post_processing']
    sol_thresholds = post_processing_config['sol_thresholds']
    sol_thresholds_indices = range(len(sol_thresholds))
    lf_nms_ranges = post_processing_config['lf_nms_ranges']
    lf_nms_ranges_indices = range(len(lf_nms_ranges))
    lf_nms_thresholds = post_processing_config['lf_nms_thresholds']
    lf_nms_thresholds_indices = range(len(lf_nms_thresholds))

    if baseline:
        sol, lf, hw = continuous_state.init_model(config)
    else:
        sol, lf, hw = continuous_state.load_model_from_checkpoint(network_config, checkpoint_dir)
    e2e = E2EModel(sol, lf, hw)
    e2e.eval()

    results = defaultdict(list)
    aligned_results = []
    best_ever_results = []
    progress_printer = printer.ProgressBarPrinter(len(loader))
    progress_printer.start()
    for batch in loader:
        input = batch[0]
        if input is None:
            progress_printer.step(skip=True)
            continue
        gt_json = input['gt_json']
        gt_lines = input['gt_lines']
        gt = '\n'.join(gt_lines)

        with torch.no_grad():
            out_original = e2e(input, lf_batch_size=config['network']['lf']['batch_size'])

        out_original = e2e_postprocessing.results_to_numpy(out_original)
        out_original['idx'] = np.arange(out_original['sol'].shape[0])
        e2e_postprocessing.trim_ends(out_original)
        decoded_hw, decoded_raw_hw = e2e_postprocessing.decode_handwriting(out_original, idx_to_char)
        pick, costs = e2e_postprocessing.align_to_gt_lines(decoded_hw, gt_lines)

        best_ever_pred_lines, improved_idxs = validation_utils.update_ideal_results(pick, costs, decoded_hw, gt_json)
        validation_utils.save_improved_idxs(improved_idxs, decoded_hw, decoded_raw_hw, out_original, input, json_dir)

        best_ever_pred_lines = "\n".join(best_ever_pred_lines)
        error = error_rates.cer(gt, best_ever_pred_lines)
        best_ever_results.append(error)

        aligned_pred_lines = [decoded_hw[i] for i in pick]
        aligned_pred_lines = "\n".join(aligned_pred_lines)
        error = error_rates.cer(gt, aligned_pred_lines)
        aligned_results.append(error)

        if is_validation_set:
            # We only care about the hyper-parameter postprocessing search for the validation set
            for key in itertools.product(sol_thresholds_indices, lf_nms_ranges_indices, lf_nms_thresholds_indices):
                i, j, k = key
                sol_threshold = sol_thresholds[i]
                lf_nms_params = {
                    'overlap_range': lf_nms_ranges[j],
                    'overlap_threshold': lf_nms_thresholds[k]
                }
                out = copy.copy(out_original)
                out = e2e_postprocessing.postprocess(out, sol_threshold=sol_threshold, lf_nms_params=lf_nms_params)
                order = e2e_postprocessing.read_order(out)
                e2e_postprocessing.filter_on_pick(out, order)
                e2e_postprocessing.trim_ends(out)
                predicts = [decoded_hw[i] for i in out['idx']]
                predict = '\n'.join(predicts)
                error = error_rates.cer(gt, predict)
                results[key].append(error)

        torch.cuda.empty_cache()
        progress_printer.step()

    sum_results = None
    if is_validation_set:
        # Skipping because we didn't do the hyper-parameter search
        sum_results = {k: np.mean(v) for k, v in results.items()}
        sum_results = min(sum_results.items(), key=operator.itemgetter(1))

    return sum_results, np.mean(aligned_results), np.mean(best_ever_results), sol, lf, hw


def main():
    init_mode = 'init' in sys.argv
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = yaml.load(f)
    char_set_path = config['network']['hw']['char_set_path']
    with open(char_set_path) as f:
        char_set = json.load(f)
    idx_to_char = {int(k): v for k, v in char_set['idx_to_char'].items()}

    train_set_list = load_file_list(config['training']['training_set'])
    train_dataset = AlignmentDataset(train_set_list, None)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0,
                                  collate_fn=alignment_dataset.collate)
    train_dataloader = DatasetWrapper(train_dataloader, config['training']['alignment']['validate_after'])
    eval_set_list = load_file_list(config['training']['validation_set'])
    eval_dataset = AlignmentDataset(eval_set_list, None)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0,
                                 collate_fn=alignment_dataset.collate)

    print('Running validation with best overall weight for baseline')
    error, i_error, mi_error, _, _, _ = alignment_step(config, idx_to_char, eval_dataloader, baseline=True)
    print(f'Baseline Validation = {error[1]}, (sol_threshold_id, lf_nms_range_id, lf_nms_threshold_id) = {error[0]}')
    best_cer = error[1]

    total_time = 0
    no_improvement_count = 0
    while True:
        for phase in ['validation', 'train']:
            print()
            start_time = time.time()
            is_validation_set = (phase == 'validation')
            if is_validation_set:
                print(f'Alignment on the validation set')
                loader = train_dataloader
            else:
                print(f'Alignment on the next {len(train_dataloader)} samples of the training set')
                loader = eval_dataloader
            error, i_error, mi_error, sol, lf, hw = alignment_step(config, idx_to_char, loader, is_validation_set)
            phase_time = time.time() - start_time
            total_time += phase_time
            print(f'CER = {error} -- Current best CER: {best_cer}')
            print(f'Time elapsed = {total_time} -- Last step time = {phase_time}')
            if init_mode:
                return
            if is_validation_set:
                no_improvement_count += 1
                if error[1] < best_cer:
                    no_improvement_count = 0
                    save_model(sol, lf, hw, config['training']['snapshot']['best_overall'])
                    best_cer = error[1]
                if no_improvement_count > config['training']['alignment']['stop_after_no_improvement']:
                    return


def save_model(sol, lf, hw, dirname):
    print('Better CER. Saving...')
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    save_path = os.path.join(dirname, 'sol.pt')
    torch.save(sol.state_dict(), save_path)
    save_path = os.path.join(dirname, 'lf.pt')
    torch.save(lf.state_dict(), save_path)
    save_path = os.path.join(dirname, 'hw.pt')
    torch.save(hw.state_dict(), save_path)


if __name__ == "__main__":
    sys.stdout.flush()
    ContinuousTrainingUtil.start()
    main()
    ContinuousTrainingUtil.stop()
