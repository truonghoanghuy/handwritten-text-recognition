import numpy as np
import torch
import cv2

from e2e import e2e_postprocessing


def get_lines(image_path, e2e, config, mode='hw'):
    org_img = cv2.imread(image_path)
    # print(image_path, org_img.shape if isinstance(org_img, np.ndarray) else None)
    h, w = org_img.shape[:2]

    txt_path = image_path.split('.')[0] + '.txt'
    label = open(txt_path, encoding='utf8').read()
    len_label = torch.tensor([len(label)])

    target_height = 512
    scale = target_height / float(w + 0)
    pad_top = pad_bottom = 128
    pad_left = pad_right = 128
    pad_values = [np.argmax(cv2.calcHist([org_img], channels=[x], mask=None, histSize=[256], ranges=[0, 256]))
                  for x in range(3)]  # pad the mode value for each color channel
    padded_img = np.dstack([np.pad(org_img[:, :, x], ((pad_top, pad_bottom), (pad_left, pad_right)),
                                   constant_values=pad_values[x]) for x in range(3)])
    target_width = int(padded_img.shape[0] * scale)
    target_height = int(padded_img.shape[1] * scale)

    full_img = padded_img.astype(np.float32)
    full_img = full_img.transpose([2, 1, 0])[None, ...]  # (H, W, C) -> (1, C, W, H)
    full_img = torch.from_numpy(full_img)
    full_img = full_img / 128 - 1

    resized_img = cv2.resize(padded_img, (target_height, target_width), interpolation=cv2.INTER_CUBIC)
    resized_img = resized_img.astype(np.float32)
    resized_img = resized_img.transpose([2, 1, 0])[None, ...]
    resized_img = torch.from_numpy(resized_img)
    resized_img = resized_img / 128 - 1

    e2e_input = {
        'resized_img': resized_img,
        'full_img': full_img,
        'resize_scale': 1.0 / scale,
        'len_label': len_label,
    }
    try:
        with torch.no_grad():
            out = e2e.forward(e2e_input, mode=mode)
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            e2e.to_cpu()
            with torch.no_grad():
                out = e2e.forward(e2e_input, lf_batch_size=100, mode=mode)
            e2e.to_cuda()
        else:
            raise e
    out = e2e_postprocessing.results_to_numpy(out)

    if out is None:
        print('No results for image \"{}\"'.format(image_path))
        assert False

    # take into account the padding
    out['sol'][:, 0] -= pad_left
    out['sol'][:, 1] -= pad_top
    for line in out['lf']:
        line[:, 0, :2] -= pad_left
        line[:, 1, :2] -= pad_top

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
                                         )
    order = e2e_postprocessing.read_order(out)
    e2e_postprocessing.filter_on_pick(out, order)

    return out
