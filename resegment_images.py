import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from e2e import alignment_dataset
from e2e.alignment_dataset import AlignmentDataset
from utils import transformation_utils
from utils.dataset_parse import load_file_list


def main():
    config_path = sys.argv[1]

    with open(config_path) as f:
        config = yaml.load(f)

    for dataset_lookup in ['training_set', 'validation_set']:
        set_list = load_file_list(config['training'][dataset_lookup])
        dataset = AlignmentDataset(set_list, None)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8,
                                collate_fn=alignment_dataset.collate)

        output_grid_size = config['network']['hw']['input_height']
        t = ((np.arange(output_grid_size) + 0.5) / float(output_grid_size))[:, None].astype(np.float32)
        t = np.repeat(t, axis=1, repeats=output_grid_size)
        t = torch.from_numpy(t)
        s = t.t()

        t = t[:, :, None]
        s = s[:, :, None]

        # noinspection PyTypeChecker
        interpolations = torch.cat([
            (1 - t) * s,
            (1 - t) * (1 - s),
            t * s,
            t * (1 - s),
        ], dim=-1)

        for L in dataloader:
            for l_i in L:
                img = l_i['full_img']
                renorm_matrix = transformation_utils.compute_renorm_matrix(img)[None, ...]

                print(l_i['img_key'])

                all_lf_paths = defaultdict(list)
                for j, item in enumerate(l_i['gt_json']):

                    if 'lf' not in item or 'hw_path' not in item:
                        continue

                    # start = time.time()
                    hw_path = item['hw_path']

                    lf_pts = item['lf']
                    if 'after_lf' in item:
                        lf_pts += item['after_lf']

                    lf_path = []
                    for i, step in enumerate(lf_pts):
                        x0 = step['x0']
                        x1 = step['x1']
                        y0 = step['y0']
                        y1 = step['y1']

                        pt = torch.tensor([[x1, x0], [y1, y0], [1, 1]])[None, ...]
                        lf_path.append(pt)

                    all_lf_paths[len(lf_path)].append((lf_path, hw_path))

                for cnt, pairs in iter(all_lf_paths.items()):
                    lf_paths = [p[0] for p in pairs]
                    hw_paths = [p[1] for p in pairs]

                    to_join = [[] for _ in range(cnt)]
                    for lf_path in lf_paths:
                        for i in range(len(lf_path)):
                            to_join[i].append(lf_path[i])

                    # for i in range(len(to_join)):
                    #     to_join[i] = torch.cat(to_join[i], dim=0)
                    to_join = [torch.cat(x, dim=0) for x in to_join]

                    lf_path = to_join

                    grid_line = []
                    for i in range(0, len(lf_path) - 1):
                        pts_0 = lf_path[i]
                        pts_1 = lf_path[i + 1]
                        pts = torch.cat([pts_0, pts_1], dim=2)

                        grid_pts = renorm_matrix.matmul(pts)

                        grid = interpolations[None, :, :, None, :] * grid_pts[:, None, None, :, :]
                        grid = grid.sum(dim=-1)[..., :2]

                        grid_line.append(grid)

                    grid_line = torch.cat(grid_line, dim=1)

                    expand_img = img.expand(grid_line.size(0), img.size(1), img.size(2), img.size(3))

                    line = torch.nn.functional.grid_sample(expand_img.transpose(2, 3), grid_line, align_corners=True)
                    line = line.transpose(2, 3)

                    for line_i, line_i_path in zip(line, hw_paths):
                        line_i = line_i.transpose(0, 1).transpose(1, 2)
                        line_i = (line_i + 1) * 128
                        l_np = line_i.data.cpu().numpy()

                        cv2.imwrite(line_i_path, l_np)


if __name__ == "__main__":
    main()
