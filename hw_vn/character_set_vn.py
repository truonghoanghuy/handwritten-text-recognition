import json
import sys
from collections import defaultdict


def load_char_set(char_set_path):
    with open(char_set_path) as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k, v in iter(char_set['idx_to_char'].items()):
        idx_to_char[int(k)] = v

    return idx_to_char, char_set['char_to_idx']


if __name__ == "__main__":
    character_set_path = sys.argv[-1]
    out_char_to_idx = {}
    out_idx_to_char = {}
    char_freq = defaultdict(int)
    for i in range(1, len(sys.argv) - 1):
        data_file = sys.argv[i]
        with open(data_file) as f:
            paths = json.load(f)

        for image_path, txt_path in paths:
            with open(txt_path, 'r', encoding='utf8') as f:
                data = f.read()

            cnt = 1  # this is important that this starts at 1 not 0
            for c in data:
                if c is None:
                    print("There was a None GT")
                    continue
                if c not in out_char_to_idx:
                    out_char_to_idx[c] = cnt
                    out_idx_to_char[cnt] = c
                    cnt += 1
                char_freq[c] += 1

    # Dataset line level lacks a character 'Z'
    if 'Z' not in out_char_to_idx:
        out_char_to_idx['Z'] = 1

    out_char_to_idx2 = {}
    out_idx_to_char2 = {}

    for i, c in enumerate(sorted(out_char_to_idx.keys())):
        out_char_to_idx2[c] = i + 1
        out_idx_to_char2[i + 1] = c

    output_data = {
        "char_to_idx": out_char_to_idx2,
        "idx_to_char": out_idx_to_char2
    }

    for k, v in sorted(char_freq.items(), key=lambda x: x[1]):
        print(k, v)

    print("Size:", len(output_data['char_to_idx']))

    with open(character_set_path, 'w', encoding='utf8') as outfile:
        json.dump(output_data, outfile, ensure_ascii=False)