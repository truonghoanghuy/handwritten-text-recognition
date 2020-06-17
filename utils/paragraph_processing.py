import numpy as np


def softmax(x):
    m = np.expand_dims(np.max(x, axis=-1), -1)
    e = np.exp(x - m)
    return e / np.expand_dims(e.sum(axis=-1), -1)


def combine_lines_into_paragraph(paragraphs, space_idx, len_decoder):
    modified_paragraph = paragraphs[0]

    for line in paragraphs[1:]:
        space_arr = np.zeros(len_decoder)
        space_arr[space_idx] = 1
        modified_paragraph = np.vstack((modified_paragraph, space_arr))
        modified_paragraph = np.vstack((modified_paragraph, line))

    return modified_paragraph
