from beamsearch_with_lm.ctc_decoders import ctc_beam_search_decoder, Scorer

pathVocab = 'data/decoding/chars.txt'
vocab = list(open(pathVocab, encoding='utf8').read())

lm = 'data/decoding/lm_5grams_probing.binary'
alpha = 2
beta = 1
scorer = Scorer(alpha, beta, model_path=lm, vocabulary=vocab)


def beam_search_with_lm(in_mat, beam_size=25):
    res = ctc_beam_search_decoder(in_mat, vocab, beam_size, ext_scoring_func=scorer)
    return res[0][1]
