import numpy as np
from PIL import Image, ImageDraw


CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']

# def _softmax(x):
#     return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), axis=-1)


def ocr_postprocessing(endnodes, device_pre_post_layers=None, **kwargs):
    logits = np.mean(endnodes[0], axis=1)
    # probs = _softmax(logits)
    return {'predictions': np.argmax(logits, axis=2)}


def visualize_ocr_result(probs, img, text_color=(255, 0, 0), **kwrgs):
    probs = np.expand_dims(probs['predictions'][0], axis=0)
    pred = greedy_decoder(probs)[0]
    pred = ''.join([CHARS[int(x)] for x in pred])
    img_orig = Image.fromarray(img[0])
    ImageDraw.Draw(img_orig).text((0, 0), "{}".format(pred), text_color)
    return np.array(img_orig, np.uint8)


def greedy_decoder(probs):
    predictions = list()
    for prob in probs:
        no_repeat_blank_label = list()
        pre_c = prob[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
        for c in prob:  # dropout repeate label and blank label
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            pre_c = c
        predictions.append(no_repeat_blank_label)
    return predictions
