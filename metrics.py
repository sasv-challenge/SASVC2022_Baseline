from typing import List, Union

import numpy
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve


def get_all_EERs(
    preds: Union[torch.Tensor, List, numpy.ndarray], keys: List
) -> List[float]:
    """
    Calculate all three EERs used in the SASV Challenge 2022.
    preds and keys should be pre-calculated using dev or eval protocol in
    either 'protocols/ASVspoof2019.LA.asv.dev.gi.trl.txt' or
    'protocols/ASVspoof2019.LA.asv.eval.gi.trl.txt'

    :param preds: list of scores in tensor
    :param keys: list of keys where each element should be one of
    ['target', 'nontarget', 'spoof']
    """
    sasv_labels, sv_labels, spf_labels = [], [], []
    sv_preds, spf_preds = [], []

    for pred, key in zip(preds, keys):
        if key == "target":
            sasv_labels.append(1)
            sv_labels.append(1)
            spf_labels.append(1)
            sv_preds.append(pred)
            spf_preds.append(pred)

        elif key == "nontarget":
            sasv_labels.append(0)
            sv_labels.append(0)
            sv_preds.append(pred)

        elif key == "spoof":
            sasv_labels.append(0)
            spf_labels.append(0)
            spf_preds.append(pred)
        else:
            raise ValueError(
                f"should be one of 'target', 'nontarget', 'spoof', got:{key}"
            )

    fpr, tpr, _ = roc_curve(sasv_labels, preds, pos_label=1)
    sasv_eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    fpr, tpr, _ = roc_curve(sv_labels, sv_preds, pos_label=1)
    sv_eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    fpr, tpr, _ = roc_curve(spf_labels, spf_preds, pos_label=1)
    spf_eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    return sasv_eer, sv_eer, spf_eer
