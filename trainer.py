import argparse
from tqdm import tqdm
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    args: argparse.Namespace,
):

    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for asv_enr, asv_tst, cm_tst, ans in trn_loader:
        num_total += args.batch_size
        ii += 1
        asv_enr, asv_tst, cm_tst = asv_enr.to(device), asv_tst.to(device), cm_tst.to(device)
        batch_y = ans.type(torch.int64).to(device)
        batch_out = model(asv_enr, asv_tst, cm_tst)
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * args.batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if args.lr_decay in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


# Note that this is the trivial modified version of original code
def produce_evaluation_file(
    data_loader: DataLoader, model, device: torch.device, save_path: str, epoch, trial_lines, mode
) -> None:

    model.eval()

    fname_list = []
    score_list = []

    for asv_enr, asv_tst, cm_tst, key in data_loader:
        with torch.no_grad():
            asv_enr, asv_tst, cm_tst = asv_enr.to(device), asv_tst.to(device), cm_tst.to(device)
            batch_out = model(asv_enr, asv_tst, cm_tst)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()

            fname_list.extend(key)
            score_list.extend(batch_score)

    assert len(trial_lines) == len(score_list)

    SASV_labels, SV_labels, CM_labels = [], [], []
    SV_scores, CM_scores = [], []

    if mode == "eval":
        fs = open(save_path + "{}_e{}_score.txt".format(mode, epoch), "w")

    for fn, sco, trl in zip(fname_list, score_list, trial_lines):
        _, utt_id, src, key = trl.strip().split(" ")  # asv

        assert fn == utt_id
        if key == "target":
            SASV_labels.append(1)
            SV_labels.append(1)
            SV_scores.append(sco)
            CM_labels.append(1)
            CM_scores.append(sco)

        elif key == "nontarget":
            SASV_labels.append(0)
            SV_labels.append(0)
            SV_scores.append(sco)

        elif key == "spoof":
            SASV_labels.append(0)
            CM_labels.append(0)
            CM_scores.append(sco)

        if mode == "eval":
            fs.write("{} {} {} {}\n".format(utt_id, src, key, sco))

    fpr, tpr, _ = roc_curve(SASV_labels, score_list, pos_label=1)
    SASV_EER = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    fpr, tpr, _ = roc_curve(SV_labels, SV_scores, pos_label=1)
    SPF_EER = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    fpr, tpr, _ = roc_curve(CM_labels, CM_scores, pos_label=1)
    SV_EER = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)

    return SASV_EER, SPF_EER, SV_EER
