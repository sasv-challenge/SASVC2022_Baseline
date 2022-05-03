import os
import pickle as pk
import random
import sys
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn


def str_to_bool(val):
    """Convert a string representation of truth to true (1) or false (0).
    Copied from the python implementation distutils.utils.strtobool

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    >>> str_to_bool('YES')
    1
    >>> str_to_bool('FALSE')
    0
    """
    val = val.lower()
    if val in ("y", "yes", "t", "true", "on", "1"):
        return True
    if val in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError("invalid truth value {}".format(val))


def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def keras_decay(step, decay=0.0001):
    """Learning rate decay in Keras-style"""
    return 1.0 / (1.0 + decay * step)


def set_seed(args):
    """
    set initial seed for reproduction
    """

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = args.cudnn_deterministic_toggle
        torch.backends.cudnn.benchmark = args.cudnn_benchmark_toggle


def set_init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        try:
            m.bias.data.fill_(0.0001)
        except:
            pass
    elif isinstance(m, nn.BatchNorm1d):
        pass
    else:
        try:
            torch.nn.init.kaiming_normal_(m.weight, a=0.01)
        except:
            pass

def load_parameters(trg_state, path):
    loaded_state = torch.load(path, map_location=lambda storage, loc: storage)
    for name, param in loaded_state.items():
        origname = name
        if name not in trg_state:
            name = name.replace("module.", "")
            name = name.replace("speaker_encoder.", "")
            if name not in trg_state:
                print("%s is not in the model."%origname)
                continue
        if trg_state[name].size() != loaded_state[origname].size():
            print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, trg_state[name].size(), loaded_state[origname].size()))
            continue
        trg_state[name].copy_(param)


def find_gpus(nums=4, min_req_mem=None) -> str:
    """
    Allocates 'nums' GPUs that have the most free memory.
    Original source:
    https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/10

    :param nums: number of GPUs to find
    :param min_req_mem: required GPU memory (in MB)
    :return: string of GPU indices separated with comma
    """

    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp_free_gpus')
    with open('tmp_free_gpus', 'r', encoding="utf-8") as lines_txt:
        frees = lines_txt.readlines()
        idx_freememory_pair = [ (idx,int(x.split()[2]))
                              for idx,x in enumerate(frees) ]
    idx_freememory_pair.sort(key=lambda my_tuple:my_tuple[1],reverse=True)
    using_gpus = [str(idx_memory_pair[0])
                    for idx_memory_pair in idx_freememory_pair[:nums] ]

    # return error signal if minimum required memory is given and
    # at least one GPU does not have sufficient memory
    if min_req_mem is not None and \
        int(idx_freememory_pair[nums][1]) < min_req_mem:

        return -1

    using_gpus =  ','.join(using_gpus)
    print('using GPU idx: #', using_gpus)
    return using_gpus


def get_spkdic(cm_meta: str) -> Dict:
    l_cm_meta = open(cm_meta, "r").readlines()

    d_spk = {}
    # dictionary of speakers
    # d_spk : {
    #   'spk_id1':{
    #       'bonafide': [utt1, utt2],
    #       'spoof': [utt5]
    #   },
    #   'spk_id2':{
    #       'bonafide': [utt3, utt4, utt8],
    #       'spoof': [utt6, utt7]
    #   } ...
    # }

    for line in l_cm_meta:
        spk, filename, _, _, ans = line.strip().split(" ")
        if spk not in d_spk:
            d_spk[spk] = {}
            d_spk[spk]["bonafide"] = []
            d_spk[spk]["spoof"] = []

        if ans == "bonafide":
            d_spk[spk]["bonafide"].append(filename)
        elif ans == "spoof":
            d_spk[spk]["spoof"].append(filename)

    return d_spk


def generate_spk_meta(config) -> None:
    d_spk_train = get_spkdic(config.dirs.cm_trn_list)
    d_spk_dev = get_spkdic(config.dirs.cm_dev_list)
    d_spk_eval = get_spkdic(config.dirs.cm_eval_list)
    os.makedirs(config.dirs.spk_meta, exist_ok=True)

    # save speaker dictionaries
    with open(config.dirs.spk_meta + "spk_meta_trn.pk", "wb") as f:
        pk.dump(d_spk_train, f)
    with open(config.dirs.spk_meta + "spk_meta_dev.pk", "wb") as f:
        pk.dump(d_spk_dev, f)
    with open(config.dirs.spk_meta + "spk_meta_eval.pk", "wb") as f:
        pk.dump(d_spk_eval, f)
