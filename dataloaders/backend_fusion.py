import random
from typing import Dict, List

from torch.utils.data import Dataset


class SASV_Trainset(Dataset):
    def __init__(self, cm_embd, asv_embd, spk_meta):
        self.cm_embd = cm_embd
        self.asv_embd = asv_embd
        self.spk_meta = spk_meta

    def __len__(self):
        return len(self.cm_embeds.keys())

    def __getitem__(self, index):

        ans_type = random.randint(0, 1)
        if ans_type == 1:  # target
            spk = random.choice(list(self.spk_meta.keys()))
            enr, tst = random.sample(self.spk_meta[spk]["bonafide"], 2)

        elif ans_type == 0:  # nontarget
            nontarget_type = random.randint(1, 2)

            if nontarget_type == 1:  # zero-effort nontarget
                spk, ze_spk = random.sample(self.spk_meta.keys(), 2)
                enr = random.choice(self.spk_meta[spk]["bonafide"])
                tst = random.choice(self.spk_meta[ze_spk]["bonafide"])

            if nontarget_type == 2:  # spoof nontarget
                spk = random.choice(list(self.spk_dic.keys()))
                if len(self.spk_meta[spk]["spoof"]) == 0:
                    while True:
                        spk = random.choice(list(self.spk_meta.keys()))
                        if len(self.spk_meta[spk]["spoof"]) != 0:
                            break
                enr = random.choice(self.spk_meta[spk]["bonafide"])
                tst = random.choice(self.spk_meta[spk]["spoof"])

        return self.asv_embd[enr], self.asv_embd[tst], self.cm_embd[tst], ans_type


class SASV_DevEvalset(Dataset):
    def __init__(self, utt_list, spk_model, asv_embd, cm_embd):
        self.utt_list = utt_list
        self.spk_model = spk_model
        self.asv_embd = asv_embd
        self.cm_embd = cm_embd

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        line = self.utt_list[index]
        spkmd, key, _, _ = line.strip().split(" ")

        return self.spk_model[spkmd], self.asv_embd[key], self.cm_embd[key], key


def get_trnset(
    cm_embd_trn: Dict, asv_embd_trn: Dict, spk_meta_trn: Dict
) -> SASV_DevEvalset:
    return SASV_Trainset(
        cm_embd=cm_embd_trn, asv_embd=asv_embd_trn, spk_meta=spk_meta_trn
    )


def get_dev_evalset(
    utt_list: List, cm_embd: Dict, asv_embd: Dict, spk_model: Dict
) -> SASV_DevEvalset:
    return SASV_DevEvalset(
        utt_list=utt_list, cm_embd=cm_embd, asv_embd=asv_embd, spk_model=spk_model
    )
