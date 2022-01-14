import random
import pickle as pk

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def _get_spkdic(l_cm_meta):

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


class trainset_loader(Dataset):
    def __init__(self, cm_embeds, asv_embeds, spk_dic, mode="train"):
        self.cm_embeds = cm_embeds
        self.asv_embeds = asv_embeds
        self.spk_dic = spk_dic
        self.mode = mode

    def __len__(self):
        return len(self.cm_embeds.keys())

    def __getitem__(self, index):

        ans_type = random.randint(0, 1)
        if ans_type == 1:  # target
            spk = random.choice(list(self.spk_dic.keys()))
            enr, tst = random.sample(self.spk_dic[spk]["bonafide"], 2)

        elif ans_type == 0:  # nontarget
            nontarget_type = random.randint(1, 2)

            if nontarget_type == 1:  # zero-effort nontarget
                spk, ze_spk = random.sample(self.spk_dic.keys(), 2)
                enr = random.choice(self.spk_dic[spk]["bonafide"])
                tst = random.choice(self.spk_dic[ze_spk]["bonafide"])

            if nontarget_type == 2:  # spoof nontarget
                spk = random.choice(list(self.spk_dic.keys()))
                if len(self.spk_dic[spk]["spoof"]) == 0:
                    while True:
                        spk = random.choice(list(self.spk_dic.keys()))
                        if len(self.spk_dic[spk]["spoof"]) != 0:
                            break
                enr = random.choice(self.spk_dic[spk]["bonafide"])
                tst = random.choice(self.spk_dic[spk]["spoof"])

        return self.asv_embeds[enr][0], self.asv_embeds[tst][0], self.cm_embeds[tst], ans_type


# for asvspoof asv evaluation
class testset_loader(Dataset):
    def __init__(self, list_IDs, spk_model, asv_embeds, cm_embeds):
        self.list_IDs = list_IDs
        self.spk_model = spk_model
        self.asv_embeds = asv_embeds
        self.cm_embeds = cm_embeds

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        line = self.list_IDs[index]
        spkmd, key, _, _ = line.strip().split(" ")

        return self.spk_model[spkmd], self.asv_embeds[key][0], self.cm_embeds[key], key


def get_loader(args):

    if args.save_spkdic:
        with open(
            args.meta_path + "ASVspoof2019.LA.cm.train.trn.txt", "r"
        ) as f:
            l_train_trial = f.readlines()
        with open(
            args.meta_path + "ASVspoof2019.LA.cm.dev.trl.txt", "r"
        ) as f:
            l_dev_trial = f.readlines()
        with open(
            args.meta_path + "ASVspoof2019.LA.cm.eval.trl.txt", "r"
        ) as f:
            l_eval_trial = f.readlines()

        d_spk_train = _get_spkdic(l_train_trial)
        d_spk_dev = _get_spkdic(l_dev_trial)
        d_spk_eval = _get_spkdic(l_eval_trial)

        # save speaker dictionaries
        with open(args.save_path + "d_spk_train.pk", "wb") as f:
            pk.dump(d_spk_train, f)
        with open(args.save_path + "d_spk_dev.pk", "wb") as f:
            pk.dump(d_spk_dev, f)
        with open(args.save_path + "d_spk_eval.pk", "wb") as f:
            pk.dump(d_spk_eval, f)
    else:
        with open(args.save_path + "d_spk_train.pk", "rb") as f:
            d_spk_train = pk.load(f)
        with open(args.save_path + "d_spk_dev.pk", "rb") as f:
            d_spk_dev = pk.load(f)
        with open(args.save_path + "d_spk_eval.pk", "rb") as f:
            d_spk_dev = pk.load(f)

    # load saved countermeasures(CM) related preparations
    with open(args.cm_embd_path + "train_embeds.pk", "rb") as f:
        d_cm_train = pk.load(f)
    with open(args.cm_embd_path + "dev_embeds.pk", "rb") as f:
        d_cm_dev = pk.load(f)
    with open(args.cm_embd_path + "eval_embeds.pk", "rb") as f:
        d_cm_eval = pk.load(f)

    # load saved automatic speaker verification(ASV) related preparations
    with open(args.asv_embd_path + "train_embeds.pk", "rb") as f:
        d_asv_train = pk.load(f)
    with open(args.asv_embd_path + "dev_embeds.pk", "rb") as f:
        d_asv_dev = pk.load(f)
    with open(args.asv_embd_path + "eval_embeds.pk", "rb") as f:
        d_asv_eval = pk.load(f)

    # check embeddings and dictionary for all datasets
    print(
        "cm_embeddings: train/dev/eval ",
        len(d_cm_train.keys()),
        len(d_cm_dev.keys()),
        len(d_cm_eval.keys()),
    )
    print(
        "asv_embeddings: train/dev/eval ",
        len(d_asv_train.keys()),
        len(d_asv_dev.keys()),
        len(d_asv_eval.keys()),
    )
    print("d_spk", len(d_spk_train.keys()), len(d_spk_dev.keys()), len(d_spk_eval.keys()))

    """ # Note that there are some speaker without spoofing data in evaluation set
    print (non_spoof_spk from d_spk_eval)
    # ['LA_0057', 'LA_0061', 'LA_0068', 'LA_0064', 'LA_0060', 
    # 'LA_0058', 'LA_0051', 'LA_0049', 'LA_0059', 'LA_0066', 
    # 'LA_0056', 'LA_0055', 'LA_0063', 'LA_0052', 'LA_0065', 
    # 'LA_0067', 'LA_0054', 'LA_0050', 'LA_0053']
    """

    train_set = trainset_loader(d_cm_train, d_asv_train, d_spk_train, mode="train")

    # train_sampler = train_dataset_sampler(train_set)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=False
    )
    # sampler=train_sampler)

    # test trial
    with open(
        args.meta_path + "ASVspoof2019.LA.asv.dev.gi.trl.txt", "r"
    ) as f:
        asv_dev_trial = f.readlines()

    with open(
        args.meta_path + "ASVspoof2019.LA.asv.eval.gi.trl.txt", "r"
    ) as f:
        asv_eval_trial = f.readlines()

    dev_spkmodels = pk.load(open(args.asv_embd_path + "dev_spkmodels.pk", "rb"))
    print(len(dev_spkmodels.keys()), "dev speaker models are loaded!")

    eval_spkmodels = pk.load(open(args.asv_embd_path + "eval_spkmodels.pk", "rb"))
    print(len(eval_spkmodels.keys()), "eval speaker models are loaded!")

    dev_set = testset_loader(asv_dev_trial, dev_spkmodels, d_asv_dev, d_cm_dev)
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=False
    )

    eval_set = testset_loader(asv_eval_trial, eval_spkmodels, d_asv_eval, d_cm_eval)
    eval_loader = DataLoader(
        eval_set, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=False
    )

    return train_loader, dev_loader, eval_loader, asv_dev_trial, asv_eval_trial