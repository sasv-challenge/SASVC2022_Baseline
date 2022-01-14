import argparse
import os
import json
import numpy as np

import torch
import torch.nn as nn

from data_utils import get_loader
from trainer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter

from models.concat_embeds import int_model


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-name", type=str, required=True)
    parser.add_argument("-save_path", type=str, default="./exp/")
    parser.add_argument("-meta_path", type=str, default="./data/_meta/")
    parser.add_argument("-cm_embd_path", type=str, default="./data/CM/")
    parser.add_argument("-asv_embd_path", type=str, default="./data/ASV/")
    parser.add_argument("-eval_output", type=str, default="eval_scores_using_best_dev_model.txt")

    # hyper-params
    parser.add_argument("-batch_size", type=int, default=24)
    parser.add_argument("-lr", type=float, default=0.0001)
    parser.add_argument("-wd", type=float, default=0.001)
    parser.add_argument("-optimizer", type=str, default="adam")
    parser.add_argument("-nb_worker", type=int, default=16)
    parser.add_argument("-seed", type=int, default=1234)
    parser.add_argument("-lr_decay", type=str, default="keras")
    parser.add_argument("-loss", type=str, default="cce")
    parser.add_argument("-loss_embd_wgt", type=float, default=0.25)
    parser.add_argument(
        "-model", type=json.loads, default='{"dnn_l_nodes":[256, 128, 64],' '"code_dim":544}'
    )  # ECAPA: 192, AASIST: 160 = > 544 : 192*2 + 160
    parser.add_argument("-opt_mom", type=float, default=0.9)
    parser.add_argument("-epoch", type=int, default=50)

    # flag
    parser.add_argument("-save_spkdic", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("-amsgrad", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("-nesterov", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("-debug", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("-comet_disable", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("-save_best_only", type=str_to_bool, nargs="?", const=True, default=True)
    parser.add_argument("-do_lr_decay", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("-mg", type=str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument(
        "-cudnn_deterministic_toggle", type=str_to_bool, nargs="?", const=True, default=True
    )
    parser.add_argument(
        "-cudnn_benchmark_toggle", type=str_to_bool, nargs="?", const=True, default=False
    )

    return parser.parse_args()


def main():
    args = get_args()

    # device setting
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # device setting
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print("Device: {}".format(device))

    # reproducibility of experiment
    set_seed(args)

    # get dataloaders and trial(lines)
    train_loader, dev_loader, eval_loader, asv_dev_trial, asv_eval_trial = get_loader(args)

    # set save directory
    save_path = args.save_path + args.name + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(save_path + "results/"):
        os.makedirs(save_path + "results/")
    if not os.path.exists(save_path + "models/"):
        os.makedirs(save_path + "models/")
    model_save_path = save_path + "models/"
    eval_score_path = save_path + "results/"
    writer = SummaryWriter(save_path + "tensorboard/")

    # log experiment parameters to local
    f_params = open(save_path + "f_params.txt", "w")
    for k, v in sorted(vars(args).items()):
        print(k, v)
        f_params.write("{}:\t{}\n".format(k, v))
    for k, v in sorted(args.model.items()):
        print(k, v)
        f_params.write("{}:\t{}\n".format(k, v))
    f_params.close()

    model = int_model(args.model).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model.apply(set_init_weights)
    print("int_model_nb_params: {}".format(nb_params))

    # set optimizer
    params = [
        {"params": [param for name, param in model.named_parameters() if "bn" not in name]},
        {
            "params": [param for name, param in model.named_parameters() if "bn" in name],
            "weight_decay": 0,
        },
    ]

    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.opt_mom, weight_decay=args.wd, nesterov=args.nesterov
        )
    elif args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wd, amsgrad=args.amsgrad)
    else:
        raise NotImplementedError("Add other optimizers if needed")

    # set learning rate decay
    if bool(args.do_lr_decay):
        if args.lr_decay == "keras":
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda step: keras_decay(step)
            )
        else:
            raise NotImplementedError("Not implemented yet, got:{}".format(args.lr_decay))
    else:
        lr_scheduler = None

    best_eer = 99.0
    f_res = open(save_path + "evaluation_summary.txt", "w")
    for epoch in range(args.epoch):
        loss = train(
            model=model,
            trn_loader=train_loader,
            optim=optimizer,
            args=args,
            device=device,
            scheduler=lr_scheduler,
        )

        dev_eer, dev_sv_eer, dev_cm_eer = produce_evaluation_file(
            dev_loader, model, device, eval_score_path, epoch, asv_dev_trial, mode="dev"
        )

        print(
            "EPOCH: %d:\t Loss: %.5f\t SASV_EER:%f\t SV_EER:%f\t CM_EER:%f"
            % (epoch, loss, dev_eer, dev_sv_eer, dev_cm_eer)
        )
        writer.add_scalar("loss", loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)

        save_model_dict = model.state_dict()
        if float(dev_eer) < best_eer:
            print("New best dev EER: %f" % float(dev_eer))
            best_eer = float(dev_eer)
            torch.save(save_model_dict, model_save_path + "best_dev_e%d_%.4f.pt" % (epoch, dev_eer))
            torch.save(
                optimizer.state_dict(),
                model_save_path + "best_dev_opt_e%d_%.4f.pt" % (epoch, dev_eer),
            )

            eval_eer, eval_sv_eer, eval_cm_eer = produce_evaluation_file(
                eval_loader, model, device, eval_score_path, epoch, asv_eval_trial, mode="eval"
            )

            eval_txt = (
                "!!! EVAL: EPOCH: :{}\t SASV_EER:{:.4f}\t SV_EER:{:.4f}\t CM_EER:{:.4f} !!!".format(
                    epoch, eval_eer, eval_sv_eer, eval_cm_eer
                )
            )
            print(eval_txt)
            f_res.write(eval_txt[4:-5] + "\n")
            writer.add_scalar("eval_eer", eval_eer, epoch)
    f_res.close()


if __name__ == "__main__":
    main()