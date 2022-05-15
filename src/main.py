#!/usr/bin/env python
# coding: utf-8


import torch
import my_models as md
import my_utils as ut
import attacks as at
from options import args_parser
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from time import ctime
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings("ignore")

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


args = args_parser()


# datasets & dataloaders 
tr_ds, te_ds = ut.get_datasets(args.data)
v_size = int(len(tr_ds)*0.1)
tr_ds, v_ds = torch.utils.data.random_split(tr_ds, [len(tr_ds)-v_size, v_size])
tr_loader = DataLoader(tr_ds, batch_size=args.bs, shuffle=True, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
v_loader = DataLoader(v_ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True, persistent_workers=True)
te_loader = DataLoader(te_ds, batch_size=args.bs, shuffle=False, num_workers=args.num_workers, pin_memory=True)


# model & loss & etc.
model = md.get_model(args.data).to(args.device)
criterion = torch.nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters())
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5, verbose=True)
es = ut.EarlyStopping(patience=10, verbose=True)

# loggers
file_name = f"""time:{ctime()}_data:{args.data}"""
writer = SummaryWriter('../logs/' + file_name)


for ep in tqdm(range(args.num_epochs)):
    model.train()
    tr_loss, tr_acc = 0, 0
    for _, (inp, lbl) in enumerate(tr_loader):
        inp, lbl = inp.to(device=args.device, non_blocking=True), \
            lbl.to(device=args.device, non_blocking=True)

        #forward-pass
        out = model(inp)
        loss = criterion(out, lbl)
        
        # backprop
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        
        # keeping track of tr loss/acc
        tr_loss += loss.item()*out.shape[0]
        _, pred_lbl = torch.max(out, 1)
        tr_acc += torch.sum(torch.eq(pred_lbl.view(-1), lbl)).item()
      
    # inference on val data after epoch
    with torch.inference_mode():
        tr_loss, tr_acc = tr_loss/len(tr_loader.dataset), tr_acc/len(tr_loader.dataset)
        (v_loss, v_acc), _ = ut.get_loss_n_accuracy(model, v_loader, device=args.device)
        #loggers
        writer.add_scalar('Train/Loss', tr_loss, ep)
        writer.add_scalar('Train/Acc', tr_acc, ep)
        writer.add_scalar('Val/Loss', v_loss, ep)
        writer.add_scalar('Val/Acc', v_acc, ep)
        print(f'|Tr/Val Loss: {tr_loss:.3f} / {v_loss:.3f}|')
        print(f'|Tr/Val Acc: {tr_acc:.3f} / {v_acc:.3f}|')
        # lr scheduling & early-stopping
        lr_scheduler.step(v_loss)
        es(v_loss)
        if es.early_stop:
            print("Early stopping")
            break    


with torch.inference_mode():
    # metrics after training
    (tr_loss, tr_acc), (tr_per_class_acc, tr_per_class_loss) = ut.get_loss_n_accuracy(model, tr_loader, device=args.device)
    (te_loss, te_acc), (te_per_class_acc, te_per_class_loss) = ut.get_loss_n_accuracy(model, te_loader, device=args.device)
    # logging test set metrics
    bias = torch.max(te_per_class_acc) - torch.min(te_per_class_acc)
    writer.add_scalar('Test/Acc', te_acc, 0)
    writer.add_scalar('Test/Bias', bias, 0) 

    # run MIA
    (ds_bacc, ds_tpr, ds_fpr), (class_bacc, class_tpr, class_fpr) = at.mia_by_threshold(model, tr_loader, te_loader, threshold=tr_loss, device=args.device)

    # loggin performance of MIA
    writer.add_scalar('MIA/GenGap', tr_acc - te_acc, 0)
    writer.add_scalar('MIA/Bacc', ds_bacc, 0)
    writer.add_scalar('MIA/TPR', ds_tpr, 0)
    writer.add_scalar('MIA/FPR', ds_fpr, 0)
        
    # fin
    writer.flush()
    writer.close()        

