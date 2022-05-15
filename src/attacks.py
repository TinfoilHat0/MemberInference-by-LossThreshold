import torch


def mia_by_threshold(model, tr_loader, te_loader, threshold, device='cuda:0', n_classes=10):
   
    with torch.inference_mode():
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        model.eval()
        tp, fp = torch.zeros(n_classes, device=device), torch.zeros(n_classes, device=device)
        tn, fn = torch.zeros(n_classes, device=device), torch.zeros(n_classes, device=device)
        
        # on training loader (members, i.e., positive class)
        for _, (inputs, labels) in enumerate(tr_loader):
            inputs, labels = inputs.to(device=device, non_blocking=True),\
                    labels.to(device=device, non_blocking=True)

            outputs = model(inputs)
            losses = criterion(outputs, labels)
            # with global threshold
            predictions = losses < threshold
            # class-wise confusion matrix values
            for i in range(n_classes):
                preds = predictions[labels == i]
                n_member_pred = preds.sum()
                tp[i] += n_member_pred
                fn[i] += len(preds) - n_member_pred
        
        # on test loader (non-members, i.e., negative class)
        for _, (inputs, labels) in enumerate(te_loader):
            inputs, labels = inputs.to(device=device, non_blocking=True),\
                    labels.to(device=device, non_blocking=True)
            outputs = model(inputs)
            losses = criterion(outputs, labels)
            # with global threshold
            predictions = losses < threshold
             # class-wise confusion matrix values
            for i in range(n_classes):
                preds = predictions[labels == i]
                n_member_pred = preds.sum()
                fp[i] += n_member_pred
                tn[i] += len(preds) - n_member_pred
        
        
        
        # class-wise bacc, tpr, fpr computations
        class_tpr, class_fpr = torch.zeros(n_classes, device=device),  torch.zeros(n_classes, device=device)
        class_bacc = torch.zeros(n_classes, device=device)
        for i in range(n_classes):
            class_i_tpr, class_i_tnr = tp[i]/(tp[i] + fn[i]), tn[i]/(tn[i] + fp[i])
            class_tpr[i], class_fpr[i] = class_i_tpr, 1-class_i_tnr
            class_bacc[i] = (class_i_tpr+class_i_tnr)/2
        
        # dataset-wise bacc, tpr, fpr computations
        ds_tp, ds_fp = tp.sum(), fp.sum()
        ds_tn, ds_fn = tn.sum(), fn.sum()
        ds_tpr, ds_tnr = ds_tp/(ds_tp + ds_fn), ds_tn/(ds_tn + ds_fp)
        ds_bacc, ds_fpr = (ds_tpr + ds_tnr)/2, 1 - ds_tnr
    
    return (ds_bacc, ds_tpr, ds_fpr), (class_bacc, class_tpr, class_fpr)
    