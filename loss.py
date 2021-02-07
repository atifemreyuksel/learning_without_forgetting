import torch
from torch import nn
from torch.nn import functional as F

class KDLoss():
    def __init__(self, temp=2):
        self.temp = temp
        self.log_sotfmax = nn.LogSoftmax(dim=-1)

    def __call__(self, preds, gts, strategy):
        preds = F.softmax(preds, dim=-1)
        preds = torch.pow(preds, 1./self.temp)
        #l_preds = F.softmax(preds, dim=-1)
        l_preds = self.log_sotfmax(preds)

        if strategy == "lwf":
            gts = F.softmax(gts, dim=-1)
            gts = torch.pow(gts, 1./self.temp)
            l_gts = self.log_sotfmax(gts)
        
        elif strategy == "lwf_eq_prob":
            eq_prob = 1. / gts.size()[1]
            gts = torch.zeros_like(gts) + eq_prob.float()[:, None]
            gts = torch.pow(gts, 1./self.temp)
            l_gts = self.log_sotfmax(gts)
            
        l_preds = torch.log(l_preds)
        l_preds[l_preds != l_preds] = 0.
        loss = torch.mean(torch.sum(-l_gts * l_preds, axis=1))
        return loss

class TotalLoss():
    def __init__(self, strategy="lwf", num_new_classes=10, lambda_old=1., temp=2):
        self.kd_loss = KDLoss(temp=temp)
        self.ce_loss = nn.CrossEntropyLoss() 
        self.strategy = strategy
        self.num_new_classes = num_new_classes
        self.lambda_old = lambda_old

    def __call__(self, preds, gts, old_preds=None, old_gts=None, is_warmup=False):
        preds_old, preds_new = preds[:, :-self.num_new_classes], preds[:, -self.num_new_classes:]
        if self.strategy == "lwf" or self.strategy == "lwf_eq_prob":
            if not is_warmup:
                old_task_loss = self.kd_loss(preds_old, old_gts, self.strategy)
            else:
                old_task_loss = 0.
        else:
            old_task_loss = 0.
        new_task_loss = self.ce_loss(preds_new, gts)
        total_loss = self.lambda_old * old_task_loss + new_task_loss
        return total_loss
