from torch import nn
from torch.nn import functional as F

class KDLoss():
    def __init__(self, temp=2):
        self.temp = temp
    
    def __call__(self, preds, gts):
        l_preds = F.log_softmax(preds, dim=-1)
        l_gts = F.log_softmax(gts, dim=-1)
        loss = F.nll_loss(l_preds, l_gts)
        return loss

class TotalLoss():
    def __init__(self, strategy="lwf", num_new_classes=10, lambda_old=1., temp=2):
        self.kd_loss = KDLoss(temp=temp)
        self.ce_loss = nn.CrossEntropyLoss() 
        self.strategy = strategy
        self.num_new_classes = num_new_classes
        self.lambda_old = lambda_old

    def __call__(self, preds, gts, old_preds=None, old_gts=None):
        preds_old, preds_new = preds[:, :-self.num_new_classes], preds[:, -self.num_new_classes:]
        if self.strategy == "lwf":
            old_task_loss = self.kd_loss(preds_old, old_gts)
        else:
            old_task_loss = 0.
        new_task_loss = self.ce_loss(preds_new, gts)
        total_loss = self.lambda_old * old_task_loss + new_task_loss
        return total_loss
