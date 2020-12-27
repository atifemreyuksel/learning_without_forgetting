from torch import nn

class KDLoss():
    def __init__(self, temp=2):
        self.temp = temp
    
    def __call__(self, preds, old_gts):
        pass

class TotalLoss():
    def __init__(self, strategy="lwf", lambda_old=1., temp=2):
        self.kd_loss = KDLoss(temp=temp)
        self.ce_loss = nn.CrossEntropyLoss() 
        self.strategy = strategy
        self.lambda_old = lambda_old

    def __call__(self, preds, gts, old_gts):
        if self.strategy == "lwf":
            preds_old, preds_new = preds[:, :-10], preds[:, -10:]
            old_task_loss = self.kd_loss(preds_old, old_gts)
            new_task_loss = self.ce_loss(preds_new, gts)
            total_loss = self.lambda_old * old_task_loss + new_task_loss
        else:
            total_loss = self.ce_loss(preds, gts)
        return total_loss
