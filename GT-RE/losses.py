import torch
import torch.nn as nn
import torch.nn.functional as F


class ATLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, labels):
        # TH label
        th_label = torch.zeros_like(labels, dtype=torch.float).to(labels)
        th_label[:, 0] = 1.0
        labels[:, 0] = 0.0

        p_mask = labels + th_label
        n_mask = 1 - labels

        # Rank positive classes to TH
        logit1 = logits - (1 - p_mask) * 1e30
        loss1 = -(F.log_softmax(logit1, dim=-1) * labels).sum(1)

        # Rank TH to negative classes
        logit2 = logits - (1 - n_mask) * 1e30
        loss2 = -(F.log_softmax(logit2, dim=-1) * th_label).sum(1)

        # Sum two parts
        loss = loss1 + loss2
        loss = loss.mean()
        return loss

    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output


class SPULoss(nn.Module):
    def __init__(self, priors_l, num_class, dataset):
        super().__init__()
        if dataset == 'chemdisgene':
            self.margin = 0.25
            self.e = 3
        elif dataset == 'docred':
            self.margin = 1.0
            self.e = 3.0
        else:
            self.margin = 1.0
            # self.margin = 0.75
            # self.margin = 0.25
            self.e = 1.0
        self.priors_l = priors_l
        self.priors_o = priors_l * self.e
        self.priors_u = (self.priors_o - self.priors_l) / (1. - self.priors_l)
        self.weight = ((1 - self.priors_o)/self.priors_o) ** 0.5
        self.rels = num_class - 1

    def forward(self, logits, labels):
        risk_sum = []
        # TH label
        risk_sum_ = torch.FloatTensor([0]).to(self.priors_l)


        for i in range(self.rels):
            neg = (logits[(labels[:, i + 1] != 1), i + 1] - logits[(labels[:, i + 1] != 1), 0])
            pos = (logits[(labels[:, i + 1] == 1), i + 1] - logits[(labels[:, i + 1] == 1), 0])

            priors_u = (self.priors_o[i] - self.priors_l[i]) / (1. - self.priors_l[i])
            risk1 = ((1. - self.priors_o[i]) / (1. - priors_u)) * self.square_loss(neg, -1., self.margin) - ((priors_u - priors_u * self.priors_o[i]) / (1. - priors_u)) * self.square_loss(pos, -1., self.margin)
            risk2 = self.priors_o[i] * self.square_loss(pos, 1., self.margin) * self.weight[i]
            risk = risk1 + risk2

            if risk1 < 0.0:
                # risk = - 1.0 * risk1
                risk = risk2

            risk_sum_ += risk
        risk_sum.append(risk_sum_)
        loss = sum(risk_sum)
        return loss
    
    def square_loss(self, yPred, yTrue, margin=1.):
        if len(yPred) == 0:
            return torch.FloatTensor([0]).to(self.priors_l)
        loss = (yPred * yTrue - margin) ** 2
        return torch.mean(loss.sum() / yPred.shape[0])
    
    def get_label(self, logits, num_labels=-1):
        th_logit = logits[:, 0].unsqueeze(1)
        output = torch.zeros_like(logits).to(logits)
        mask = (logits > th_logit)
        if num_labels > 0:
            top_v, _ = torch.topk(logits, num_labels, dim=1)
            top_v = top_v[:, -1]
            mask = (logits >= top_v.unsqueeze(1)) & mask
        output[mask] = 1.0
        output[:, 0] = (output.sum(1) == 0.).to(logits)
        return output