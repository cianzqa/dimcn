import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from ...utils import MetricsTop, dict_to_str
logger = logging.getLogger('MMSA')

class DIMCN():
    def __init__(self, args):
        self.args = args
        self.args.tasks = 'MTAV'
        self.criterion = nn.MSELoss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.loss_cmd = CMD()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        self.model = model
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.args.learning_rate)
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        while True: 
            epochs += 1
            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            losses = []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    text = batch_data['text'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    vision = batch_data['vision'].to(self.args.device)
                    M_labels = batch_data['labels']['M'].to(self.args.device)
                    T_labels = batch_data['labels']['T'].to(self.args.device)
                    A_labels = batch_data['labels']['A'].to(self.args.device)
                    V_labels = batch_data['labels']['V'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        M_labels = M_labels.view(-1).long()
                    else:
                        M_labels = M_labels.view(-1, 1)
                        T_labels = T_labels.view(-1, 1)
                        A_labels = A_labels.view(-1, 1)
                        V_labels = V_labels.view(-1, 1)
                    outputs = model(text, audio, vision)
                    num_t = 0
                    num_a = 0
                    num_v = 0
                    for i in range(len(M_labels.tolist())):
                        if M_labels[i].item() * T_labels[i].item() >=0:
                            num_t += 1
                        if M_labels[i].item() * A_labels[i].item() >=0:
                            num_a += 1
                        if M_labels[i].item() * V_labels[i].item() >=0:
                            num_v += 1
                    cls_t_loss = self.criterion(outputs['T'], T_labels)
                    cls_a_loss = self.criterion(outputs['A'], A_labels)
                    cls_v_loss = self.criterion(outputs['V'], V_labels)
                    if num_t < self.args.num:
                        cls_t_loss = 0
                    if num_a < self.args.num:
                        cls_a_loss = 0
                    if num_v < self.args.num:
                        cls_v_loss = 0
                    cmd_loss = self.get_cmd_loss()
                    loss = self.args.m_weight*self.criterion(outputs['M'], M_labels)+self.args.t_weight*cls_t_loss+self.args.a_weight*cls_a_loss+self.args.v_weight*cls_v_loss + self.args.sim_weight * cmd_loss
                    loss.backward()
                    if self.args.grad_clip != -1.0:
                        torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], self.args.grad_clip)
                    train_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                    y_true['M'].append(M_labels.cpu())
                    y_true['T'].append(T_labels.cpu())
                    y_true['A'].append(A_labels.cpu())
                    y_true['V'].append(V_labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])
            for m in self.args.tasks:
                pred = torch.cat(y_pred[m])
                true = torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >>'%(m) + dict_to_str(train_results))
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred, y_true = [], []
        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    if self.args.train_mode == 'classification':
                        labels = labels.view(-1).long()
                    else:
                        labels = labels.view(-1, 1)
                    outputs = model(text, audio, vision)
                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        all_labels.extend(labels.cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()
                        sample_results.extend(preds.squeeze())
                    loss = self.criterion(outputs['M'], labels)
                    eval_loss += loss.item()
                    y_pred.append(outputs['M'].cpu())
                    y_true.append(labels.cpu())
        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")
        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels
        return eval_results

    def get_cmd_loss(self,):

        loss = self.loss_cmd(self.model.Model.utt_shared_t, self.model.Model.utt_shared_v, 5)
        loss += self.loss_cmd(self.model.Model.utt_shared_t, self.model.Model.utt_shared_a, 5)
        loss = loss/2.0
        return loss

class CMD(nn.Module):
    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)
