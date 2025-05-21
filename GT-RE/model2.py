import torch
import torch.nn as nn
from long_seq import process_long_input
from losses import ATLoss, SPULoss
import numpy as np
from graph import build_Graph
from tokenGT import tokenGT

class DocREModel(nn.Module):
    def __init__(self, config, model, dataset='', emb_size=768, block_size=64, priors_l=None, num_labels=-1,
                 temperature=1,  device='cpu'):
        super().__init__()
        self.config = config
        self.device = device
        self.model = model
        self.hidden_size = config.hidden_size
        self.temperature = temperature
 
        self.n = config.num_labels
        self.loss_fnt = ATLoss()


        # self.head_extractor = nn.Linear(3 * config.hidden_size, emb_size)
        # self.tail_extractor = nn.Linear(3 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels)

        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

        self.null_token = nn.Embedding(1, config.hidden_size)
        self.orf_node_id_dim = 128
        self.random_ortho_encoder = nn.Linear(self.orf_node_id_dim,config.hidden_size)
        self.type_embedding = nn.Embedding(2, config.hidden_size)
        self.tokengt = tokenGT(n_layers=4, dim_in=config.hidden_size, dim_out=config.hidden_size, dim_hidden=config.hidden_size, dim_qk=64, dim_v=64, dim_ff=3072, n_heads=4,
                        drop_input=0.1, dropout=0.1, drop_mu=0.1, last_layer_n_heads=4)
        
        self.spu_loss = None
        # s-pu
        print("priors_l", priors_l)
        if priors_l is not None:
            self.spu_loss = SPULoss(priors_l, config.num_labels, dataset)

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention
        
    def get_gt(self, final_G_null_list, entity_pos, hts):
        gss = []
        assert len(final_G_null_list) == len(entity_pos)
        for i in range(len(final_G_null_list)):
            final_G_null = final_G_null_list[i]
            output = final_G_null.values  # [|E|, dim_hidden]
            indices = final_G_null.indices  # [|E|, 2]
            # 获取[sub_start_idx, obj_start_idx]在indices中的位置
            # 要查找的 [sub_start_idx, obj_start_idx] 对
            ht_i = torch.LongTensor(hts[i]).to(final_G_null.device)
            
            offset = (ht_i[:, 0] > ht_i[:, 1]).int()  
            index =len(entity_pos[i]) + ht_i[:, 0] * len(entity_pos[i]) + ht_i[:, 1] - ht_i[:, 0] + offset - 1

            flag1 = indices[index, 0] == ht_i[:, 0]
            flag2 = indices[index, 1] == ht_i[:, 1]
            # flag是否有false
            assert flag1.all()
            assert flag2.all()

            # 查找 target_pair 在 indices 中的位置
            gs = torch.index_select(output, 0, index)
            gss.append(gs)
        
        gss = torch.cat(gss, dim=0)
        return gss
            
    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = torch.einsum("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def activation(self, x):
        return torch.minimum(torch.maximum(x, torch.zeros_like(x)), torch.ones_like(x))

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                output_for_LogiRE=False,
                ):
        sequence_output, attention = self.encode(input_ids, attention_mask)

        G_list = build_Graph(sequence_output, attention, entity_pos, hts)
        
        final_G_null_list = []
        attn_score_list = []

        for i, G_null in enumerate(G_list):
            # get attention map of model
            attn_score, final_G_null = self.tokengt(G_null)  
            final_G_null_list.append(final_G_null)
            attn_score_list.append(attn_score)
        
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        gs = self.get_gt(final_G_null_list, entity_pos, hts)


        # hs = torch.tanh(self.head_extractor(torch.cat([hs, rs, gs], dim=1)))
        # ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs, gs], dim=1)))


        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear(bl)


        if self.spu_loss is not None:
            output = (self.spu_loss.get_label(logits, num_labels=self.num_labels),)
        else:
            output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels),)

        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            
            if self.spu_loss is not None:
                loss_cls = self.spu_loss(logits.float(), labels.float())
                loss_dict = {'spu': True, 'loss_cls': loss_cls.item()}
            else:
                loss_cls = self.loss_fnt(logits.float(), labels.float())
                loss_dict = {'loss_cls': loss_cls.item()}

            loss = loss_cls

            print(loss_dict)
            output = (loss.to(sequence_output), loss_dict) + output
        return output
