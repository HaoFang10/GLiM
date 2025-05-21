import os
import os.path
import json
import numpy as np
from evaluation import gen_train_facts
from collections import defaultdict
from tqdm import tqdm
import argparse
import torch
from model2 import DocREModel
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from utils import set_seed, collate_fn
from prepro import read_docred, read_chemdisgene, evaluate_bio
import random
from time import time
from losses import ATLoss
from train2 import report, official_evaluate

docred_rel2id = json.load(open('../dataset/dataset_docred/meta/rel2id.json', 'r'))
chemdisgene_rel2id = json.load(open('../dataset/dataset_chemdisgene/meta/rel2id.json', 'r'))
redocred_rel2id = json.load(open('../dataset/dataset_redocred/meta/rel2id.json', 'r'))

docred_id2rel = {value: key for key, value in docred_rel2id.items()}
chemdisgene_id2rel = {value+1: key for key, value in chemdisgene_rel2id.items()}
chemdisgene_id2rel[0] = 'NA'
redocred_id2rel = {value: key for key, value in redocred_rel2id.items()}

biored_rel2id = {'1:NR:2': 0, '1:Association:2': 1, '1:Bind:2': 2, '1:Comparison:2': 3, '1:Conversion:2': 4,
                 '1:Cotreatment:2': 5, '1:Drug_Interaction:2': 6, '1:Negative_Correlation:2': 7, '1:Positive_Correlation:2': 8}
biored_id2rel = {value: key for key, value in biored_rel2id.items()}

title2context = {}

pos = False

def to_official_plus(preds, scores, features, dataset ='docred'):
    global id2rel
    if dataset == 'docred':
        id2rel = docred_id2rel
    elif dataset == 'chemdisgene':
        id2rel = chemdisgene_id2rel
    elif dataset == 'biored':
        id2rel = biored_id2rel
        
    h_idx, t_idx, title = [], [], []

    for f in features:
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    for i in range(preds.shape[0]):
        pred = preds[i].tolist()
        pos = np.nonzero(pred)[0].tolist()

        score = scores[i].tolist()

        for idx in range(len(score)):
            if h_idx[i] != t_idx[i]:
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r'    : id2rel[idx],
                        'score': score[idx],
                        'postive': idx in pos if idx != 0 else False,
                        'th'   : score[0]
                    }
                )
                if idx in pos and idx != 0:
                    # print("title:{} h_idx:{} t_idx:{} r:{} score:{}".format(title[i], h_idx[i], t_idx[i], id2rel[idx], score[idx]))
                    assert score[idx] > score[0]

    return res

def gen_for_llm_data(args, path, tag, dataset, tmp = None):
    

    truth = json.load(open(os.path.join(path, tag + ".json")))

    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}
    
    for x in truth:
        title = x['title']
        titleset.add(title)

        context = ""
        if "sents" in x.keys():
            for sl in x['sents']:
                s = " ".join(sl)
                context += s + " "
        
        # print(context)
        title2context[title] = context

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label.get('evidence', []))
            tot_evidences += len(label.get('evidence', []))


    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]

    pred = {}
    
    pred[(tmp[0]['title'], tmp[0]['r'], tmp[0]['h_idx'], tmp[0]['t_idx'])] = \
        (title2vectexSet[tmp[0]['title']][tmp[0]['h_idx']][0]['type'], \
         title2vectexSet[tmp[0]['title']][tmp[0]['t_idx']][0]['type'],\
         [x['name'] for x in title2vectexSet[tmp[0]['title']][tmp[0]['h_idx']]], \
         [x['name'] for x in title2vectexSet[tmp[0]['title']][tmp[0]['t_idx']]],\
            tmp[0]['score'], tmp[0]['postive'], tmp[0]['th'])
    # tqdm
    for i in tqdm(range(1, len(tmp)), desc='Processing'):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])
            h_type = title2vectexSet[tmp[i]['title']][tmp[i]['h_idx']][0]['type']
            t_type = title2vectexSet[tmp[i]['title']][tmp[i]['t_idx']][0]['type']
            h_name = [x['name'] for x in title2vectexSet[tmp[i]['title']][tmp[i]['h_idx']]]
            t_name = [x['name'] for x in title2vectexSet[tmp[i]['title']][tmp[i]['t_idx']]]
            r_score = tmp[i]['score']
            r_postive = tmp[i]['postive']
            th = tmp[i]['th']
            pred[(x['title'], x['r'], x['h_idx'], x['t_idx'])] = (h_type, t_type, h_name, t_name, r_score, r_postive, th)


    pos_pred = { key: value for key, value in pred.items() if value[5] == True}
    neg_pred = { key: value for key, value in pred.items() if value[5] == False}

    llm_datas = []
    if tag == 'dev':
        for skey, svalue in tqdm(std.items(), desc="Processing items"):
            title, r, h_idx, t_idx = skey[0], skey[1], skey[2], skey[3]
            if (title, r, h_idx, t_idx) not in pos_pred:
                h_type = title2vectexSet[title][h_idx][0]['type']
                t_type = title2vectexSet[title][t_idx][0]['type']

                score = pred[(title, r, h_idx, t_idx)][4]

                th = pred[(title, id2rel[0], h_idx, t_idx)][4] 

                llm_data = {
                    'title': title,
                    'h_idx': h_idx,
                    't_idx': t_idx,
                    'r': r,
                    'h_type': h_type,
                    't_type': t_type,
                    "h_name": [x['name'] for x in title2vectexSet[title][h_idx]],
                    "t_name": [x['name'] for x in title2vectexSet[title][t_idx]],
                    "score": score,
                    "th": th
                }
                llm_datas.append(llm_data)

    elif tag == 'test':  
        for key, val in tqdm(neg_pred.items(), desc="Processing items"):
            title, r, h_idx, t_idx = key[0], key[1], key[2], key[3]
            h_type = title2vectexSet[title][h_idx][0]['type']
            t_type = title2vectexSet[title][t_idx][0]['type']
            h_name = [x['name'] for x in title2vectexSet[title][h_idx]]
            t_name = [x['name'] for x in title2vectexSet[title][t_idx]]
            
            r_score = val[4]
            llm_data = {
                'title': title,
                'h_idx': h_idx,
                't_idx': t_idx,
                'h_type': h_type,
                't_type': t_type,
                'h_name': h_name,
                't_name': t_name,
                'r': r,
                'score': r_score,
                "th": pred[(title, id2rel[0], h_idx, t_idx)][4] 
            }
            llm_datas.append(llm_data)

    return llm_datas
  
def evaluate_gen(args, model, features, tag="dev", use_ILP = False):
    # if args.dataset == 'chemdisgene':
    #     return evaluate_bio(args, model, features, tag)
    
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn,
                            drop_last=False, num_workers=0)
    preds = []
    scores = []

    begin_time = time()

    for batch in dataloader:
        model.eval()


        inputs = {'input_ids'     : batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'labels': batch[2],
                  'entity_pos'    : batch[3],
                  'hts'           : batch[4],
                  'output_for_LogiRE': True
                  }

        with torch.no_grad():
            logits, logits_soft = model(**inputs)
            pred = ATLoss().get_label(logits_soft, num_labels=args.num_labels)
            score = logits_soft

            pred = pred.cpu().numpy()
            score = score.cpu().numpy()
           
            pred[np.isnan(pred)] = 0
            score[np.isnan(score)] = 0

            preds.append(pred)
            scores.append(score)

            
    end_time = time()
    print('the time used by inference:', end_time - begin_time)

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    scores = np.concatenate(scores, axis=0).astype(np.float32)
    res = to_official_plus(preds, scores, features, dataset=args.dataset)
    data = gen_for_llm_data(args, args.data_dir, tag, args.dataset, res)
    
    return data
          

def make_r_data(args, test_ans, ratio=0.0):
    test_data = []
    test_filter_data = []

    ht_info = {}
    ht_r = {}
    ths = {}
    for i, data in tqdm(enumerate(test_ans), total=len(test_ans), desc="Processing test_ans"):
        title = data['title']
        h_idx = data['h_idx']
        t_idx = data['t_idx']
        h_type = data['h_type']
        t_type = data['t_type']
        h_name = data['h_name']
        t_name = data['t_name']
        r = data['r']
        score = data['score']
        th = data['th']


        if (title, h_idx, t_idx) not in ht_info:
            ht_info[(title, h_idx, t_idx)] = (h_type, h_name, t_type, t_name)

        if (title, h_idx, t_idx) not in ht_r:
            ht_r[(title, h_idx, t_idx)] = {}
        
        if (title, h_idx, t_idx) not in ths:
            ths[(title, h_idx, t_idx)] = th
        
        if r == id2rel[0]:
            continue
        
        ht_r[(title, h_idx, t_idx)][r] = score

    
    for key, value in ht_info.items():
        test_data.append({
            'title': key[0],
            'h_idx': key[1],
            't_idx': key[2],
            'h_type': value[0],
            'h_name': value[1],
            't_type': value[2],
            't_name': value[3],
            'r_s': ht_r[key],
            'na_th': ths[key]
        })

    for key, value in ht_info.items():
        rs = {k:v for k,v in ht_r[key].items() if v > ths[key]*ratio}
        if len(rs) == 0:
            continue

        test_filter_data.append({
            'title': key[0],
            'h_idx': key[1],
            't_idx': key[2],
            'h_type': value[0],
            'h_name': value[1],
            't_type': value[2],
            't_name': value[3],
            'r_s': rs,
            'na_th': ths[key],
            "content": title2context[key[0]]
        })
    
    return test_data, test_filter_data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="REGT", type=str)
    parser.add_argument("--dataset", default="docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="../PLM/bert-base-cased", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="../trained_model/model_REGT.pth", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=4, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--lambda_al", default=1.0, type=float)
    parser.add_argument("--Type_Enhance", action='store_true')
    parser.add_argument("--tau", default=0.2, type=float)
    parser.add_argument("--display_name", default="REGT", type=str)
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument("--ratio", default=1.0, type=float)
    parser.add_argument("--k", default=1.0, type=float)

    args = parser.parse_args()
    args.data_dir = "../dataset/dataset_{}/".format(args.dataset)
    args.load_ner_path = "../dataset/dataset_{}/meta/ner2id.json".format(args.dataset)
    args.load_rel2id_path = "../dataset/dataset_{}/meta/rel2id.json".format(args.dataset)


    global dataset
    dataset = args.dataset
    print(str(args))
    device = torch.device("cuda:"+args.gpu if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
 
    global pos
    pos = args.pos
    print("pos:{}".format(pos))

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )
    rel2id = json.load(open(args.load_rel2id_path, 'r'))
    rel2id = sorted(rel2id.items(), key=lambda kv: (kv[1], kv[0]))
    rel2id = {kv[0]:kv[1] for kv in rel2id}
    read_map = {
        'docred': read_docred,
        'chemdisgene': read_chemdisgene,
        'biored': read_biored
    }
    read = read_map[args.dataset]
 
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
   
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length, Type_Enhance=False)
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length, Type_Enhance=False)

    if args.dataset == 'chemdisgene' or args.dataset == 'redocred' or args.dataset == 'docred' or args.dataset == 'biored':
        dev_features = dev_features[0]
        test_features = test_features[0]

    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    if args.Type_Enhance:
        model.resize_token_embeddings(len(tokenizer))

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, model, dataset=args.dataset, num_labels=args.num_labels,
                       temperature=args.tau, device=args.device)
    model.to(args.device)

    if args.load_path == "":  
        # wrong
        print("Please specify the path of the model to load")
        return
        
    elif args.load_path != "":  # Testing
        model.load_state_dict(torch.load(args.load_path), strict=False)

        # dev_ans = evaluate_gen(args, model, dev_features, tag="dev")
        # dev_ans,_ = make_r_data(args, dev_ans)

        # th = [ x['na_th'] for x in dev_ans]
        # t_min = min(th)
        # t_max = max(th)
        # th = sum(th) / len(th)

        # score = [ v for x in dev_ans for k,v in x['r_s'].items() ]
        # s_min = min(score)
        # s_max = max(score)
        # score = sum(score) / len(score)

        # print("th:{}, t_min:{}, t_max:{}".format(th, t_min, t_max))
        # print("score:{}, s_min:{}, s_max:{}".format(score, s_min, s_max))
       
        ratio = args.ratio * args.k

        test_ans = evaluate_gen(args, model, test_features, tag="test")
        test_ans, test_filter = make_r_data(args, test_ans, ratio=ratio)
        final_data = linkpred_dataset(test_filter, {}, dataset)
        linkd_json = []

        assert len(final_data) == len(test_filter)
        for idx, item in enumerate(final_data):
            linkd_json.append({
                "instruction": item["instruction"],
                "input": "",
                "output": ' '.join(item["answer"]),
                "info": test_filter[idx] 
            })
        print(len(linkd_json))
        with open("step2_inputs.json" , "w") as f:
            json.dump(linkd_json, f, indent=4)




if __name__ == "__main__":
    main() 