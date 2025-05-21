from tqdm import tqdm
import ujson as json
from collections import defaultdict
import numpy as np
import unidecode
import torch
from torch.utils.data import DataLoader
from utils import collate_fn
from evaluation import to_official


docred_rel2id = json.load(open('../dataset/dataset_docred/meta/rel2id.json', 'r'))
docred_fact_in_train = json.load(open('../dataset/dataset_docred/ref/train_annotated.fact', 'r'))
docred_fact_in_train = set(map(lambda x: tuple(x), docred_fact_in_train))
ctd_rel2id = json.load(open('../dataset/dataset_chemdisgene/meta/rel2id.json', 'r'))
ENTITY_PAIR_TYPE_SET = set(
    [("Chemical", "Disease"), ("Chemical", "Gene"), ("Gene", "Disease")])


def map_index(chars, tokens):
    # position index mapping from character level offset to token level offset
    ind_map = {}
    i, k = 0, 0  # (character i to token k)
    len_char = len(chars)
    num_token = len(tokens)
    while k < num_token:
        if i < len_char and chars[i].strip() == "":
            ind_map[i] = k
            i += 1
            continue
        token = tokens[k]
        if token[:2] == "##":
            token = token[2:]
        if token[:1] == "Ġ":
            token = token[1:]

        # assume that unk is always one character in the input text.
        if token != chars[i:(i+len(token))]:
            ind_map[i] = k
            i += 1
            k += 1
        else:
            for _ in range(len(token)):
                ind_map[i] = k
                i += 1
            k += 1

    return ind_map

def read_chemdisgene(file_in, tokenizer, max_seq_length=1024, Type_Enhance = False, lower=True):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    pos, neg, pos_labels, neg_labels = {}, {}, {}, {}
    for pair in list(ENTITY_PAIR_TYPE_SET):
        pos[pair] = 0
        neg[pair] = 0
        pos_labels[pair] = 0
        neg_labels[pair] = 0
    ent_nums = 0
    rel_nums = 0
    max_len = 0
    features = []
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    padid = tokenizer.pad_token_id
    cls_token_length = len(cls_token)
    print(cls_token, sep_token)
    if file_in == "":
        return None
    with open(file_in, "r") as user:
        data = json.load(user)

    re_fre = np.zeros(len(ctd_rel2id))
    for sample in tqdm(data, desc="Example"):
        if "title" in sample and "abstract" in sample:
            text = sample["title"] + sample["abstract"]
            if lower == True:
                text = text.lower()
        else:
            text = sample["text"]
            if lower == True:
                text = text.lower()

        text = unidecode.unidecode(text)
        tokens = tokenizer.tokenize(text)
        tokens = [cls_token] + tokens + [sep_token]
        text = cls_token + " " + text + " " + sep_token

        ind_map = map_index(text, tokens)

        entities = sample['entity']
        entity_start, entity_end = [], []

        train_triple = {}
        if "relation" in sample:
            for label in sample['relation']:
                if label['type'] not in ctd_rel2id:
                    continue
                if 'evidence' not in label:
                    evidence = []
                else:
                    evidence = label['evidence']
                r = int(ctd_rel2id[label['type']])

                if (label['subj'], label['obj']) not in train_triple:
                    train_triple[(label['subj'], label['obj'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['subj'], label['obj'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        entity_dict = {}
        entity2id = {}
        entity_type = {}
        eids = 0
        offset = 0

        for e in entities:

            entity_type[e["id"]] = e["type"]
            if e["start"] + cls_token_length in ind_map:
                startid = ind_map[e["start"] + cls_token_length] + offset
                tokens = tokens[:startid] + ['*'] + tokens[startid:]
                offset += 1
            else:
                continue
                startid = 0


            if e["end"] + cls_token_length in ind_map:
                endid = ind_map[e["end"] + cls_token_length] + offset
                if ind_map[e["start"] + cls_token_length] >= ind_map[e["end"] + cls_token_length]:
                    endid += 1
                tokens = tokens[:endid] + ['*'] + tokens[endid:]
                endid += 1
                offset += 1
            else:
                continue
                endid = 0

            if startid >= endid:
                endid = startid + 1

            if e["id"] not in entity_dict:
                entity_dict[e["id"]] = [(startid, endid,)]
                entity2id[e["id"]] = eids
                eids += 1
                if e["id"] != "-":
                    ent_nums += 1
            else:
                entity_dict[e["id"]].append((startid, endid,))

        relations, hts, in_trains = [], [] , []
        for h, t in train_triple.keys():
            if h not in entity2id or t not in entity2id or ((entity_type[h], entity_type[t]) not in ENTITY_PAIR_TYPE_SET):
                continue
            in_train = [0] * (len(ctd_rel2id) + 1)
            relation = [0] * (len(ctd_rel2id) + 1)
            for mention in train_triple[h, t]:
                if relation[mention["relation"] + 1] == 0:
                    re_fre[mention["relation"]] += 1
                relation[mention["relation"] + 1] = 1
                in_train[mention["relation"] + 1] = 1
                evidence = mention["evidence"]
                
            relations.append(relation)
            in_trains.append(in_train)
            hts.append([entity2id[h], entity2id[t]])

            rel_num = sum(relation)
            rel_nums += rel_num
            pos_labels[(entity_type[h], entity_type[t])] += rel_num
            pos[(entity_type[h], entity_type[t])] += 1
            pos_samples += 1

        for h in entity_dict.keys():
            for t in entity_dict.keys():
                if (h != t) and ([entity2id[h], entity2id[t]] not in hts) and ((entity_type[h], entity_type[t]) in ENTITY_PAIR_TYPE_SET) and (h != "-") and (t != "-"):
                # if (h != t) and ([entity2id[h], entity2id[t]] not in hts):
                    if (entity_type[h], entity_type[t]) not in neg:
                        neg[(entity_type[h], entity_type[t])] = 1
                    else:
                        neg[(entity_type[h], entity_type[t])] += 1
                    
                    in_train = [0] * (len(ctd_rel2id)+1)
                    relation = [1] + [0] * (len(ctd_rel2id))
                    relations.append(relation)
                    in_trains.append(in_train)
                    hts.append([entity2id[h], entity2id[t]])
                    neg_samples += 1

        if len(tokens) > max_len:
            max_len = len(tokens)

        tokens = tokens[1:-1][:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        
        ne = len(list(entity_dict.values()))
        # if ( ne < 2 or ne*(ne-1) != len(hts)):
        if (len(hts)==0 or len(relations)==0 or ne < 2 ):
            continue
        i_line += 1

        feature = {'input_ids': input_ids,
                'entity_pos': list(entity_dict.values()),
                'labels': relations,
                'hts': hts,
                'in_trains': in_trains,
                'title': sample['docid'],
                'entity2id': entity2id,
                }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    re_fre = 1. * re_fre / (pos_samples + neg_samples)
    print(re_fre)
    print(max_len)
    print(pos)
    print(pos_labels)
    print(neg)
    print("# ents per doc", 1. * ent_nums / i_line)
    print("# rels per doc", 1. * rel_nums / i_line)
    return features, re_fre


def evaluate_bio(args, model, features, tag="test"):
    if args.dataset == 'biored' :
        return evaluate_bio2(args, model, features, tag)

    dataloader = DataLoader(features, batch_size=args.test_batch_size, 
                            shuffle=False, collate_fn=collate_fn, drop_last=False, num_workers=0)
    preds = []
    golds = []
    
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'labels': batch[2],
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            _, _, pred = model(**inputs)


            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0

            preds.append(pred)
            labels = [np.array(label, np.float32) for label in batch[2]]
            golds.append(np.concatenate(labels, axis=0))
            

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ori_preds = preds
        

    preds = preds[:,1:]
    golds = np.concatenate(golds, axis=0).astype(np.float32)[:,1:]

    TPs = preds * golds  # (N, R)
    TP = TPs.sum()
    P = preds.sum()
    T = golds.sum()
   

    micro_p = TP / P if P != 0 else 0
    micro_r = TP / T if T != 0 else 0
    micro_f = 2 * micro_p * micro_r / \
        (micro_p + micro_r) if micro_p + micro_r > 0 else 0
    mi_output = {
            tag + "_F1": micro_f * 100,
            tag + "_re_p": micro_p * 100,
            tag + "_re_r": micro_r * 100,
        }
    print("TP:{}, P:{}, T:{}".format(TP, P, T))
    ans = to_official(ori_preds, features,dataset=args.dataset)
    return micro_f, mi_output, ans


def evaluate_bio2(args, model, features, tag="test"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, 
                            shuffle=False, collate_fn=collate_fn, drop_last=False, num_workers=0)
    preds = []
    golds = []
    ori_preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'labels': batch[2],
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            _, _, pred = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            ori_preds.append(pred)
            golds.append(np.concatenate([np.array(label, np.float32) for label in batch[2]], axis=0))
            
    ori_preds = np.concatenate(ori_preds, axis=0).astype(np.float32)
    preds = np.concatenate(preds, axis=0).astype(np.float32)
    golds = np.concatenate(golds, axis=0).astype(np.float32)
    print(golds.shape)

    tp = ((preds[:, 1:9] == 1) & (golds[:, 1:9] == 1)).astype(np.float32).sum()
    tn = ((golds[:, 1:9] == 1) & (preds[:, 1:9] != 1)).astype(np.float32).sum()
    fp = ((preds[:, 1:9] == 1) & (golds[:, 1:9] != 1)).astype(np.float32).sum()
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + tn + 1e-5)
    micro_f = 2 * precision * recall / (precision + recall + 1e-5)
    mi_output = {
        "{}_p".format(tag): precision * 100,
        "{}_r".format(tag): recall * 100,
        "{}_f1".format(tag): micro_f * 100,
    }
    TP = tp
    P = tp + fp
    T = tp + tn
   
    print("TP:{}, P:{}, T:{}".format(TP, P, T))
    ans = to_official(ori_preds, features, dataset=args.dataset)
    return micro_f, mi_output, ans

def export_test_bio(args, features):
    if args.dataset != 'chemdisgene':
        return export_test_bio2(args, features)

    chemdisgene_rel2id = json.load(open('../dataset/dataset_chemdisgene/meta/rel2id.json', 'r'))
    chemdisgene_id2rel = {value+1: key for key, value in chemdisgene_rel2id.items()}
    ori_path = "../dataset/dataset_chemdisgene/valid.json"
    id2rel = chemdisgene_id2rel
    results = []

    ori_data = json.load(open(ori_path))
    ori_data = {x["docid"]: x for x in ori_data}
    for f in features:
        samples = {}
        title = f["title"]
        
        hts = f["hts"]
        olabels = f["labels"]
        sents = [[ori_data[title]["title"]],[ori_data[title]["abstract"]]]
        labels = []
        for i in range(len(hts)):
            h, t = hts[i]
            label = olabels[i]
            for idx, rel in enumerate(label):
                if rel == 1 and idx != 0:
                    labels.append({
                        "h": h,
                        "t": t,
                        "r": id2rel[idx],
                        "evidence": []
                    })
        
        samples["title"] = title
        samples["sents"] = sents
        samples["labels"] = labels

        vertexSet = []
        entity2id = f["entity2id"]
        for i in range(len(entity2id)):
            vertexSet.append([])
        
        entity = ori_data[title]["entity"]
        for e in entity:
            if e["id"] in entity2id:
                idx = entity2id[e["id"]]
                vertexSet[idx].append({
                    "type": e["type"],
                    "name": e["mention"],
                    "id": e["id"],
                })
        
        samples["vertexSet"] = vertexSet
        results.append(samples)
    
    # 保存
    with open("dev_bio.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    
def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res

def export_test_bio2(args, features):
    biored_rel2id = json.load(open('../dataset/dataset_biored/meta/rel2id.json', 'r'))
    biored_id2rel = {value: key for key, value in biored_rel2id.items()}
    ori_path = "../dataset/dataset_biored/biored_test.data"
    id2rel = biored_id2rel
    results = []

    ori_data = {}
    with open(ori_path, 'r') as infile:
        lines = infile.readlines()
        # lines = lines[:10]
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]
            


            if pmid not in ori_data.keys():
                text = line[1]
                sents = [t for t in text.split('|')]
                content = ' '.join(sents)

                prs = chunks(line[2:], 17)
                entity = set()
                for p in prs:
                    htpy, ttpy = p[7], p[13]   
                    h_id, t_id = p[5], p[11]
                    h_name, t_name = p[6], p[12]

                    entity.add((h_id, h_name, htpy))
                    entity.add((t_id, t_name, ttpy))

                ori_data[pmid] = {
                    "content": content,
                    "entity":  entity
                }
    for f in features:
        samples = {}
        title = f["title"]
        
        hts = f["hts"]
        olabels = f["labels"]
        sents = [[ori_data[title]["content"]]]
        labels = []
        for i in range(len(hts)):
            h, t = hts[i]
            label = olabels[i]
            for idx, rel in enumerate(label):
                if rel == 1 and idx != 0:
                    labels.append({
                        "h": h,
                        "t": t,
                        "r": id2rel[idx],
                        "evidence": []
                    })
        
        samples["title"] = title
        samples["sents"] = sents
        samples["labels"] = labels

        vertexSet = []
        entity2id = f["entity2id"]
        for i in range(len(entity2id)):
            vertexSet.append([])
        
        oentity = ori_data[title]["entity"]
        entity = []
        for eid, name, tpy in list(oentity):
            names = name.split('|')
            for en in names:
                entity.append({
                    'id': eid,
                    'mention': en,
                    'type': tpy
                })

        for e in entity:
            if e["id"] in entity2id:
                idx = entity2id[e["id"]]
                vertexSet[idx].append({
                    "type": e["type"],
                    "name": e["mention"],
                    "id": e["id"],
                })
        
        samples["vertexSet"] = vertexSet
        results.append(samples)
    
    # 保存
    with open("./test_bio.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    


def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res

def read_docred(file_in, tokenizer, max_seq_length=1024, Type_Enhance=False):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    re_fre = np.zeros(len(docred_rel2id)-1) # exclude NA
    if file_in == "":
        return None
    with open(file_in, "r") as user:
        data = json.load(user)

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        if len(entities) <= 1:
            continue
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        train_triple = {}
        in_train_dict = defaultdict(bool)
        if "labels" in sample:
            for label in sample['labels']:
                if label['h'] == label['t']:
                    continue
                r = int(docred_rel2id[label['r']])

                for n1 in sample['vertexSet'][label['h']]:
                    for n2 in sample['vertexSet'][label['t']]:
                        if (n1['name'], n2['name'], label['r']) in docred_fact_in_train:
                            in_train_dict[(label['h'], label['t'],  r)] = True

                evidence = label.get('evidence', [])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))
        relations, hts, in_trains = [], [], []
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            in_train = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                if relation[mention["relation"]] == 0:
                    re_fre[mention["relation"] - 1] += 1
                relation[mention["relation"]] = 1
                in_train[mention["relation"]] = int(in_train_dict[(h, t, mention["relation"] )])
                evidence = mention["evidence"]
            relations.append(relation)
            in_trains.append(in_train)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    in_train = [0] * len(docred_rel2id)
                    relation = [1] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    in_trains.append(in_train)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)

        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'in_trains': in_trains,
                   'title': sample['title'],
                   }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    re_fre = 1. * re_fre / (pos_samples + neg_samples)
    return features, re_fre

