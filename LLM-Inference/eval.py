import os
import json
from tqdm import tqdm

def official_evaluate(tmp, path, tag, dataset):

    rlist = json.load(open('{}/meta/rel2id.json'.format(path), 'r'))
    r_info = json.load(open('{}/rel_info.json'.format(path), 'r'))

    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = {}

    truth = json.load(open(os.path.join(path, tag + ".json")))


    std = {}
    tot_evidences = 0
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label.get('evidence', []))
            tot_evidences += len(label.get('evidence', []))

    tot_relations = len(std)
 
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0

    ne_none = 0

    titleset2 = set([])

    error_sample = []
    correct_sample = []

    tht = set([])

    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        tht.add((title, h_idx, t_idx))

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if dataset == 'docred' and (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1

            correct_sample.append({
                    "title": title,
                    "h_idx": str(h_idx) + ":" + title2vectexSet[title][h_idx][0]["name"],
                    "t_idx": str(t_idx) + ":" + title2vectexSet[title][t_idx][0]["name"],
                    "r": r_info[x['r']] + " " + x['r'],
                    "r_s": x["r_s"] if "r_s" in x.keys() else 0,
                    "na_th": x["na_th"] if "na_th" in x.keys() else 0,
                })
        else:
            # pass
            tr = []
            for r in rlist.keys():
                if (title, r, h_idx, t_idx) in std:
                    tr.append(r)
            if len(tr) == 0:
                ne_none +=1
                error_sample.append({
                    "title": title,
                    "h_idx": str(h_idx) + ":" + title2vectexSet[title][h_idx][0]["name"],
                    "t_idx": str(t_idx) + ":" + title2vectexSet[title][t_idx][0]["name"],
                    "r": r_info[x['r']] + " " + x['r'],
                    "r_s": x["r_s"] if "r_s" in x.keys() else 0,
                    "na_th": x["na_th"] if "na_th" in x.keys() else 0,
                })


    missing_sample = []
    for (title, r, h_idx, t_idx) in std.keys():
        tar = {
            "title": title,
            "h_idx": h_idx,
            "t_idx": t_idx,
            "r": r,
        }
        if tar not in submission_answer and (title, h_idx, t_idx) in tht:
            tmp =  (title, h_idx, t_idx)
            
            missing_sample.append({
                    "title": title,
                    "h_idx": str(h_idx) + ":" + title2vectexSet[title][h_idx][0]["name"],
                    "t_idx": str(t_idx) + ":" + title2vectexSet[title][t_idx][0]["name"],
                    "r": r_info[r] + " " + r,
                    "r_s": thtr2info[tmp]['r_s'] if tmp in thtr2info.keys() else 0,
                    "na_th":  thtr2info[tmp]['na_th'] if tmp in thtr2info.keys() else 0,
                })

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)
    print('re_p:{}, re_r:{}'.format(re_p, re_r))
    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / max(1, tot_evidences)
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (
                len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (
                len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train, re_p, re_r




    test_data = []
    llm_path = path
    llm_data = json.load(open(llm_path, 'r'))
    print(len(llm_data))
    for item in llm_data:
        for k, v in item["info"]["r_s"].items():
            test_data.append({
                "title": item["info"]["title"],
                "h_idx": item["info"]["h_idx"], 
                "t_idx": item["info"]["t_idx"], 
                "r": k,
                # "evidence": [],
            })
    print(len(test_data))
    test_f1("", dataset, test_data)
    return test_data

def test_f1(path, dataset, tmp=None):
    
    tag = 'test'

    if tmp is None:
        test_data = json.load(open(path, 'r'))
    else:
        test_data = tmp


    data_dir = "../dataset/dataset_{}".format(dataset)
    
    best_f1, _, best_f1_ign, _, p, r = official_evaluate(test_data, data_dir, tag, dataset=dataset)

    output = {
        tag + "_F1"    : best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        tag + "_p": p * 100,
        tag + "_r": r * 100,
    }
    print(output)
    return output






