import json
from eval import test_f1
from tqdm import tqdm

def statistic(ratio, k):
    path = "./output/chemdisgene_results.json"
    origin_path = "./input/chemdisgene.json"


    origin = json.load(open(origin_path, 'r'))


    with open(path, 'r') as f:
        lines = f.readlines()
    cnt = 0  
    pos = []
    for line in lines:

        line = json.loads(line)
        ans = line['answer'][0]
        idx = line['idx']
        
        v = origin[idx]["info"]["r_s"][origin[idx]["t_re"].split(":^:")[0]]
        ratiok = ratio*k
        t = origin[idx]["info"]["na_th"]*ratiok
        if v <= t:
            # print(v,t)

            continue
        a = ans.split("\n\n")[-1]
        if "not" not in a:
            cnt += 1
            pos.append({
                "title": origin[idx]["info"]["title"],
                "h_idx": origin[idx]["info"]["h_idx"], 
                "t_idx": origin[idx]["info"]["t_idx"], 
                "r": origin[idx]["t_re"].split(":^:")[0],
            })
    print(cnt)
    # test_f1("", "chemdisgene", pos)

    path1 = "./step1/results_for_chemdisgene/result_REGT_test.json"
    t1 = json.load(open(path1, 'r'))
    # test_f1("", "chemdisgene", t1)


    t = t1 + pos
    print(len(t))

    t = list({json.dumps(item, sort_keys=True): item for item in t}.values())
    print(len(t))
    out = test_f1("", "chemdisgene", t)
    return out


if __name__ == '__main__':

    # cdg
    ratio = 0.5470931041421813
    k = 0.9   
    
    out = statistic(ratio, k)
