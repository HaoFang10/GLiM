import json
import re
import os
from openai import OpenAI
from transformers.utils.versions import require_version
from tqdm import tqdm


require_version("openai>=1.5.0", "To fix: pip install openai>=1.5.0")

def infer(path, port=8000, task="", results_path=""):
    # change to your custom port
    client = OpenAI(
        api_key="0",
        base_url="http://localhost:{}/v1".format(os.environ.get("API_PORT", port)),
    )

    # load
    test = json.load(open(path))

    with open(results_path, "a", encoding="utf-8") as re_file:

        re1 = []
        failed = []
        for idx, item in tqdm(enumerate(test)):

            ins = item["instruction"]
            pattern1 = r"Only return the JSON"
            match1= re.search(pattern1, ins)

            if match1 is None:
                input = ins + task
            else:
                input = ins[:match1.start()] + task

            print(input)
            try:
                ans_dict = {}
                ans_list = []
        
                messages = []
                messages.append({"role": "user", "content": input})
                result = client.chat.completions.create(messages=messages, model="test")
                ans = result.choices[0].message.content;
                ans_list.append(ans)
                
                print("ANS\n\n")
                print(ans_list[0])
                print("\n\n")

                tmp = {"idx": idx, 
                    "instruction": input,
                    "answer": ans_list, 
                    # "info": item["info"],
                    }
                re1.append(tmp)
                
                json.dump(tmp, re_file, ensure_ascii=False)
                re_file.write("\n") 
            
            except Exception as e:
                print(e)
                failed.append({"idx": idx, 
                        "instruction": item["instruction"],
                        "error": str(e),
                        # "info": item["info"],
                        })
            

def main():
    path = ["input/biored.json",
            "input/chemdisgene1.json",
           ]

    rep = ["output/biored_results.json",
           "output/chemdisgene_results.json",
           ]
    
    port = 8000

    for idx, p in enumerate(path):
        if idx == 0:
            continue
        infer(p, port, "Let's think step by step", results_path = rep[idx])


if __name__ == "__main__":
    main()