import os
import json
import pandas as pd

base_path = "/home/arda/yina/sw/LLMSuperWeight/outputs/meta-llama/Meta-Llama-3-8B-Instruct"

def walk_directory(dir_path):
    result_list = []
    for root, dirs, files in os.walk(dir_path):
        if files:
            for file in files:
                path = os.path.join(root, file)
                root_list = root.split("/")
                if len(root_list) == 12:
                    clip_method = root_list[10]
                    
                    restore_sw = clip_method.split("-")[-1]
                else:
                    continue
                clip_method_list = clip_method.split("_")
                blk_size = clip_method.split("_")[3]
                clip_method_short = f"{clip_method_list[4]}_{clip_method_list[6]}" if clip_method_list[4] == "no" else f"{clip_method_list[4]}_{clip_method_list[5]}_{clip_method_list[6]}"
                
                with open(path, "r") as f:
                    data = json.load(f)
                    results = data.get("results", None)
                    arc_c = results.get("arc_challenge", {}).get("acc,none", None)
                    arc_e = results.get("arc_easy", {}).get("acc,none", None)
                    lamb = results.get("lambada_openai", {}).get("acc,none", None)
                    sciq = results.get("sciq", {}).get("acc,none", None)
                    wiki = results.get("wikitext", {})
                    wiki_word_ppl = wiki.get("word_perplexity,none", None)
                    wiki_byte_ppl = wiki.get("byte_perplexity,none", None)
                    result_list.append([
                        blk_size, clip_method_short, restore_sw, "{:.4f}".format(arc_c * 100), "{:.4f}".format(arc_e * 100),
                        "{:.4f}".format(lamb * 100), "{:.4f}".format(sciq * 100), "{:.4f}".format(wiki_byte_ppl), "{:.4f}".format(wiki_word_ppl)
                    ])
                print(f"root: {root}, file {file}")
            print("-" * 40)
    df = pd.DataFrame(result_list, columns=["blk_size", "clip_method", "restore_sw", "arc_c", "arc_e", "lamb", "sciq", "wiki_byte_ppl", "wiki_word_ppl"])
    df_sorted = df.sort_values(by=['blk_size', 'clip_method', 'restore_sw'])
    print(df_sorted)
    df_sorted.to_csv(f'Meta-Llama-3-8B-Instruct-results.csv', mode='w', index=False)

# Example usage
walk_directory(base_path)
