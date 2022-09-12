import pandas as pd
import spacy
import de_core_news_lg
from tqdm import tqdm


pd.set_option('display.max_colwidth', None)


class DataGenerator:
    @staticmethod
    def load_and_save_raw_data(run=False):
        # with open('data/dewiki-20220201-clean.txt', 'r') as f:
        #     while True:
        #         next_line = f.readline()
        #         if not next_line:
        #             break
        #         print(next_line.strip())
        if run:
            for i in range(6):
                df = pd.read_fwf('data/dewiki-20220201-clean.txt',
                                names=["sentences"],
                                widths=[-1],
                                skiprows=10000000*i,
                                nrows=10000000)
                print(df.shape)
                df.to_csv(f"data/dewiki-20220201-clean-0{i+1}.csv", index=False)

    @staticmethod
    def generate_training_data(run=False):
        if run:
            df = pd.read_csv("data/dewiki-20220201-clean-01.csv", nrows=100000)
            print(df.head(5))

            spacy.prefer_gpu()
            nlp = de_core_news_lg.load()
            disabled = nlp.select_pipes(enable=['tok2vec', 'tagger', 'morphologizer', 'parser'])
            print("DISABLED:", disabled)
            print("ENABLED: ", nlp.pipe_names)

            # doc = nlp("Sein Haus in Deutschland. Ein Haus. Meiner Freundin Auto. Der Mann. Den Mann. Haus des Geldes")
            # doc = nlp("Sein Haus in Deutschland. Der Mann.")
            # print(doc)
            # for token in doc:
            #     print(token.text, token.pos_,token.tag_, token.dep_, token.morph, token.whitespace_)
            #     print(doc[:token.i].text_with_ws + "[MASK]" + token.whitespace_ + doc[token.i + 1:].text)

            result = {
                "sentence": [],
                "masked_sentence": [],
                "masked_token": [],
                "pos_tag": [],
                "dependency_tag": [],
                "morph_case": [],
                "morph_definite": [],
                "morph_gender": [],
                "morph_number": [],
                "morph_pron_type": [],
                "morph_poss": [],
            }

            for idx in tqdm(range(df.shape[0])):
                doc = nlp(df.iloc[idx]["sentences"])
                for token in doc:
                    if token.pos_ == "DET":
                        result["sentence"].append(doc)
                        result["masked_sentence"].append(doc[:token.i].text_with_ws + "[MASK]" + token.whitespace_ + doc[token.i + 1:].text)
                        result["masked_token"].append(token.text)
                        result["pos_tag"].append(token.tag_)
                        result["dependency_tag"].append(token.dep_)
                        if token.morph.get("Case"):
                            result["morph_case"].append(token.morph.get("Case")[0])
                        else:
                            result["morph_case"].append("")
                        if token.morph.get("Definite"):
                            result["morph_definite"].append(token.morph.get("Definite")[0])
                        else:
                            result["morph_definite"].append("")
                        if token.morph.get("Gender"):
                            result["morph_gender"].append(token.morph.get("Gender")[0])
                        else:
                            result["morph_gender"].append("")
                        if token.morph.get("Number"):
                            result["morph_number"].append(token.morph.get("Number")[0])
                        else:
                            result["morph_number"].append("")
                        if token.morph.get("PronType"):
                            result["morph_pron_type"].append(token.morph.get("PronType")[0])
                        else:
                            result["morph_pron_type"].append("")
                        if token.morph.get("Poss"):
                            result["morph_poss"].append(token.morph.get("Poss")[0])
                        else:
                            result["morph_poss"].append("")

            df = pd.DataFrame(result)
            df.to_csv("data/raw_data_short.csv", index=False)


DataGenerator.load_and_save_raw_data(False)

DataGenerator.generate_training_data(False)
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 100000/100000 [16:17<00:00, 102.32it/s]