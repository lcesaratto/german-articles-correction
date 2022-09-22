import pandas as pd
import spacy
import de_core_news_lg
import random
from tqdm import tqdm


pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)


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
                df.to_csv(
                    f"data/dewiki-20220201-clean-0{i+1}.csv", index=False)

    @staticmethod
    def generate_training_data_one_mask_per_sentence(run=False):
        if run:
            df = pd.read_csv("data/dewiki-20220201-clean-01.csv", nrows=100000)
            print(df.head(5))

            spacy.prefer_gpu()
            nlp = de_core_news_lg.load()
            disabled = nlp.select_pipes(
                enable=['tok2vec', 'tagger', 'morphologizer', 'parser'])
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
                        result["masked_sentence"].append(
                            doc[:token.i].text_with_ws + "[MASK]" + token.whitespace_ + doc[token.i + 1:].text)
                        result["masked_token"].append(token.text)
                        result["pos_tag"].append(token.tag_)
                        result["dependency_tag"].append(token.dep_)
                        if token.morph.get("Case"):
                            result["morph_case"].append(
                                token.morph.get("Case")[0])
                        else:
                            result["morph_case"].append("")
                        if token.morph.get("Definite"):
                            result["morph_definite"].append(
                                token.morph.get("Definite")[0])
                        else:
                            result["morph_definite"].append("")
                        if token.morph.get("Gender"):
                            result["morph_gender"].append(
                                token.morph.get("Gender")[0])
                        else:
                            result["morph_gender"].append("")
                        if token.morph.get("Number"):
                            result["morph_number"].append(
                                token.morph.get("Number")[0])
                        else:
                            result["morph_number"].append("")
                        if token.morph.get("PronType"):
                            result["morph_pron_type"].append(
                                token.morph.get("PronType")[0])
                        else:
                            result["morph_pron_type"].append("")
                        if token.morph.get("Poss"):
                            result["morph_poss"].append(
                                token.morph.get("Poss")[0])
                        else:
                            result["morph_poss"].append("")

            df = pd.DataFrame(result)
            df.to_csv(
                "data/raw_data_short_one_mask_per_sentence.csv", index=False)

    @staticmethod
    def generate_training_data_multiple_masks_per_sentence(run=False):
        if run:
            df = pd.read_csv("data/dewiki-20220201-clean-01.csv", nrows=100000)

            spacy.prefer_gpu()
            nlp = de_core_news_lg.load()
            nlp.select_pipes(
                enable=['tok2vec', 'tagger', 'morphologizer', 'parser'])

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
                result["sentence"].append(doc)
                result["masked_token"].append([])
                result["pos_tag"].append([])
                result["dependency_tag"].append([])
                result["morph_case"].append([])
                result["morph_definite"].append([])
                result["morph_gender"].append([])
                result["morph_number"].append([])
                result["morph_pron_type"].append([])
                result["morph_poss"].append([])
                masked_sentence = ""
                last_token_idx = 0

                for token in doc:
                    if token.pos_ == "DET":
                        masked_sentence += doc[last_token_idx +
                                               1: token.i].text_with_ws + "[MASK]" + token.whitespace_
                        last_token_idx = token.i

                        result["masked_token"][idx].append(token.text)
                        result["pos_tag"][idx].append(token.tag_)
                        result["dependency_tag"][idx].append(token.dep_)
                        if token.morph.get("Case"):
                            result["morph_case"][idx].append(
                                token.morph.get("Case")[0])
                        else:
                            result["morph_case"][idx].append("")
                        if token.morph.get("Definite"):
                            result["morph_definite"][idx].append(
                                token.morph.get("Definite")[0])
                        else:
                            result["morph_definite"][idx].append("")
                        if token.morph.get("Gender"):
                            result["morph_gender"][idx].append(
                                token.morph.get("Gender")[0])
                        else:
                            result["morph_gender"][idx].append("")
                        if token.morph.get("Number"):
                            result["morph_number"][idx].append(
                                token.morph.get("Number")[0])
                        else:
                            result["morph_number"][idx].append("")
                        if token.morph.get("PronType"):
                            result["morph_pron_type"][idx].append(
                                token.morph.get("PronType")[0])
                        else:
                            result["morph_pron_type"][idx].append("")
                        if token.morph.get("Poss"):
                            result["morph_poss"][idx].append(
                                token.morph.get("Poss")[0])
                        else:
                            result["morph_poss"][idx].append("")

                masked_sentence += doc[last_token_idx + 1:].text
                result["masked_sentence"].append(masked_sentence)

            df = pd.DataFrame(result)
            df.to_csv(
                "data/raw_data_short_multiple_masks_per_sentence.csv", index=False)

    @staticmethod
    def generate_testing_data_multiple_masks_per_sentence(run=False):
        if run:
            df = pd.read_csv("data/dewiki-20220201-clean-01.csv",
                             skiprows=range(1, 100001), nrows=10000)

            spacy.prefer_gpu()
            nlp = de_core_news_lg.load()
            nlp.select_pipes(
                enable=['tok2vec', 'tagger', 'morphologizer', 'parser'])

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
                result["sentence"].append(doc)
                result["masked_token"].append([])
                result["pos_tag"].append([])
                result["dependency_tag"].append([])
                result["morph_case"].append([])
                result["morph_definite"].append([])
                result["morph_gender"].append([])
                result["morph_number"].append([])
                result["morph_pron_type"].append([])
                result["morph_poss"].append([])
                masked_sentence = ""
                last_token_idx = 0

                for token in doc:
                    if token.pos_ == "DET":
                        masked_sentence += doc[last_token_idx +
                                               1: token.i].text_with_ws + "[MASK]" + token.whitespace_
                        last_token_idx = token.i

                        result["masked_token"][idx].append(token.text)
                        result["pos_tag"][idx].append(token.tag_)
                        result["dependency_tag"][idx].append(token.dep_)
                        if token.morph.get("Case"):
                            result["morph_case"][idx].append(
                                token.morph.get("Case")[0])
                        else:
                            result["morph_case"][idx].append("")
                        if token.morph.get("Definite"):
                            result["morph_definite"][idx].append(
                                token.morph.get("Definite")[0])
                        else:
                            result["morph_definite"][idx].append("")
                        if token.morph.get("Gender"):
                            result["morph_gender"][idx].append(
                                token.morph.get("Gender")[0])
                        else:
                            result["morph_gender"][idx].append("")
                        if token.morph.get("Number"):
                            result["morph_number"][idx].append(
                                token.morph.get("Number")[0])
                        else:
                            result["morph_number"][idx].append("")
                        if token.morph.get("PronType"):
                            result["morph_pron_type"][idx].append(
                                token.morph.get("PronType")[0])
                        else:
                            result["morph_pron_type"][idx].append("")
                        if token.morph.get("Poss"):
                            result["morph_poss"][idx].append(
                                token.morph.get("Poss")[0])
                        else:
                            result["morph_poss"][idx].append("")

                masked_sentence += doc[last_token_idx + 1:].text
                result["masked_sentence"].append(masked_sentence)

            df = pd.DataFrame(result)
            df.to_csv(
                "data/test_raw_data_short_multiple_masks_per_sentence.csv", index=False)

    @staticmethod
    def show_info(run=False):
        if run:
            df = pd.read_csv(
                "data/raw_data_short_one_mask_per_sentence.csv", nrows=None)

            df = df[(df["pos_tag"] == "ART") & (
                df["morph_pron_type"] == "Art")]

            lowercase_tokens = ["die", "der", "das", "den", "dem", "des"]

            print(df[df["masked_token"].isin(lowercase_tokens)][["masked_token",
                                                                 "morph_definite",
                                                                 "morph_case",
                                                                 "morph_gender",
                                                                 "morph_number"]].drop_duplicates().sort_values(["morph_case",
                                                                                                                 "morph_definite",
                                                                                                                 "morph_case",
                                                                                                                 "morph_gender",
                                                                                                                 "morph_number"]).reset_index(drop=True))

            lowercase_tokens = ["eine", "einen",
                                "ein", "einer", "einem", "eines"]

            print(df[df["masked_token"].isin(lowercase_tokens)][["masked_token",
                                                                 "morph_definite",
                                                                 "morph_case",
                                                                 "morph_gender",
                                                                 "morph_number"]].drop_duplicates().sort_values(["morph_case",
                                                                                                                 "morph_definite",
                                                                                                                 "morph_case",
                                                                                                                 "morph_gender",
                                                                                                                 "morph_number"]).reset_index(drop=True))

    @staticmethod
    def add_wrong_sentences_to_training_data_one_mask_per_sentence(run=False):
        if run:
            df = pd.read_csv(
                "data/raw_data_short_one_mask_per_sentence.csv", nrows=None)

            possible_tokens = {"die": ["der", "das", "den", "dem", "des"],
                               "der": ["die", "das", "den", "dem", "des"],
                               "das": ["die", "der", "den", "dem", "des"],
                               "den": ["die", "der", "das", "dem", "des"],
                               "dem": ["die", "der", "das", "den", "des"],
                               "des": ["die", "der", "das", "den", "dem"],
                               "Die": ["Der", "Das", "Den", "Dem", "Des"],
                               "Der": ["Die", "Das", "Den", "Dem", "Des"],
                               "Das": ["Die", "Der", "Den", "Dem", "Des"],
                               "Den": ["Die", "Der", "Das", "Dem", "Des"],
                               "Dem": ["Die", "Der", "Das", "Den", "Des"],
                               "Des": ["Die", "Der", "Das", "Den", "Dem"], }

            df = df[(df["morph_definite"] == "Def") & (
                df["pos_tag"] == "ART") & (df["morph_pron_type"] == "Art")]
            df = df[df["masked_token"].isin(possible_tokens.keys())]

            print(df["masked_token"].unique())
            print(len(df))

            df["wrong_token"] = df["masked_token"].apply(
                lambda x: random.choice(possible_tokens[x]))
            df["wrong_sentence"] = df.apply(
                lambda x: x["masked_sentence"].replace("[MASK]", x["wrong_token"]), axis=1)

            print(df[["sentence", "masked_sentence", "wrong_sentence"]].iloc[0])

            df.to_csv("data/data_short_one_mask_per_sentence.csv", index=False)

    @staticmethod
    def add_wrong_sentences_to_training_data_multiple_masks_per_sentence(run=False):
        if run:
            df = pd.read_csv(
                "data/raw_data_short_multiple_masks_per_sentence.csv", nrows=None)
            df = df.dropna(subset=["masked_sentence"])

            possible_tokens = {"die": ["der", "das", "den", "dem", "des"],
                               "der": ["die", "das", "den", "dem", "des"],
                               "das": ["die", "der", "den", "dem", "des"],
                               "den": ["die", "der", "das", "dem", "des"],
                               "dem": ["die", "der", "das", "den", "des"],
                               "des": ["die", "der", "das", "den", "dem"],
                               "Die": ["Der", "Das", "Den", "Dem", "Des"],
                               "Der": ["Die", "Das", "Den", "Dem", "Des"],
                               "Das": ["Die", "Der", "Den", "Dem", "Des"],
                               "Den": ["Die", "Der", "Das", "Dem", "Des"],
                               "Dem": ["Die", "Der", "Das", "Den", "Des"],
                               "Des": ["Die", "Der", "Das", "Den", "Dem"], }

            df["wrong_sentence"] = ["" for _ in range(len(df))]

            df["masked_token"] = df["masked_token"].apply(eval)
            df["pos_tag"] = df["pos_tag"].apply(eval)
            df["dependency_tag"] = df["dependency_tag"].apply(eval)
            df["morph_case"] = df["morph_case"].apply(eval)
            df["morph_definite"] = df["morph_definite"].apply(eval)
            df["morph_gender"] = df["morph_gender"].apply(eval)
            df["morph_number"] = df["morph_number"].apply(eval)
            df["morph_pron_type"] = df["morph_pron_type"].apply(eval)
            df["morph_poss"] = df["morph_poss"].apply(eval)

            for idx in tqdm(range(df.shape[0])):

                masked_sentence = df.iloc[idx]["masked_sentence"]
                wrong_sentence = df.iloc[idx]["masked_sentence"]
                masked_token_list = []
                pos_tag_list = []
                dependency_tag_list = []
                morph_case_list = []
                morph_definite_list = []
                morph_gender_list = []
                morph_number_list = []
                morph_pron_type_list = []
                morph_poss_list = []

                for token_idx, masked_token in enumerate(df.iloc[idx]["masked_token"]):

                    pos_tag = df.iloc[idx]["pos_tag"][token_idx]
                    dependency_tag = df.iloc[idx]["dependency_tag"][token_idx]
                    morph_case = df.iloc[idx]["morph_case"][token_idx]
                    morph_definite = df.iloc[idx]["morph_definite"][token_idx]
                    morph_gender = df.iloc[idx]["morph_gender"][token_idx]
                    morph_number = df.iloc[idx]["morph_number"][token_idx]
                    morph_pron_type = df.iloc[idx]["morph_pron_type"][token_idx]
                    morph_poss = df.iloc[idx]["morph_poss"][token_idx]

                    if (morph_definite == "Def") and (pos_tag == "ART") and (masked_token in possible_tokens.keys()):

                        masked_sentence = masked_sentence.replace(
                            "[MASK]", "[NOMASK]", 1)
                        wrong_sentence = wrong_sentence.replace(
                            "[MASK]", random.choice(possible_tokens[masked_token]), 1)

                        masked_token_list.append(masked_token)
                        pos_tag_list.append(pos_tag)
                        dependency_tag_list.append(dependency_tag)
                        morph_case_list.append(morph_case)
                        morph_definite_list.append(morph_definite)
                        morph_gender_list.append(morph_gender)
                        morph_number_list.append(morph_number)
                        morph_pron_type_list.append(morph_pron_type)
                        morph_poss_list.append(morph_poss)

                    else:

                        masked_sentence = masked_sentence.replace(
                            "[MASK]", masked_token, 1)
                        wrong_sentence = wrong_sentence.replace(
                            "[MASK]", masked_token, 1)

                masked_sentence = masked_sentence.replace(
                    "[NOMASK]", "[MASK]")
                df["masked_sentence"][idx] = masked_sentence
                df["wrong_sentence"][idx] = wrong_sentence

                df["masked_token"][idx] = masked_token_list
                df["pos_tag"][idx] = pos_tag_list
                df["dependency_tag"][idx] = dependency_tag_list
                df["morph_case"][idx] = morph_case_list
                df["morph_definite"][idx] = morph_definite_list
                df["morph_gender"][idx] = morph_gender_list
                df["morph_number"][idx] = morph_number_list
                df["morph_pron_type"][idx] = morph_pron_type_list
                df["morph_poss"][idx] = morph_poss_list

            df = df[df["masked_sentence"].str.contains("[MASK]", regex=False)]
            df.dropna(subset=["wrong_sentence"], inplace=True)
            print(df.shape[0])
            df.to_csv(
                "data/data_short_multiple_masks_per_sentence.csv", index=False)

    @staticmethod
    def add_partially_wrong_sentences_to_training_data_multiple_masks_per_sentence(run=False):
        if run:
            df = pd.read_csv(
                "data/raw_data_short_multiple_masks_per_sentence.csv", nrows=None)
            df = df.dropna(subset=["masked_sentence"])

            possible_tokens = {"die": ["die", "der", "das", "den", "dem", "des"],
                               "der": ["die", "der", "das", "den", "dem", "des"],
                               "das": ["die", "der", "das", "den", "dem", "des"],
                               "den": ["die", "der", "das", "den", "dem", "des"],
                               "dem": ["die", "der", "das", "den", "dem", "des"],
                               "des": ["die", "der", "das", "den", "dem", "des"],
                               "Die": ["Die", "Der", "Das", "Den", "Dem", "Des"],
                               "Der": ["Die", "Der", "Das", "Den", "Dem", "Des"],
                               "Das": ["Die", "Der", "Das", "Den", "Dem", "Des"],
                               "Den": ["Die", "Der", "Das", "Den", "Dem", "Des"],
                               "Dem": ["Die", "Der", "Das", "Den", "Dem", "Des"],
                               "Des": ["Die", "Der", "Das", "Den", "Dem", "Des"], }

            df["wrong_sentence"] = ["" for _ in range(len(df))]

            df["masked_token"] = df["masked_token"].apply(eval)
            df["pos_tag"] = df["pos_tag"].apply(eval)
            df["dependency_tag"] = df["dependency_tag"].apply(eval)
            df["morph_case"] = df["morph_case"].apply(eval)
            df["morph_definite"] = df["morph_definite"].apply(eval)
            df["morph_gender"] = df["morph_gender"].apply(eval)
            df["morph_number"] = df["morph_number"].apply(eval)
            df["morph_pron_type"] = df["morph_pron_type"].apply(eval)
            df["morph_poss"] = df["morph_poss"].apply(eval)

            for idx in tqdm(range(df.shape[0])):

                masked_sentence = df.iloc[idx]["masked_sentence"]
                wrong_sentence = df.iloc[idx]["masked_sentence"]
                masked_token_list = []
                pos_tag_list = []
                dependency_tag_list = []
                morph_case_list = []
                morph_definite_list = []
                morph_gender_list = []
                morph_number_list = []
                morph_pron_type_list = []
                morph_poss_list = []

                for token_idx, masked_token in enumerate(df.iloc[idx]["masked_token"]):

                    pos_tag = df.iloc[idx]["pos_tag"][token_idx]
                    dependency_tag = df.iloc[idx]["dependency_tag"][token_idx]
                    morph_case = df.iloc[idx]["morph_case"][token_idx]
                    morph_definite = df.iloc[idx]["morph_definite"][token_idx]
                    morph_gender = df.iloc[idx]["morph_gender"][token_idx]
                    morph_number = df.iloc[idx]["morph_number"][token_idx]
                    morph_pron_type = df.iloc[idx]["morph_pron_type"][token_idx]
                    morph_poss = df.iloc[idx]["morph_poss"][token_idx]

                    if (morph_definite == "Def") and (pos_tag == "ART") and (masked_token in possible_tokens.keys()):

                        masked_sentence = masked_sentence.replace(
                            "[MASK]", "[NOMASK]", 1)
                        wrong_sentence = wrong_sentence.replace(
                            "[MASK]", random.choice(possible_tokens[masked_token]), 1)

                        masked_token_list.append(masked_token)
                        pos_tag_list.append(pos_tag)
                        dependency_tag_list.append(dependency_tag)
                        morph_case_list.append(morph_case)
                        morph_definite_list.append(morph_definite)
                        morph_gender_list.append(morph_gender)
                        morph_number_list.append(morph_number)
                        morph_pron_type_list.append(morph_pron_type)
                        morph_poss_list.append(morph_poss)

                    else:

                        masked_sentence = masked_sentence.replace(
                            "[MASK]", masked_token, 1)
                        wrong_sentence = wrong_sentence.replace(
                            "[MASK]", masked_token, 1)

                masked_sentence = masked_sentence.replace(
                    "[NOMASK]", "[MASK]")
                df["masked_sentence"][idx] = masked_sentence
                df["wrong_sentence"][idx] = wrong_sentence

                df["masked_token"][idx] = masked_token_list
                df["pos_tag"][idx] = pos_tag_list
                df["dependency_tag"][idx] = dependency_tag_list
                df["morph_case"][idx] = morph_case_list
                df["morph_definite"][idx] = morph_definite_list
                df["morph_gender"][idx] = morph_gender_list
                df["morph_number"][idx] = morph_number_list
                df["morph_pron_type"][idx] = morph_pron_type_list
                df["morph_poss"][idx] = morph_poss_list

            df = df[df["masked_sentence"].str.contains("[MASK]", regex=False)]
            df.dropna(subset=["wrong_sentence"], inplace=True)
            print(df.shape[0])
            df.to_csv(
                "data/data_short_multiple_masks_per_sentence_partially_wrong.csv", index=False)

    @staticmethod
    def add_partially_wrong_sentences_to_testing_data_multiple_masks_per_sentence(run=False):
        if run:
            df = pd.read_csv(
                "data/test_raw_data_short_multiple_masks_per_sentence.csv", nrows=None)
            df = df.dropna(subset=["masked_sentence"])

            possible_tokens = {"die": ["die", "der", "das", "den", "dem", "des"],
                               "der": ["die", "der", "das", "den", "dem", "des"],
                               "das": ["die", "der", "das", "den", "dem", "des"],
                               "den": ["die", "der", "das", "den", "dem", "des"],
                               "dem": ["die", "der", "das", "den", "dem", "des"],
                               "des": ["die", "der", "das", "den", "dem", "des"],
                               "Die": ["Die", "Der", "Das", "Den", "Dem", "Des"],
                               "Der": ["Die", "Der", "Das", "Den", "Dem", "Des"],
                               "Das": ["Die", "Der", "Das", "Den", "Dem", "Des"],
                               "Den": ["Die", "Der", "Das", "Den", "Dem", "Des"],
                               "Dem": ["Die", "Der", "Das", "Den", "Dem", "Des"],
                               "Des": ["Die", "Der", "Das", "Den", "Dem", "Des"], }

            df["wrong_sentence"] = ["" for _ in range(len(df))]

            df["masked_token"] = df["masked_token"].apply(eval)
            df["pos_tag"] = df["pos_tag"].apply(eval)
            df["dependency_tag"] = df["dependency_tag"].apply(eval)
            df["morph_case"] = df["morph_case"].apply(eval)
            df["morph_definite"] = df["morph_definite"].apply(eval)
            df["morph_gender"] = df["morph_gender"].apply(eval)
            df["morph_number"] = df["morph_number"].apply(eval)
            df["morph_pron_type"] = df["morph_pron_type"].apply(eval)
            df["morph_poss"] = df["morph_poss"].apply(eval)

            for idx in tqdm(range(df.shape[0])):

                masked_sentence = df.iloc[idx]["masked_sentence"]
                wrong_sentence = df.iloc[idx]["masked_sentence"]
                masked_token_list = []
                pos_tag_list = []
                dependency_tag_list = []
                morph_case_list = []
                morph_definite_list = []
                morph_gender_list = []
                morph_number_list = []
                morph_pron_type_list = []
                morph_poss_list = []

                for token_idx, masked_token in enumerate(df.iloc[idx]["masked_token"]):

                    pos_tag = df.iloc[idx]["pos_tag"][token_idx]
                    dependency_tag = df.iloc[idx]["dependency_tag"][token_idx]
                    morph_case = df.iloc[idx]["morph_case"][token_idx]
                    morph_definite = df.iloc[idx]["morph_definite"][token_idx]
                    morph_gender = df.iloc[idx]["morph_gender"][token_idx]
                    morph_number = df.iloc[idx]["morph_number"][token_idx]
                    morph_pron_type = df.iloc[idx]["morph_pron_type"][token_idx]
                    morph_poss = df.iloc[idx]["morph_poss"][token_idx]

                    if (morph_definite == "Def") and (pos_tag == "ART") and (masked_token in possible_tokens.keys()):

                        masked_sentence = masked_sentence.replace(
                            "[MASK]", "[NOMASK]", 1)
                        wrong_sentence = wrong_sentence.replace(
                            "[MASK]", random.choice(possible_tokens[masked_token]), 1)

                        masked_token_list.append(masked_token)
                        pos_tag_list.append(pos_tag)
                        dependency_tag_list.append(dependency_tag)
                        morph_case_list.append(morph_case)
                        morph_definite_list.append(morph_definite)
                        morph_gender_list.append(morph_gender)
                        morph_number_list.append(morph_number)
                        morph_pron_type_list.append(morph_pron_type)
                        morph_poss_list.append(morph_poss)

                    else:

                        masked_sentence = masked_sentence.replace(
                            "[MASK]", masked_token, 1)
                        wrong_sentence = wrong_sentence.replace(
                            "[MASK]", masked_token, 1)

                masked_sentence = masked_sentence.replace(
                    "[NOMASK]", "[MASK]")
                df["masked_sentence"][idx] = masked_sentence
                df["wrong_sentence"][idx] = wrong_sentence

                df["masked_token"][idx] = masked_token_list
                df["pos_tag"][idx] = pos_tag_list
                df["dependency_tag"][idx] = dependency_tag_list
                df["morph_case"][idx] = morph_case_list
                df["morph_definite"][idx] = morph_definite_list
                df["morph_gender"][idx] = morph_gender_list
                df["morph_number"][idx] = morph_number_list
                df["morph_pron_type"][idx] = morph_pron_type_list
                df["morph_poss"][idx] = morph_poss_list

            df = df[df["masked_sentence"].str.contains("[MASK]", regex=False)]
            df.dropna(subset=["wrong_sentence"], inplace=True)
            print(df.shape[0])
            df.to_csv(
                "data/test_data_short_multiple_masks_per_sentence_partially_wrong.csv", index=False)


DataGenerator.load_and_save_raw_data(False)

DataGenerator.generate_training_data_one_mask_per_sentence(False)
# 100%|████████████████████████████████████████████████████████| 100000/100000 [16:17<00:00, 102.32it/s]

DataGenerator.generate_training_data_multiple_masks_per_sentence(False)
# 100%|████████████████████████████████████████████████████████| 100000/100000 [21:23<00:00, 77.92it/s]

DataGenerator.show_info(False)

DataGenerator.add_wrong_sentences_to_training_data_one_mask_per_sentence(False)

DataGenerator.add_wrong_sentences_to_training_data_multiple_masks_per_sentence(
    False)
# 100%|████████████████████████████████████████████████████████| 99650/99650 [06:23<00:00, 260.18it/s]

DataGenerator.add_partially_wrong_sentences_to_training_data_multiple_masks_per_sentence(
    False)
# 100%|████████████████████████████████████████████████████████| 99650/99650 [06:23<00:00, 260.18it/s]

DataGenerator.generate_testing_data_multiple_masks_per_sentence(False)
# 100%|████████████████████████████████████████████████████████| 10000/10000 [02:19<00:00, 71.49it/s]

DataGenerator.add_partially_wrong_sentences_to_testing_data_multiple_masks_per_sentence(
    False)
# 100%|████████████████████████████████████████████████████████| 9972/9972 [00:31<00:00, 318.39it/s]
