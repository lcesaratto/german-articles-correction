import random
import pandas as pd


pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

df = pd.read_csv("data/raw_data_short.csv")  # , nrows=100)

# print(df[["masked_token",
#             "pos_tag",
#             "morph_definite",
#             "morph_case",
#             "morph_gender",
#             "morph_number",
#             "morph_pron_type",
#             "morph_poss"]].drop_duplicates().sort_values(["pos_tag",
#                                                             "morph_definite",
#                                                             "morph_case",
#                                                             "morph_gender",
#                                                             "morph_number",
#                                                             "morph_pron_type",
#                                                             "morph_poss"]).reset_index(drop=True))
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

df = df[(df["morph_definite"] == "Def") & (df["pos_tag"] == "ART")]
df = df[df["masked_token"].isin(possible_tokens.keys())]

lowercase_tokens = ["die", "der", "das", "den", "dem", "des"]
print(df[df["masked_token"].isin(lowercase_tokens)][["masked_token",
                                                     "morph_case",
                                                     "morph_gender",
                                                     "morph_number"]].drop_duplicates().sort_values(["morph_case",
                                                                                                    "morph_gender",
                                                                                                     "morph_number"]).reset_index(drop=True))
print(df["masked_token"].unique())
print(len(df))

df["wrong_token"] = df["masked_token"].apply(
    lambda x: random.choice(possible_tokens[x]))
df["wrong_sentence"] = df.apply(
    lambda x: x["masked_sentence"].replace("[MASK]", x["wrong_token"]), axis=1)

print(df[["sentence", "masked_sentence", "wrong_sentence"]].iloc[0])

df.to_csv("data/data_short.csv", index=False)
