import pandas as pd
import spacy
import de_core_news_lg

pd.set_option('display.max_colwidth', None)

df = pd.read_csv("data/dewiki-20220201-clean-01.csv", nrows=100)
print(df.head(5))

if spacy.prefer_gpu(): print("YES GPU")

nlp = de_core_news_lg.load()
print(nlp.pipe_names)