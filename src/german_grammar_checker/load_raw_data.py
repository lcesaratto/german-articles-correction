import wiki_dump_parser as parser
import pandas as pd


parser.xml_to_csv('data/dewiki-20220520-pages-articles-multistream.xml')

df = pd.read_csv('dump.csv', quotechar='|', index_col = False)
df['timestamp'] = pd.to_datetime(df['timestamp'],format='%Y-%m-%dT%H:%M:%SZ')

print(df.head(10))