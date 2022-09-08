References:

1. Deepset.ai German BERT model: https://www.deepset.ai/german-bert
2. Corresponding publication: https://aclanthology.org/2020.coling-main.598/
3. Hugging Face pretrained german BERT model: https://huggingface.co/bert-base-german-cased
4. Grammar and spelling checker: https://textgears.com/grammatik-uberprufen-online
5. German Wikipedia dump: https://dumps.wikimedia.org/dewiki/20220520/
6. Preprocessed wikipedia corpus: https://github.com/GermanT5/wikipedia2corpus

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

python -m spacy download de_dep_news_trf
python -m spacy download de_core_news_md
python -m spacy download de_core_news_lg
