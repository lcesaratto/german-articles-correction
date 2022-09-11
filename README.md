References:

1. Deepset.ai German BERT model: https://www.deepset.ai/german-bert
2. Corresponding publication: https://aclanthology.org/2020.coling-main.598/
3. Hugging Face pretrained german BERT model: https://huggingface.co/bert-base-german-cased
4. Grammar and spelling checker: https://textgears.com/grammatik-uberprufen-online
5. German Wikipedia dump: https://dumps.wikimedia.org/dewiki/20220520/
6. Preprocessed wikipedia corpus: https://github.com/GermanT5/wikipedia2corpus
7. https://v2.spacy.io/api/annotation
8. https://spacy.io/api/morphology#morphanalysis
9. https://spacy.io/usage/linguistic-features#pos-tagging
10. https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertForMaskedLM
11. https://spacy.io/models/de#de_core_news_lg
12. https://explosion.ai/blog/ud-benchmarks-v3-2
13. https://universaldependencies.org/treebanks/de_hdt/index.html
14. https://stackoverflow.com/questions/64156202/add-dense-layer-on-top-of-huggingface-bert-model
15. https://huggingface.co/docs/transformers/v4.21.3/en/model_doc/bert#transformers.BertModel
16. https://sunilchomal.github.io/GECwBERT/

pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

python -m spacy download de_dep_news_trf
python -m spacy download de_core_news_md
python -m spacy download de_core_news_lg
