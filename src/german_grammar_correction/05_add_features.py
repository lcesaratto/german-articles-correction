import spacy
import de_core_news_lg


spacy.prefer_gpu()
nlp = de_core_news_lg.load()
disabled = nlp.select_pipes(
    enable=['tok2vec', 'tagger', 'morphologizer', 'parser'])

doc_wrong = nlp(
    "Den Hund des Mannes ist krank. Den Hund hat wahrscheinlich der Schokolade, die auf den Boden lag, gegessen.")
doc_predicted = nlp(
    "Der Hund des Mannes ist krank. Der Hund hat wahrscheinlich die Schokolade, die auf dem Boden lag, gegessen.")
for token_wrong, token_pred in zip(doc_wrong, doc_predicted):
    if token_wrong.text != token_pred.text:
        print(token_pred.text, token_pred.morph)
