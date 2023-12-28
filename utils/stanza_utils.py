import stanza
import torch

# from predict import get_clip_embeddings
from torch.nn import functional as F
stanza.download('en')
nlp = stanza.Pipeline('en')


def get_id_map(sent, tokenizer):
    id_map = {}
    cursor = 1
    doc = nlp(sent)
    # get id_map
    for word in doc.sentences[0].words:
        temp = cursor
        cursor += len(tokenizer.tokenize(word.text))
        # cursor += len(tokenizer(word.text, return_tensors="pt").input_ids[0]) - 1
        id_map[word.id] = list(range(temp, cursor))
    return id_map

MAIN_POS = ['NOUN', 'VERB', 'ADJ', 'PROPN', 'NUM']
def find_root_by_stanza(sent):
    doc = nlp(sent)
    main_words_ids = []
    for word in doc.sentences[0].words:
        if word.deprel == 'root':
            root_text = word.lemma
        if word.pos in MAIN_POS:
            main_words_ids.append(word.id)
    return root_text, len(doc.sentences[0].words), main_words_ids

def find_main_words(sent, start_idx, tokenizer):
    doc = nlp(sent)
    id_map = {}
    cursor = start_idx + 1
    main_words_ids = []
    for word in doc.sentences[0].words:
        temp = cursor
        cursor += len(tokenizer(word.text, return_tensors="pt").input_ids[0]) - 2
        id_map[word.id] = list(range(temp, cursor))
        if word.pos in MAIN_POS:
            main_words_ids.append(word.id)
    return main_words_ids, id_map


if __name__ == '__main__':
    # sent = 'a '
    # sidx = 5
    # id_map = get_id_map(sent,tokenizer)
    # print(id_map)
    # focus_ids = []
    root_word, len_phrase, main_words_ids = find_root_by_stanza('some children')
    print(type(root_word))
    # for i in range(sidx, sidx + len_phrase):
    #     focus_ids.extend(id_map[i])
    # # for idx in main_words_ids:
    #     focus_ids.extend(id_map[sidx+idx-1])
    # tokens = tokenizer.tokenize('[CLS]'+sent)
    # for focus_id in focus_ids:
    #     print(tokens[focus_id])