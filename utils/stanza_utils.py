import stanza
import torch

# from predict import get_clip_embeddings
from torch.nn import functional as F
stanza.download('en')
nlp = stanza.Pipeline('en')

MAIN_POS = ['NOUN', 'VERB', 'ADJ', 'PROPN', 'NUM']

def find_main_words(sent, start_idx, tokenizer):
    doc = nlp(sent)
    id_map = {}
    cursor = start_idx + 1
    main_words_ids = []
    for word in doc.sentences[0].words:
        temp = cursor
        cursor += len(tokenizer(word.text, return_tensors="pt").input_ids[0]) - 1
        id_map[word.id] = list(range(temp, cursor))
        if word.pos in MAIN_POS:
            main_words_ids.append(word.id)
    return main_words_ids, id_map

def DFS_right(node, variables: list):
    if node.is_leaf() or variables[1] != '':
        return
    if node.label == 'NN':
        variables[1] = str(node.children[0])
        return
    for i in range(len(node.children)):
        idx = len(node.children) - i - 1
        DFS_right(node.children[idx], variables)


def DFS_left(node, variables: list):
    if node.is_leaf() or variables[1] != '':
        return
    if node.label == 'NP':
        variables[0] += 1
    now_find_np = variables[0]
    for child in node.children:
        DFS_left(child, variables)
    if node.label == 'NP' and variables[0] == now_find_np:
        # find rightmost NN
        DFS_right(node, variables)


def find_agent_by_stanza(sent):
    doc = nlp(sent)
    '''
    the type of passing for int and str is passing by value, while for list is passing by reference,
    to modify the value of variables during DFS_left function, we put variables into a list
    variables[0] means the number of occurrence of NP, while variables[1] records the agent
    '''
    variables = [0, '']
    DFS_left(doc.sentences[0].constituency, variables)
    agent = variables[1]
    
    if agent == '':
        for i in range(len(doc.sentences[0].words)):
            idx = len(doc.sentences[0].words) - i - 1
            if doc.sentences[0].words[idx].pos == 'NOUN':
                agent = doc.sentences[0].words[idx].text
                break
    # if agent == '':
    #     for i in range(len(doc.sentences[0].words)):
    #         if doc.sentences[0].words[i].deprel == 'root':
    #             agent = doc.sentences[0].words[i].text
    #             break
    if agent == '':
        agent = '[UNK]'
    return agent

if __name__ == '__main__':
    text = "a red and white checkered table with two wooden chairs"
    print(find_agent_by_stanza(text))
    
    