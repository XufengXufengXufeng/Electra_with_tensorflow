import re
import numpy as np
import tensorflow as tf
def get_all_han_words(l):
    results = []
    for m in re.finditer(r"[\u4e00-\u9fa5\ ]+",l):
        results.append(m.span())
    return {i:("han",i,s) for i,s in results}
def get_all_other_words(l):
    results = []
    for m in re.finditer(r"[A-Za-z0-9]+",l):
        results.append(m.span())
    return  {i:("other",i,s) for i,s in results}
def get_sep_sentence(l):
    diction = get_all_han_words(l)
    diction.update(get_all_other_words(l))
    results = []
    for k in sorted(list(diction.keys())):
        flag,start,end = diction[k]
        if flag=="han":
            results+=list(l[start:end])
        else:
            results.append(l[start:end])
    return results
def get_dicts(data):
    all_chars = set()
    for l in data:
        all_chars|=set(get_sep_sentence(l[0]))
    all_chars = list(all_chars)
    char2id = {c:i for i,c in enumerate(all_chars)}
    id2char = {i:c for i,c in enumerate(all_chars)}
    empty_pos = char2id[" "]
    char2id[id2char[0]]=empty_pos
    id2char[empty_pos]=id2char[0]
    id2char[0]=" "
    char2id[" "]=0
    return char2id,id2char
def idify_sentences(sentence,char2id):
    trans = []
    for s in get_sep_sentence(sentence):
        trans.append(char2id[s])
    return trans
def get_token_labels(sentence,keys):
    labels = np.ones(len(sentence))
    for k in keys:
        search = re.search(k,sentence)
        if type(search).__name__!="NoneType":
            span = search.span()
            labels[span[0]:span[1]]=0.0
    return labels.tolist()
def get_train_data(data,char2id):
    train_data = []
    for l in data:
        seq = idify_sentences(l[0],char2id)
        token_labels = get_token_labels(l[0],l[2])
        train_data.append((seq,token_labels,l[1]))
    return train_data
def data_batcher(data,max_len,batch_size=32):
    for i in range(0,len(data),batch_size):
        cut = data[i:i+batch_size]
        seqs = []
        #seq_lens = []
        token_labels = []
        targets = []
        for s in cut:
            seqs.append(s[0])
            #seq_lens.append(s[1])
            token_labels.append(s[1])
            targets.append(s[2])
        seqs = tf.keras.preprocessing.sequence.pad_sequences(seqs,maxlen=max_len,padding="post")
        token_labels = tf.keras.preprocessing.sequence.pad_sequences(token_labels,
                                                                     maxlen=max_len,padding="post")
        yield seqs,token_labels,targets