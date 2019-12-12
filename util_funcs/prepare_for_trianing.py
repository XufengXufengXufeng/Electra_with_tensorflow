import numpy as np


class Reader(object):
    def __init__(self,loc,max_len=512,batch_size=16):
        self.f = open(loc)
        self.loc = loc
        self.batch_size = batch_size
        self.result_buffer = self.f.readlines(max_len*self.batch_size)
        self.buffer_len = len(self.result_buffer)
        self.max_len = max_len
    def read(self):
        end_flag = False
        if self.buffer_len > self.batch_size:
            result = self.result_buffer[:self.batch_size]
            self.result_buffer = np.delete(self.result_buffer,np.arange(self.batch_size)).tolist()
            self.buffer_len = len(self.result_buffer)
        else:
            self.result_buffer += self.f.readlines(self.max_len*self.batch_size)
            if len(self.result_buffer)==self.buffer_len:# 到头了没有读到新数据
                result = self.result_buffer
                end_flag = True
                self.f.close()
            else:
                for _ in range(self.batch_size):
                    self.result_buffer += self.f.readlines(self.max_len*self.batch_size)
                    self.buffer_len = len(self.result_buffer)
                    if self.buffer_len > self.batch_size:
                        result = self.result_buffer[:self.batch_size]
                        self.result_buffer = np.delete(self.result_buffer,np.arange(self.batch_size)).tolist()
                        break
                if self.buffer_len < self.batch_size:
                    result = self.result_buffer
                    self.buffer_len = 0
                    self.result_buffer = [] 
        return [np.array(l.strip("\n").split(",")).astype("int").tolist() for l in result],end_flag
    def close(self):
        self.f.close()
        
def get_dicts(char2id_loc,id2char_loc):
    import json
    with open(char2id_loc,"r") as f:
        char2id = json.load(f)
    with open(id2char_loc,"r") as f:
        id2char = json.load(f)
    return char2id,id2char
def tokenizer_producer(dicts,already=True):
    if already:
        def tokenizer(text):
            result = [int(l) for l in text.strip("\n").split(",")]
            return result
        return tokenizer
    def tokenizer(text):
        result = [dicts.get(t,0) for t in text]
        return result
    return tokenizer
def sample_positions(tokenized,max_len,mask_index):
    padded,position_indeces,target_word_indeces,mask_values = [],[],[],[]
    sample_size = np.random.choice(np.arange(min([len(s) for s in tokenized])))
    for i,d in enumerate(tokenized):
        d = d[:max_len]
        row_indeces,row_targets = [],[]
        randomeds = np.random.choice(np.arange(len(d)),size=sample_size,replace=False)
        for j,randomed in enumerate(randomeds):
            row_indeces.append([i,randomed])
            row_targets.append([i,j,d[randomed]])
        position_indeces.append(row_indeces)
        target_word_indeces.append(row_targets)
        pad_len = max_len-len(d)
        padded.append((d+[0]*pad_len)[:max_len])
        mask_values.append([mask_index]*sample_size)
    return padded,position_indeces,target_word_indeces,mask_values
def data_batcher_producer(train_loc,max_len,batch_size,mask_index):
    def producer():
        tx = Reader(train_loc,max_len,batch_size)
        def data_batcher(close=False):

            data_list,end_flag = tx.read()
            padded,position_indeces,target_word_indeces,mask_values = sample_positions(
                data_list,max_len,mask_index)
            if close:
                tx.close()
            return end_flag,padded,position_indeces,target_word_indeces,mask_values
        return data_batcher
    return producer        