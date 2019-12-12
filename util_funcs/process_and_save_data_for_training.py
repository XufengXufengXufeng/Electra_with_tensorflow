def read_data(file):
    import json
    import re
    def filter_words(s):
        return re.sub("[^\w+?!？！：；:;.。，,@&]","",s)
    data = []
    with open(file,"r") as f:
        for l in f:
            tmp = json.loads(l)
            data += re.split("(\w+[?!？！：；:;.。]+)",filter_words(tmp["title"]))[1::2]
            data += re.split("(\w+[?!？！：；:;.。]+)",filter_words(tmp["desc"]))[1::2]
            data += re.split("(\w+[?!？！：；:;.。]+)",filter_words(tmp["answer"]))[1::2]
    return data
def process_data_for_training(datafiles,char2id_loc,id2char_loc,train_data_loc):
    import json
    import re    
    data = []
    for file in datafiles:
        data+=read_data(file)
    chars = set()
    for l in data:
        chars|=set(l)
    char2id = {c:i for i,c in enumerate(chars)}
    id2char = {i:c for i,c in enumerate(chars)}
    first = id2char[0]
    id2char[0] = " "
    char2id[" "] = 0
    char2id[first]=len(id2char)
    id2char[char2id[first]]=first
    char2id["mask"]=len(id2char)
    id2char[char2id["mask"]]="mask"
    with open(char2id_loc,"w") as f:
        json.dump(char2id,f,ensure_ascii=False)
    with open(id2char_loc,"w") as f:
        json.dump(id2char,f,ensure_ascii=False)
    idified = []
    for l in data:
        tmp = []
        for w in list(l):
            tmp.append(str(char2id.get(w)))
        idified.append(",".join(tmp))
    with open(train_data_loc,"w") as f:
        f.writelines("\n".join(idified))
    print("data are ready!")