import argparse
import yaml

from util_funcs.prepare_for_trianing import get_dicts,data_batcher_producer
from modeling.build_electra_graph import build_graph
from util_funcs.trainer import train_tf_model
parser = argparse.ArgumentParser(description='pass in some arguments')
parser.add_argument("-e","--epochs",type=int)
parser.add_argument("-b","--batch_size",type=int)
parser.add_argument("-n","--tf_model_name",type=str,help="model name after training, includes folder")

args=parser.parse_args()

with open("configure.yml") as f:
    configure = yaml.safe_load(f)
    
char2id_loc = "data/processed_data/char2id.json"
char2id,_=get_dicts(configure["char2id_loc"],configure["id2char_loc"])
vocab_size = len(char2id)
embedding_size = configure["embedding_size"]
generator_size = configure["generator_size"]
gn_blocks = configure["gn_blocks"]
gseq_length = dseq_length = configure["seq_length"]
gn_heads = configure["gn_heads"]
gff_filter_size = configure["gff_filter_size"]
g_dev = configure["g_dev"]
dn_blocks = configure["dn_blocks"]
dn_heads = configure["dn_heads"]
dff_filter_size = configure["dff_filter_size"]
d_dev = configure["d_dev"]
d_factor = configure["d_factor"]
learning_rate = float(configure["learning_rate"])
tf_model_name = args.tf_model_name
epochs = args.epochs
train_loc = configure["train_data_loc"]
max_len = configure["max_len"]
batch_size = args.batch_size
mask_index = char2id["mask"]

if __name__=="__main__":
    data_batcher = data_batcher_producer(train_loc,max_len,batch_size,mask_index)
    graph,access_dict = build_graph(vocab_size,embedding_size,generator_size,
                        gn_blocks,gseq_length,gn_heads,gff_filter_size,g_dev,
                        dn_blocks,dseq_length,dn_heads,dff_filter_size,d_dev,mask_index,
                    d_factor,learning_rate)
    train_tf_model(graph,access_dict,data_batcher,tf_model_name,epochs)