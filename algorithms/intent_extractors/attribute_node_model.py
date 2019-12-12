from ..util_funcs.attribute_model_builder import build_graph,train_tf_model
from ..util_funcs.data_preprocessing import get_dicts,get_train_data
from ..util_funcs.model_handler import get_interpreter,bot_thought

import tensorflow as tf
import numpy as np

def nlu_model_producer(config_path="nlu_config.yml"):
    import yaml
    with open(config_path,"r") as f:
        config = yaml.safe_load(f)
    def train_model(model_id,train_data):
        w2id = get_dicts(train_data)[0]
        graph,access_dict = build_graph(**config["model_hp"],vocab_size=len(w2id),token_label_size = 1,
                                        intent_size = 2)
        processed_train_data = get_train_data(train_data,w2id)
        model_path = config["model_path"]+model_id+"/"
        train_tf_model(graph,access_dict,processed_train_data,w2id,model_path,**config["train_params"])
        print("model {} is trained!".format(model_id))
    return train_model

def parser_producer(node_id,config_path="nlu_config.yml"):
    import yaml
    with open(config_path,"r") as f:
        config = yaml.safe_load(f)
    model_path = config["model_path"]+node_id+"/"
    interpreter,w2id = get_interpreter(model_path)
    return bot_thought(interpreter,w2id,config["model_hp"]["max_len"])