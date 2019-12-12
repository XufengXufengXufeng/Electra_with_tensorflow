from util_funcs.process_and_save_data_for_training import process_data_for_training

import yaml
with open("configure.yml") as f:
    configure = yaml.safe_load(f)
    
if __name__=="__main__":
    process_data_for_training(configure["datafiles"],configure["char2id_loc"]
                              ,configure["id2char_loc"],configure["train_data_loc"])