import tensorflow as tf
import shutil
def save_models(sess,tf_model,access_dict):
    import json
    import os
    
#     try: 
#         os.mkdir("{}".format(tf_model))
#     except:
#         pass
    logdir = "{}".format(tf_model)
    #from tensorflow.compat.v1.graph_util import convert_variables_to_constants
    #graph = convert_variables_to_constants(sess,sess.graph_def,["input","prediction"])
#    tf.io.write_graph(graph,logdir,"tf_model.pb",as_text=False)
    try:
        tf.saved_model.simple_save(sess,
                logdir,
                inputs={"input": access_dict["input"],
                        "training":access_dict["training"],
                       "position_indeces":access_dict["position_indeces"],
                       "target_word_indeces":access_dict["target_word_indeces"],
                       "mask_values":access_dict["mask_values"],
                       "peep":access_dict["peep"]},
                outputs={"outputs": access_dict["outputs"]})
    except:
        shutil.rmtree(logdir)
        tf.saved_model.simple_save(sess,
                logdir,
                inputs={"input": access_dict["input"],
                        "training":access_dict["training"],
                       "position_indeces":access_dict["position_indeces"],
                       "target_word_indeces":access_dict["target_word_indeces"],
                       "mask_values":access_dict["mask_values"],
                       "peep":access_dict["peep"]},
                outputs={"outputs": access_dict["outputs"]})
    print("{}".format(tf_model))
def get_interpreter(model_path):
    parser = tf.contrib.predictor.from_saved_model(model_path)
    def interpret(inputs):
        return parser({"input":inputs,"training":False,
                      "position_indeces":[[[0,0]]],
                      "target_word_indeces":[[[0,0,0]]],
                      "mask_values":[[0]]})["outputs"]
    return interpret
def bot_thought(interpreter,tokenizer,max_len=400):
    def bot_say(inputs):
        if type(inputs).__name__=="list":
            sentences = []
            tokenized = [tokenizer(s) for s in inputs]
            for t in tokenized:
                pad_len = max_len-len(t)
                sentences.append((t+[0]*pad_len)[:max_len])

            ans = interpreter(sentences)

            #ans = interpreter({"inputs":inputs})["predictions"]
            #print(ans)
        else:
            tokenized = tokenizer(inputs)
            pad_len = max_len-len(tokenized)
            tokenized = (tokenized+[0]*pad_len)[:max_len]
            ans = interpreter([tokenized])
        return ans
    return bot_say