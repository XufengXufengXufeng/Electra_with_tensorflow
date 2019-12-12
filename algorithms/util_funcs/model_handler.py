import tensorflow as tf
def save_models(sess,tf_model,w2id):
    import json
    import os
    from tensorflow.compat.v1.graph_util import convert_variables_to_constants
    graph = convert_variables_to_constants(sess,sess.graph_def,["input","prediction"])
    try: 
        os.mkdir("{}".format(tf_model))
    except:
        pass
    logdir = "{}".format(tf_model)
    tf.io.write_graph(graph,logdir,"tf_model.pb",as_text=False)
    with open("{}/word2id.json".format(tf_model),'w') as f:
        json.dump(w2id,f,ensure_ascii=False)
    print("{}".format(tf_model))

def get_interpreter(model_path):
    import json
    with open(model_path+"/word2id.json","r") as f:
        w2id = json.load(f)
    def interpret(inputs):
        with tf.Graph().as_default():
            g_def = tf.GraphDef()
            with open(model_path+"/tf_model.pb","rb") as f:
                g_def.ParseFromString(f.read())
                tf.import_graph_def(g_def,name="")
            with tf.Session() as sess:
                tf.global_variables_initializer().run()
                input_node = sess.graph.get_tensor_by_name("input:0")
                prediction_node = sess.graph.get_tensor_by_name("prediction:0")
                prediction = prediction_node.eval(feed_dict={input_node:inputs})
                return prediction
    return interpret,w2id
def bot_thought(interpreter,w2id,max_len=30):
    key_type = (str,int)[type(list(w2id.keys())[0]).__name__=="int"]
    def bot_say(inputs):
        sentences = []
        for w in inputs:
            if w in w2id.keys():
                sentences.append(w2id[key_type(w)])
            else:
                sentences.append(0)
        #seq_len = [len(sentences)]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([sentences],maxlen=max_len,padding="post")
        
        ans = interpreter(inputs)
        print(inputs)
        #ans = interpreter({"inputs":inputs})["predictions"]
        print(ans)
    return bot_say