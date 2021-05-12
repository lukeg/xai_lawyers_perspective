
from dataset import Veterans
from dataset import Distillbert_base_uncased


from multiprocessing import Process

class PoisonPill:
    pass

def predictor_process (pipe):
    print ("started predictor process")
    import tensorflow as tf

    from model import Model
    import numpy as np

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True


    #0701_171259\model.ckpt-247 -> BerENc
    #0702_020833\model.ckpt-1231 -> UKBERT
    #ckpt\0702_102616\model.ckpt-247 #word2vec
    #ckpt ckpt\0702_122249\model.ckpt-247 word2vec 1300 article length
    #ckpt\0704_224210\model.ckpt-493 bert without mean pooling
    #ckpt\0809_104201\model.ckpt-493 distillbert

    #ckpt\0831_133107\model.ckpt-493 ditillbert 2nd try
    #ckpt\0910_161031\model.ckpt-493 -> law2vec

    #ckpt\0910_233004\model.ckpt-493 -> word2vec (courtlistener 200d)


    CKPT = r'D:\UserData\lukeg\Documents\informatyka - projekty\gradcams\word2wec-gradcams\ckpt\0430_175102\model.ckpt-493'

    dataset = Veterans(Distillbert_base_uncased)
    background = dataset.train_x[:2]

    model = Model()
    classifier = tf.estimator.Estimator(model_fn=model.build,
                                        config=tf.estimator.RunConfig(session_config=config),
                                        params={
                                            'feature_columns': [tf.feature_column.numeric_column(key='x')], \
                                            'kernels': [(3,512),(4,512),(5,512)], \
                                            'num_classes': len(dataset.classes), \
                                            'max_article_length': 1300
                                        })

    def predictor (samples):
        return np.array([ result["hypothesis"] for result in [
                          list(classifier.predict(input_fn=lambda: dataset.predict_input_fn([sample], padded_size=1300), checkpoint_path=CKPT))[0] for sample in samples] ])

    while True:
        recvd = pipe.recv()
        if isinstance(recvd, PoisonPill):
            print ("finishing")
            return

        print ("performing prediction")
        predictions = predictor(recvd)
        print ("sending prediction")
        pipe.send(predictions)
        print("prediction sent")


def create_parent_predictor(parent_pipe):
    def parent_predictor(samples):
        print("parent sending samples")
        parent_pipe.send(samples)
        print ("parent sent samples")
        return parent_pipe.recv()
    return parent_predictor

def main (sample):
    import multiprocessing
    import pandas as pd
    from dataset import Distillbert_base_uncased
    import shap

    parent_conn, child_conn = multiprocessing.Pipe()
    predictor = multiprocessing.Process(target=predictor_process, args=(child_conn,))
    predictor.start()

    import transformers
    tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased')
    masker = shap.maskers.Text(tokenizer, mask_token="...")
    explainer = shap.Explainer(create_parent_predictor(parent_conn), masker)
    data=[sample]
    explanation = explainer(data)

    parent_conn.send(PoisonPill())
    predictor.join()

    return explanation
if __name__ == '__main__':
    explanation = main("The Federal Circuit has held that 38 U.S.C.A.  105 and 1110 preclude compensation for primary alcohol abuse disabilities and secondary disabilities that result from primary alcohol abuse.");    print(explanation)
