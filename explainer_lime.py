from lime.lime_text import LimeTextExplainer
from dataset import Veterans
from dataset import Distillbert_base_uncased, BertPreTrained
from model import Model
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np

if __name__ == '__main__':

    CKPT = 'ckpt/0203_172226/model.ckpt-493' #distilbert_base_uncased, ptsd

    dataset = Veterans(Distillbert_base_uncased)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

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
        return np.array([ result["hypothesis"] for result in
                          classifier.predict(input_fn=lambda: dataset.predict_input_fn(samples, padded_size=1300), checkpoint_path=CKPT)])

    explainer = LimeTextExplainer(kernel_width=20, bow=False)
    explanation = explainer.explain_instance("test text", predictor, num_features=150, num_samples=400)
    explain = lambda text: explainer.explain_instance(text, predictor, num_features=150, num_samples=400)
    ordered=lambda explanation: sorted(map(lambda element: (element[0][0], element[1][0], element[0][1]), zip(explanation.as_list(), explanation.local_exp[1])), key=lambda key: key[1])
    print (ordered(explanation))
