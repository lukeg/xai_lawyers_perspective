from lime.lime_text import LimeTextExplainer
from dataset import Veterans
from dataset import Distillbert_base_uncased
from model import Model
import transformers

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np


CKPT = 'ckpt/0920_161959/model.ckpt-178'

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


tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def grad_cam_vals (sample):
    pred_val = classifier.predict(input_fn=lambda: dataset.predict_input_fn([sample], padded_size=1300), checkpoint_path=CKPT)[0]
    tokenized = '[CLS]' + tokenizer(sample) + '[SEP]'
    pred_idx = pred_val['predict_index'][0]
    tokens_len = len(tokenizer(sample))
    vec = pred_val['grad_cam'][pred_idx][:tokens_len]
    return vec

