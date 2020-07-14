"""GloVe Embeddings + bi-LSTM + CRF"""

__author__ = "Guillaume Genthial"

'''
    对title和text分开序列建模，同时融合进self-attention信息，最后再用crf解码
'''

import json
import logging
import os
import shutil

from tf_metrics import precision, recall, f1
from models.common.embedding_layer import embedding_layer
from models.common.attention import self_attention_layer

from data.ner_datafountain_361 import *
DATADIR = '~/.datasets/ner/datafoundation_361/'
RESULT_PATH = './results_v3/'

# Logging
Path(RESULT_PATH).mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('{}/main.log'.format(RESULT_PATH)),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def lstm_block(x, nwords, dropout, training, lstm_cell_fw=None, lstm_cell_bw=None):
    '''
    :param x:
    :param nwords:
    :param dropout:
    :param training:
    :param lstm_cell_fw:
    :param lstm_cell_bw:
    :return:
    '''
    t = tf.transpose(x, perm=[1, 0, 2])
    if lstm_cell_fw is None:
        lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    if lstm_cell_bw is None:
        lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
        lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)
    return output, (lstm_cell_fw, lstm_cell_bw)

def model_fn(features, labels, mode, params):
    # For serving, features are a bit different
    if isinstance(features, dict):
        # features = features['words'], features['nwords']
        raise ValueError(features)

    # Read vocabs and inputs
    dropout = params['dropout']
    (title_words, n_title_words), (text_words, n_text_words) = features
    title_labels, text_labels = labels

    training = (mode == tf.estimator.ModeKeys.TRAIN)

    # 将词转化为int
    vocab_words = tf.contrib.lookup.index_table_from_file(
        params['words'], num_oov_buckets=params['num_oov_buckets'])
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1

    with Path(params['words']).open() as f:
        words_vocab_size = len(f.readlines()) + params['num_oov_buckets']

    # Word Embeddings
    title_word_ids = vocab_words.lookup(title_words)
    text_word_ids = vocab_words.lookup(text_words)
    if params.get('rand_embedding', False):
        title_embeddings = embedding_layer(title_word_ids, words_vocab_size, params['dim'], zero_pad=False, reuse=tf.AUTO_REUSE)
        text_embeddings = embedding_layer(text_word_ids, words_vocab_size, params['dim'], zero_pad=False, reuse=tf.AUTO_REUSE)
    else:
        raise ValueError(params['rand_embedding'])

    title_embeddings = tf.layers.dropout(title_embeddings, rate=dropout, training=training)
    text_embeddings = tf.layers.dropout(text_embeddings, rate=dropout, training=training)

    # LSTM
    title_x, (title_fw, title_bw) = lstm_block(title_embeddings, n_title_words, dropout, training) # title和text共享lstm参数是不是会更好一点？
    if params.get('share_encoder', False):
        text_x, (_, _) = lstm_block(text_embeddings, n_text_words, dropout, training, lstm_cell_fw=title_fw, lstm_cell_bw=title_bw)
    else:
        text_x, (_, _) = lstm_block(text_embeddings, n_text_words, dropout, training)

    # self attention block
    # todo: 1, 是不是self attention加上position encoding会更好？，融合进位置信息对ner还是很有用
    title_x_cond_text = self_attention_layer(title_x, text_x, text_x, num_attention_heads=params['num_attention_heads'],
                                size_per_head=params['lstm_size']*2//params['num_attention_heads'], reuse=tf.AUTO_REUSE, training=training) # todo 这里是使用embedding来self attention还是使用lstm输出？
    text_x_cond_title = self_attention_layer(text_x, title_x, title_x, num_attention_heads=params['num_attention_heads'],
                                size_per_head=params['lstm_size']*2//params['num_attention_heads'], reuse=tf.AUTO_REUSE, training=training)  # todo 这里是使用embedding来self attention还是使用lstm输出？

    # concat
    title_x = tf.concat([title_x, title_x_cond_text], axis=-1)
    text_x = tf.concat([text_x, text_x_cond_title], axis=-1)

    # CRF
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)  # todo 共享crf参数是不是更合理？

    title_logits = tf.layers.dense(title_x, num_tags) # 仅仅是打分，没有使用activation ?
    title_pred_ids, _ = tf.contrib.crf.crf_decode(title_logits, crf_params, n_title_words)

    text_logits = tf.layers.dense(text_x, num_tags)  # 仅仅是打分，没有使用activation ?
    text_pred_ids, _ = tf.contrib.crf.crf_decode(text_logits, crf_params, n_text_words)

    pred_ids = tf.concat([title_pred_ids, text_pred_ids], axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Predictions
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(
            params['tags'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {
            'pred_ids': pred_ids,
            'tags': pred_strings
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    else:
        # Loss
        vocab_tags = tf.contrib.lookup.index_table_from_file(params['tags'])
        title_tags = vocab_tags.lookup(title_labels)
        text_tags = vocab_tags.lookup(text_labels)
        logits = tf.concat([title_logits, text_logits], axis=1)
        nwords = n_title_words + n_text_words

        title_log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            title_logits, title_tags, n_title_words, crf_params)

        text_log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            text_logits, text_tags, n_text_words, crf_params)

        loss = tf.reduce_mean(-title_log_likelihood - text_log_likelihood)

        tags = tf.concat([title_tags, text_tags], axis=-1)

        # Metrics
        weights = tf.concat([tf.sequence_mask(n_title_words), tf.sequence_mask(n_text_words)], axis=-1)
        metrics = {
            'acc': tf.metrics.accuracy(tags, pred_ids, weights),
            'precision': precision(tags, pred_ids, num_tags, indices, weights),
            'recall': recall(tags, pred_ids, num_tags, indices, weights),
            'f1': f1(tags, pred_ids, num_tags, indices, weights),
        }
        for metric_name, op in metrics.items():
            tf.summary.scalar(metric_name, op[1])

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=metrics)

        elif mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.train.AdamOptimizer().minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
    # Params
    params = {
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1, # ？
        'epochs': 50,
        'batch_size': 20,
        'buffer': 15000, # ？
        'lstm_size': 100,
        'force_build_vocab': True,
        'vocab_dir': '{}/'.format(RESULT_PATH),
        'rand_embedding': True, # 随机初始化embedding
        'force_build_glove': False,
        'share_encoder': True,
        'glove': '{}/glove.npz'.format(RESULT_PATH),
        'pretrain_glove': '~/.datasets/embeddings/glove.840B.300d/glove.840B.300d.txt',
        'num_attention_heads': 4,
        'files': [
            '{}/train.csv'.format(DATADIR)
        ]
    }
    with Path('{}/params.json'.format(RESULT_PATH)).open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fname(name):
        return str(Path('{}/tmp.{}.csv'.format(RESULT_PATH, name)).expanduser())

    params['words'], params['chars'], params['tags'] = build_vocab(params['files'],
            params['vocab_dir'],
            force_build=params['force_build_vocab']
    )

    params['glove'] = build_glove(words_file=params['words'],
        output_path=params['glove'],
        glove_path=params['pretrain_glove'],
        force_build=params['force_build_glove']
    )

    # Estimator, train and evaluate
    train_idx, valid_idx = split_train_file('{}/train.csv'.format(DATADIR), seed=123, train_count=(4000//params['batch_size']+1)*params['batch_size'],
           flush_to_file=True, train_out_path=fname('train'), valid_out_path=fname('valid'))

    train_inpf = functools.partial(input_fn, fname('train'), params, shuffle_and_repeat=True, mode=None, sep_title_text=True)
    eval_inpf = functools.partial(input_fn, fname('valid'), mode=None, sep_title_text=True)

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    model_path = '{}/model'.format(RESULT_PATH)
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    estimator = tf.estimator.Estimator(model_fn, model_path, cfg, params)
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)
    hook = tf.estimator.experimental.stop_if_no_decrease_hook(
        estimator, 'f1', 500, min_steps=8000, run_every_secs=120)
    train_spec = tf.estimator.TrainSpec(input_fn=train_inpf, hooks=[hook])
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_inpf, throttle_secs=120)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    # Write predictions to file
    def write_predictions(name):
        Path('{}/score'.format(RESULT_PATH)).mkdir(parents=True, exist_ok=True)
        with Path('{}/score/{}.preds.txt'.format(RESULT_PATH, name)).open('wb') as f:
            test_inpf = functools.partial(input_fn, fname(name), mode=None, sep_title_text=True)
            golds_gen = generator_fn(fname(name))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), tags) = golds
                for word, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')

    for name in ['train', 'dev', 'test']:
        write_predictions(name)
