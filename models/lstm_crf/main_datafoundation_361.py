"""GloVe Embeddings + bi-LSTM + CRF"""

__author__ = "Guillaume Genthial"

'''
    将title和text拼接后，直接使用lstm_crf建模
'''

import json
import logging
import sys
import os
import shutil

from tf_metrics import precision, recall, f1
from models.common.embedding_layer import embedding_layer

from data.ner_datafountain_361 import *
DATADIR = '~/.datasets/ner/datafoundation_361/'
RESULT_PATH = './results_v1/'

# Logging
Path(RESULT_PATH).mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('{}/main.log'.format(RESULT_PATH)),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def model_fn(features, labels, mode, params):
    # For serving, features are a bit different
    if isinstance(features, dict):
        features = features['words'], features['nwords']

    # Read vocabs and inputs
    dropout = params['dropout']
    words, nwords = features
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
    word_ids = vocab_words.lookup(words)
    if params.get('rand_embedding', False):
        print('\n\nrand embedding\n')
        embeddings = embedding_layer(word_ids, words_vocab_size, params['dim'], zero_pad=False)
    else:
        glove = np.load(str(Path(params['glove']).expanduser()))['embeddings']  # np.array
        variable = np.vstack([glove, [[0.]*params['dim']]])
        variable = tf.Variable(variable, dtype=tf.float32, trainable=False)
        embeddings = tf.nn.embedding_lookup(variable, word_ids)
    embeddings = tf.layers.dropout(embeddings, rate=dropout, training=training)

    # LSTM
    t = tf.transpose(embeddings, perm=[1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)
    output_fw, _ = lstm_cell_fw(t, dtype=tf.float32, sequence_length=nwords)
    output_bw, _ = lstm_cell_bw(t, dtype=tf.float32, sequence_length=nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)

    # CRF
    logits = tf.layers.dense(output, num_tags) # 仅仅是打分，没有使用activation ?
    crf_params = tf.get_variable("crf", [num_tags, num_tags], dtype=tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

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
        tags = vocab_tags.lookup(labels)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, tags, nwords, crf_params)
        loss = tf.reduce_mean(-log_likelihood)

        # Metrics
        weights = tf.sequence_mask(nwords)
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
        'glove': '{}/glove.npz'.format(RESULT_PATH),
        'pretrain_glove': '~/.datasets/embeddings/glove.840B.300d/glove.840B.300d.txt',
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

    train_inpf = functools.partial(input_fn, fname('train'), params, shuffle_and_repeat=True, mode=None)
    eval_inpf = functools.partial(input_fn, fname('valid'), mode=None)

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
            test_inpf = functools.partial(input_fn, fname(name), mode=None)
            golds_gen = generator_fn(fname(name))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((words, _), tags) = golds
                for word, tag, tag_pred in zip(words, tags, preds['tags']):
                    f.write(b' '.join([word, tag, tag_pred]) + b'\n')
                f.write(b'\n')

    for name in ['train', 'dev', 'test']:
        write_predictions(name)
