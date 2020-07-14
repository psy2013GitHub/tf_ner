"""GloVe Embeddings + bi-LSTM + CRF"""

__author__ = "Guillaume Genthial"

import json
import logging
import os
import shutil

from tf_metrics import precision, recall, f1

from data.conll_input import *

# Logging
Path('results').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [
    logging.FileHandler('results/main.log'),
    logging.StreamHandler(sys.stdout)
]
logging.getLogger('tensorflow').handlers = handlers


def model_fn(features, labels, mode, params):
    # For serving, features are a bit different
    if isinstance(features, dict):
        features = features['words'], features['nwords']

    # Read vocabs and inputs
    dropout = params['dropout']
    (input_ids, input_mask, segment_ids), nwords = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)

    # 将词转化为int
    with Path(params['tags']).open() as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1

    # Bert Embeddings
    sys.path.append(params['bert_project_path'])
    from models.bert import modeling
    bert_config = modeling.BertConfig.from_json_file(params['bert_config_file'])
    bert_model = modeling.BertModel(
        config=bert_config,
        is_training=training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False
    )
    tvars = tf.trainable_variables()
    # 加载BERT模型
    if params.get('bert_init_checkpoint', False):
        init_checkpoint = params['bert_init_checkpoint']
        (assignment_map, initialized_variable_names) = \
            modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # 获取对应的embedding 输入数据[batch_size, seq_length, embedding_size]
    embeddings = bert_model.get_sequence_output()
    max_seq_length = embeddings.shape[1].value

    # dropout
    embeddings = tf.layers.dropout(embeddings, rate=0.1, training=training)

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
        weights = input_mask
        # weights = tf.sequence_mask(nwords, maxlen=max_seq_length)
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
            train_op = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(
                loss, global_step=tf.train.get_or_create_global_step())
            return tf.estimator.EstimatorSpec(
                mode, loss=loss, train_op=train_op)


if __name__ == '__main__':
    # Params
    params = {
        'max_seq_len': 50,
        'learning_rate': 1e-5,
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1, # ？
        'epochs': 25,
        'batch_size': 8,
        'buffer': 1500, # ？
        'lstm_size': 768,
        'force_build_vocab': False,
        'vocab_dir': './',
        'rand_embedding': True, # 随机初始化embedding
        'force_build_glove': False,
        'glove': './glove.npz',
        'pretrain_glove': '~/.datasets/embeddings/glove.840B.300d/glove.840B.300d.txt',
        'files': [
            '~/.datasets/ner/CoNLL-2003/train.txt',
            '~/.datasets/ner/CoNLL-2003/dev.txt',
            '~/.datasets/ner/CoNLL-2003/test.txt'
        ],
        'bert_project_path': '~/Documents/bert/',
        'bert_init_checkpoint': '/data/models/bert/cased_L-12_H-768_A-12/bert_model.ckpt',
        'bert_config_file': '/data/models/bert/cased_L-12_H-768_A-12/bert_config.json',
        'bert_config': {
            'vocab_file': '/data/models/bert/cased_L-12_H-768_A-12/vocab.txt',
            'do_lower_case': False
        },
        'RESULT_PATH': './results_v2/',
        'DATADIR': '~/.datasets/ner/CoNLL-2003/'
    }

    if not os.path.exists(params['RESULT_PATH']):
        os.mkdir(params['RESULT_PATH'])

    with Path('{}/params.json'.format(params['RESULT_PATH'])).open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    def fname(name):
        return str(Path('{}/{}.txt'.format(params['DATADIR'], name)).expanduser())

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
    train_inpf = functools.partial(input_fn, fname('train'), params, shuffle_and_repeat=True,
                bert_out=True, bert_proj_path=params.get('bert_project_path', None),
                bert_config_json=params.get('bert_config', None), max_seq_len=params.get('max_seq_len', None))
    eval_inpf = functools.partial(input_fn, fname('dev'),
                bert_out=True, bert_proj_path=params.get('bert_project_path', None),
                bert_config_json=params.get('bert_config', None), max_seq_len=params.get('max_seq_len', None))

    session_config = tf.ConfigProto(
        log_device_placement=False,
        inter_op_parallelism_threads=0,
        intra_op_parallelism_threads=0,
        allow_soft_placement=True)

    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120, session_config=session_config)
    model_path = '{}/model'.format(params['RESULT_PATH'])
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
        Path('{}/score'.format(params['RESULT_PATH'])).mkdir(parents=True, exist_ok=True)
        with Path('{}/score/{}.preds.txt'.format(params['RESULT_PATH'], name)).open('wb') as f:
            test_inpf = functools.partial(input_fn, fname(name),
                bert_out=True, bert_proj_path=params.get('bert_project_path', None),
                bert_config_json=params.get('bert_config', None), max_seq_len=params.get('max_seq_len', None))
            golds_gen = generator_fn(fname(name),
                bert_out=True, bert_proj_path=params.get('bert_project_path', None),
                bert_config_json=params.get('bert_config', None), max_seq_len=params.get('max_seq_len', None))
            preds_gen = estimator.predict(test_inpf)
            for golds, preds in zip(golds_gen, preds_gen):
                ((input_ids, input_mask, segment_ids), nwords), tags = golds
                word_count = 1
                for word, tag, tag_pred in zip(input_ids, tags, preds['tags']):
                    word_count += 1
                    if word_count > nwords:
                        break
                    if word_count == 1 or word_count == nwords: # 去掉CLS和SEP
                        continue

                    f.write(b' '.join([str(word).encode(), tag, tag_pred]) + b'\n')
                f.write(b'\n')

    for name in ['train', 'dev', 'test']:
        write_predictions(name)
