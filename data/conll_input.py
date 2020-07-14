
import os
import random
import sys
from pathlib import Path
import tensorflow as tf
import functools
from collections import Counter
import numpy as np
from .bert_formatter import convert_single_instance

MINCOUNT = 1

def parse_fn(lines, encode=True, with_char=False, bert_out=False, bert_proj_path=None, bert_config_json=None, max_seq_len=512):
    # Encode in Bytes for TF
    words, tags, chars = [], [], []
    for line in lines:
        segs = line.strip().split()
        words.append(segs[0].encode() if encode and not bert_out else segs[0])
        chars.append([c.encode() for c in segs[0]])
        tags.append(segs[-1].encode() if encode and not bert_out else segs[-1])
    assert len(words) == len(tags), "Words and tags lengths don't match"

    n_words = len(words)
    if bert_out:
        assert bert_proj_path, 'bert_proj_path must not be None'
        sys.path.append(os.path.expanduser(bert_proj_path))
        from models.bert.tokenization import FullTokenizer
        tokenizer = FullTokenizer(vocab_file=bert_config_json['vocab_file'], do_lower_case=bert_config_json['do_lower_case'])
        input_ids, input_mask, segment_ids, tags, n_words = convert_single_instance(words, tags, max_seq_len, tokenizer)
        words = input_ids, input_mask, segment_ids

        # if random.randint(0, 100) < 5:
        #     print("\nwords: %s" % " ".join([str(x) for x in words]))
        #     print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     print("label: %s" % " ".join([str(x) for x in tags]))
        #     print("n_words: %s\n" % n_words)

    tags = [_.encode() if encode else _ for _ in tags]
    if not with_char:
        return (words, n_words), tags
    else:
        # Chars
        lengths = [len(c) for c in chars]
        max_len = max(lengths)
        chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
        return ((words, n_words), (chars, lengths)), tags

def generator_fn(fname, encode=True, with_char=False, bert_out=False, bert_proj_path=None, bert_config_json=None, max_seq_len=512):
    with Path(fname).expanduser().open('r') as fid:
        lines = []
        for _line in fid:
            _line = _line.strip()
            if not _line:
                yield parse_fn(lines, encode=encode, with_char=with_char,
                               bert_out=bert_out, bert_proj_path=bert_proj_path, bert_config_json=bert_config_json, max_seq_len=max_seq_len)
                del lines[:]
            else:
                lines.append(_line)

def input_fn(file, params=None, shuffle_and_repeat=False, with_char=False, bert_out=False,
             bert_proj_path=None, bert_config_json=None, max_seq_len=512):
    params = params if params is not None else {}
    if bert_out:
        if not with_char:
            shapes = ((([None], [None], [None]), ()), [None])
            types = (((tf.int32, tf.int32, tf.int32), tf.int32), tf.string)
            defaults = (((0, 0, 0), 0), 'O')
        else:
            shapes = (
                (
                    (([None], [None], [None]), ()),
                    ([None, None], [None])
                ),  # (chars, nchars)
                [None]
            )  # tags
            types = (
                (
                    ((tf.int32, tf.int32, tf.int32), tf.int32),
                    (tf.string, tf.int32)
                ),
                tf.string
            )
            defaults = (
                (
                    ((0, 0, 0), 0),
                    ('<pad>', 0)
                ),
                'O'
            )
    else:
        if not with_char:
            shapes = (([None], ()), [None])
            types = ((tf.string, tf.int32), tf.string)
            defaults = (('<pad>', 0), 'O')
        else:
            shapes = (
                (
                    ([None], ()),  # (words, nwords)
                    ([None, None], [None])
                ),  # (chars, nchars)
                [None]
            )  # tags
            types = (
                (
                    (tf.string, tf.int32),
                    (tf.string, tf.int32)
                ),
                tf.string
            )
            defaults = (
                (
                    ('<pad>', 0),
                    ('<pad>', 0)
                ),
                'O'
            )

    dataset = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, file, with_char=with_char, bert_out=bert_out,
                          bert_proj_path=bert_proj_path, bert_config_json=bert_config_json, max_seq_len=max_seq_len),
        output_shapes=shapes, output_types=types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = (dataset
               .padded_batch(params.get('batch_size', 20), shapes, defaults)
               .prefetch(1))
    return dataset

def build_vocab(files, output_dir, min_count=MINCOUNT, force_build=False, encode=False):
    # 1. Words
    # Get Counter of words on all the data, filter by min count, save

    words_path = '{}/vocab.words.txt'.format(output_dir)
    chars_path = '{}/vocab.chars.txt'.format(output_dir)
    tags_path = '{}/vocab.tags.txt'.format(output_dir)

    if not force_build:
        if Path(words_path).expanduser().exists() \
            and Path(chars_path).expanduser().exists() \
            and Path(tags_path).expanduser().exists():
            print('vocab already build, pass. {} {} {}'.format(words_path, chars_path, tags_path))
            return words_path, chars_path, tags_path

    print('Build vocab words/tags (may take a while)')
    counter_words = Counter()
    vocab_tags = set()
    for file in files:
        for (words, words_len), tags in generator_fn(file, encode=encode):
            counter_words.update(words)
            vocab_tags.update(tags)

    vocab_words = {w for w, c in counter_words.items() if c >= min_count}


    with Path(words_path).expanduser().open('w') as f:
        for w in sorted(list(vocab_words)):
            f.write('{}\n'.format(w))
    print('- done. Kept {} out of {}'.format(
        len(vocab_words), len(counter_words)))

    # 2. Chars
    # Get all the characters from the vocab words
    print('Build vocab chars')
    vocab_chars = set()
    for w in vocab_words:
        print(w)
        vocab_chars.update(w)


    with Path(chars_path).expanduser().open('w') as f:
        for c in sorted(list(vocab_chars)):
            f.write('{}\n'.format(c))
    print('- done. Found {} chars'.format(len(vocab_chars)))


    with Path(tags_path).expanduser().open('w') as f:
        for t in sorted(list(vocab_tags)):
            f.write('{}\n'.format(t))
    print('- done. Found {} tags.'.format(len(vocab_tags)))

    return words_path, chars_path, tags_path


def build_glove(words_file='vocab.words.txt', output_path='glove.npz', glove_path='glove.840B.300d.txt', force_build=False):

    if not force_build:
        if Path(output_path).expanduser().exists():
            print('glove already build, pass. {}'.format(output_path))
            return output_path

    with Path(words_file).expanduser().open() as f:
        word_to_idx = {line.strip(): idx for idx, line in enumerate(f)}
    size_vocab = len(word_to_idx)

    # Array of zeros
    embeddings = np.zeros((size_vocab, 300))

    # Get relevant glove vectors
    found = 0
    print('Reading GloVe file (may take a while)')
    with Path(glove_path).expanduser().open() as f:
        for line_idx, line in enumerate(f):
            if line_idx % 100000 == 0:
                print('- At line {}'.format(line_idx))
            line = line.strip().split()
            if len(line) != 300 + 1:
                continue
            word = line[0]
            embedding = line[1:]
            if word in word_to_idx:
                found += 1
                word_idx = word_to_idx[word]
                embeddings[word_idx] = embedding
    print('- done. Found {} vectors for {} words'.format(found, size_vocab))

    # Save np.array to file
    np.savez_compressed(str(Path(output_path).expanduser()), embeddings=embeddings)

    return output_path
