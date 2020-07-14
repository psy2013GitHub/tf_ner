

from pathlib import Path
import tensorflow as tf
import functools
from collections import Counter
import numpy as np
import sys

MINCOUNT = 1

FIELD_SEP = ','
ENTITY_SEP = ';'

def gen_tag_by_char(text, targets):
    targets = targets.split(ENTITY_SEP)
    tags = ['O', ] * len(text)
    if len(targets) == 1 and not targets[0]:
        return tags

    start_pos = 0
    for _target in targets:
        assert len(_target) > 1, text
        target_tag = ['B', ] + ['I', ] * (len(_target) - 2) + ['E', ]
        while True:
            idx = text[start_pos:].find(_target)
            if idx < 0:
                break
            tags[start_pos + idx:start_pos + idx + len(target_tag)] = target_tag
            start_pos = start_pos + idx + len(target_tag)
    return tags

def parse_fn(line, encode=True, with_char=False, sep_title_text=False):
    '''

    :param line:
    :param encode:
    :param with_char:
    :param sep_title_text: title和text是不是分开
    :return:
    '''
    # Encode in Bytes for TF
    prefix, targets = line.rsplit(FIELD_SEP, 1)
    uid, all_text = prefix.split(FIELD_SEP, 1)
    title, text = all_text.split(FIELD_SEP, 1)

    # all_text = title + FIELD_SEP + text
    if not sep_title_text:
        words = [c.encode() if encode else c for c in all_text]
        chars = [c.encode() if encode else c for c in all_text]
        tags = gen_tag_by_char(all_text, targets)
        assert len(words) == len(tags), "Words and tags lengths don't match"
        if not with_char:
            return (words, len(words)), tags
        else:
            # Chars
            lengths = [len(c) for c in chars]
            max_len = max(lengths)
            chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
            return ((words, len(words)), (chars, lengths)), tags
    else:
        title_words = [c.encode() if encode else c for c in title]
        title_chars = [c.encode() if encode else c for c in title]
        title_tags = gen_tag_by_char(title, targets)

        text_words = [c.encode() if encode else c for c in text]
        text_chars = [c.encode() if encode else c for c in text]
        text_tags = gen_tag_by_char(text, targets)

        if not with_char:
            return ((title_words, len(title_words)), (text_words, len(text_words))), (title_tags, text_tags)
        else:
            # Chars
            return (((title_words, len(title_words)), (title_chars, len(title_chars))), ((text_words, len(text_words)), (text_chars, len(text_chars)))), \
                    (title_tags, text_tags)

def generator_fn(fname, encode=True, with_char=False, train_idx_set=None, valid_idx_set=None, mode='train', sep_title_text=False):
    with Path(fname).expanduser().open('r') as fid:
        lines = []
        for i, _line in enumerate(fid):
            if i == 0:
                continue
            _line = _line.strip()
            if _line:
                if mode == 'train':
                    if i in train_idx_set:
                        yield parse_fn(_line, encode=encode, with_char=with_char, sep_title_text=sep_title_text)
                elif mode == 'valid':
                    if i in valid_idx_set:
                        yield parse_fn(_line, encode=encode, with_char=with_char, sep_title_text=sep_title_text)
                else:
                    yield parse_fn(_line, encode=encode, with_char=with_char, sep_title_text=sep_title_text)


def input_fn(file, params=None, shuffle_and_repeat=False, with_char=False, train_idx_set=None, valid_idx_set=None, mode='train', sep_title_text=False):
    params = params if params is not None else {}
    if not with_char:
        if sep_title_text:
            shapes = ((([None], ()), ([None], ())), ([None], [None]))
            types = (((tf.string, tf.int32), (tf.string, tf.int32)), (tf.string, tf.string))
            defaults = ((('<pad>', 0), ('<pad>', 0)), ('O', 'O'))
        else:
            shapes = (([None], ()), [None])
            types = ((tf.string, tf.int32), tf.string)
            defaults = (('<pad>', 0), 'O')
    else:
        if sep_title_text:
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
        functools.partial(generator_fn, file, with_char=with_char, train_idx_set=train_idx_set, valid_idx_set=valid_idx_set, mode=mode, sep_title_text=sep_title_text),
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
        for (words, words_len), tags in generator_fn(file, encode=encode, mode=None):
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


def split_train_file(train_path, seed=123, train_count=10, flush_to_file=False, train_out_path='', valid_out_path=''):
    invalid_line_num = 0
    train_idxs, valid_idxs = [], []
    with Path(train_path).expanduser().open() as f:
        for line_idx, line in enumerate(f):
            if line_idx == 0:
                continue
            if len(line.strip().split(FIELD_SEP)) != 4:
                invalid_line_num += 1

            if len(train_idxs) <= train_count:
                train_idxs.append(line_idx)
            else:
                idx = np.random.randint(1, line_idx)
                if idx <= train_count:
                    train_idxs[idx-1] = line_idx
                else:
                    valid_idxs.append(line_idx)

    if flush_to_file:
        with Path(train_path).expanduser().open() as f:
            with Path(train_out_path).expanduser().open('w') as f1:
                with Path(valid_out_path).expanduser().open('w') as f2:
                    for line_idx, line in enumerate(f):
                        if line_idx in train_idxs:
                            f1.write('{}'.format(line))
                        elif line_idx in valid_idxs:
                            f2.write('{}'.format(line))


    print('invalid row number: {}'.format(invalid_line_num))
    return train_idxs, valid_idxs
