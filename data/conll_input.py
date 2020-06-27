
from pathlib import Path
import tensorflow as tf
import functools
from collections import Counter
import numpy as np

MINCOUNT = 1

def parse_fn(lines, encode=True, with_char=False):
    # Encode in Bytes for TF
    words, tags, chars = [], [], []
    for line in lines:
        segs = line.strip().split()
        words.append(segs[0].encode() if encode else segs[0])
        chars.append([c.encode() for c in segs[0]])
        tags.append(segs[-1].encode() if encode else segs[-1])
    assert len(words) == len(tags), "Words and tags lengths don't match"
    if not with_char:
        return (words, len(words)), tags
    else:
        # Chars
        lengths = [len(c) for c in chars]
        max_len = max(lengths)
        chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
        return ((words, len(words)), (chars, lengths)), tags

def generator_fn(fname, encode=True, with_char=False):
    with Path(fname).expanduser().open('r') as fid:
        lines = []
        for _line in fid:
            _line = _line.strip()
            if not _line:
                yield parse_fn(lines, encode=encode, with_char=with_char)
                del lines[:]
            else:
                lines.append(_line)

def input_fn(file, params=None, shuffle_and_repeat=False, with_char=False):
    params = params if params is not None else {}
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
        functools.partial(generator_fn, file, with_char=with_char),
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
