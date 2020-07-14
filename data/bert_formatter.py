
import random


BERT_SPECIAL_SYMBOL_MAP = {
    'CLS': 'O', # O OR CLS 没有任何影响，不过我觉得O 会减少标签个数,不过拒收和句尾使用不同的标志来标注，使用LCS 也没毛病
    'SEP': 'O',
}

LABELS_NEXT = {
    'B'
}

PAD_TAG = 'O'



def labels_next(label, first=False, last=False):
    '''
        因为bert子词tokenize的原因，所以需要进行必要的转换
    :param label:
    :param first:
    :return:
    '''
    if label.startswith('B'):
        if first:
            return label
        return 'I' + label[1:]
    elif label.startswith('E'):
        if last:
            return label
        return 'I' + label[1:]
    return label




def convert_single_instance(words, labels, max_seq_length, tokenizer, pad_tag=PAD_TAG,
        bert_special_symbol_map=BERT_SPECIAL_SYMBOL_MAP, labels_next_callable=labels_next):
    """
    将一个样本进行分析，然后将字转化为id, 标签转化为id,然后结构化到InputFeatures对象中
    :param ex_index: index
    :param example: 一个样本
    :param label_list: 标签列表
    :param max_seq_length:
    :param tokenizer:
    :param output_dir
    :param mode:
    :return:
    """
    tokens = ["[CLS]", ]
    new_labels = [bert_special_symbol_map["CLS"], ]
    for i, word in enumerate(words):
        # 分词，如果是中文，就是分字,但是对于一些不在BERT的vocab.txt中得字符会被进行WordPice处理（例如中文的引号），可以将所有的分字操作替换为list(input)
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label_1 = labels[i]
        if len(token) == 1:
            new_labels.append(label_1)
        else:
            for m in range(len(token)):
                # 一般不会出现else
                new_labels.append(labels_next_callable(label_1, first = m==0, last = m==len(token)-1)) # fixme 这个地方这样换是不是合适？？？
    # tokens = tokenizer.tokenize(example.text)
    # 序列截断
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]  # -2 的原因是因为序列需要加一个句首和句尾标志
        new_labels = new_labels[0:(max_seq_length - 2)]
    labels = new_labels

    tokens.append("[SEP]")

    segment_ids = [0, ] * len(tokens)

    labels.append(bert_special_symbol_map["SEP"])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1, ] * len(input_ids)
    # label_mask = [1] * len(input_ids)
    # padding, 使用

    n_tokens = len(input_ids)
    if n_tokens < max_seq_length:
        gap = max_seq_length - n_tokens
        input_ids.extend([0,] * gap)
        input_mask.extend([0, ] * gap)
        segment_ids.extend([0, ] * gap)
        # we don't concerned about it!
        labels.extend([pad_tag, ] * gap)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(labels) == max_seq_length

    return input_ids, input_mask, segment_ids, labels, n_tokens


if __name__ == '__main__':
    from models.bert.tokenization import FullTokenizer
    tokenizer = FullTokenizer('/data/models/bert/chinese-bert_chinese_wwm_L-12_H-768_A-12/publish/vocab.txt', do_lower_case=False)
    print(convert_single_instance('who is your mothering ?'.split(), 'OOBEO', 10, tokenizer, pad_tag=PAD_TAG,
                            bert_special_symbol_map=BERT_SPECIAL_SYMBOL_MAP, labels_next_callable=labels_next))