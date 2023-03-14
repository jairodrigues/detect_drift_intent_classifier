import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import bert


class Pre_process():

    def __init__(self):
        self.embedding = self.bert_layer()
        self.tokenizer = self.tokenizer_sentence()

    def tokenizer_sentence(self):
        FullTokenizer = bert.bert_tokenization.FullTokenizer
        bert_layer = self.embedding
        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
        return FullTokenizer(vocab_file, do_lower_case)

    def bert_layer():
        return hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1', trainable=False)

    def encode_sentence(self, sent):
        return ["[CLS]"] + self.tokenizer.tokenize(sent) + ["[SEP]"]

    def get_ids(self, tokens):
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def get_mask(tokens):
        return np.char.not_equal(tokens, "[PAD]").astype(int)

    def get_segments(tokens):
        seg_ids = []
        current_seg_id = 0
        for tok in tokens:
            seg_ids.append(current_seg_id)
            if tok == "[SEP]":
                current_seg_id = 1 - current_seg_id
        return seg_ids

    def transfor_label(intent):
        if intent == "movie":
            return 0
        elif intent == "music":
            return 1
        elif intent == "politic":
            return 2
        elif intent == "sport":
            return 3
        elif intent == "technology":
            return 4

    def createBertLayer(self, sent):
        text = self.encode_sentence(text)
        sent, tokens = self.bert_layer([
            tf.expand_dims(tf.cast(self.get_ids(text), tf.int32), 0),
            tf.expand_dims(tf.cast(self.get_mask(text), tf.int32), 0),
            tf.expand_dims(tf.cast(self.get_segments(text), tf.int32), 0)
        ])
        return sent
