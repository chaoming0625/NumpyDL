# -*- coding: utf-8 -*-

import npdl


class Seq2Seq:
    def __init__(self, hidden_size):
        self.encoder_lstm1 = npdl.layers.BatchLSTM(n_out=hidden_size)


