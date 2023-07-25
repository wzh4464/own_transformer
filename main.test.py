import torch
import unittest
from main import EncoderLayer, DecodeerLayer

class TestEncoderLayer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_length = 3
        self.d_model = 4
        self.num_heads = 2
        self.d_ff = 16
        self.dropout = 0.1
        self.device = 'cpu'
        self.encoder_layer = EncoderLayer(self.d_model, self.num_heads, self.d_ff, self.dropout, self.device)
        self.x = torch.randn(self.batch_size, self.seq_length, self.d_model)
        self.mask = torch.ones(self.batch_size, self.seq_length, self.seq_length)
        
    def test_forward(self):
        output = self.encoder_layer(self.x, self.mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.d_model))

class TestDecoderLayer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_length = 3
        self.d_model = 4
        self.num_heads = 2
        self.d_ff = 16
        self.dropout = 0.1
        self.device = 'cpu'
        self.decoder_layer = DecodeerLayer(self.d_model, self.num_heads, self.d_ff, self.dropout, self.device)
        self.x = torch.randn(self.batch_size, self.seq_length, self.d_model)
        self.enc_output = torch.randn(self.batch_size, self.seq_length, self.d_model)
        self.src_mask = torch.ones(self.batch_size, self.seq_length, self.seq_length)
        self.tgt_mask = torch.ones(self.batch_size, self.seq_length, self.seq_length)
        
    def test_forward(self):
        output = self.decoder_layer(self.x, self.enc_output, self.src_mask, self.tgt_mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.d_model))

if __name__ == '__main__':
    unittest.main()