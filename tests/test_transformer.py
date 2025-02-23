import unittest
import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from coom.model.transformer import Transformer

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_length = 16
        self.vocab_size = 1000
        
    def test_transformer_tiny(self):
        model = Transformer(
            config=TransformerConfig(
                num_layers=2,
                hidden_size=8,
                num_attention_heads=4
            )
        )
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        
        # Test forward pass
        outputs = model(input_ids)
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_length, model.config.hidden_size))
        
    # def test_transformer_base(self):
    #     model = Transformer(model_size='base')
    #     input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        
    #     # Test forward pass
    #     outputs = model(input_ids)
    #     self.assertEqual(outputs.shape, (self.batch_size, self.seq_length, model.config.hidden_size))
        
    # def test_attention_mask(self):
    #     model = Transformer(model_size='tiny')
    #     input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
    #     attention_mask = torch.ones_like(input_ids)
    #     attention_mask[:, self.seq_length//2:] = 0  # Mask out second half
        
    #     # Test with attention mask
    #     outputs_with_mask = model(input_ids, attention_mask=attention_mask)
    #     outputs_without_mask = model(input_ids)
        
    #     # Outputs should be different with and without mask
    #     self.assertTrue(torch.any(outputs_with_mask != outputs_without_mask))
        
    # def test_position_ids(self):
    #     model = Transformer(model_size='tiny')
    #     input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
    #     position_ids = torch.arange(self.seq_length).unsqueeze(0).expand(self.batch_size, -1)
        
    #     # Test with custom position ids
    #     outputs_with_pos = model(input_ids, position_ids=position_ids)
    #     outputs_without_pos = model(input_ids)
        
    #     # Outputs should be the same with default and explicit position ids
    #     self.assertTrue(torch.all(outputs_with_pos == outputs_without_pos))
        
    # def test_custom_config(self):
    #     config = TransformerConfig(
    #         hidden_size=128,
    #         num_hidden_layers=2,
    #         num_attention_heads=4,
    #         intermediate_size=512
    #     )
    #     model = Transformer(config=config)
    #     input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
        
    #     # Test forward pass with custom config
    #     outputs = model(input_ids)
    #     self.assertEqual(outputs.shape, (self.batch_size, self.seq_length, config.hidden_size))
        
    # def test_weight_initialization(self):
    #     model = Transformer(model_size='tiny')
        
    #     # Test embedding initialization
    #     self.assertAlmostEqual(model.word_embeddings.weight.std().item(), 0.02, places=3)
        
    #     # Test layer norm initialization
    #     for layer in model.layers:
    #         self.assertTrue(torch.all(layer.layer_norm.weight == 1.0))
    #         self.assertTrue(torch.all(layer.layer_norm.bias == 0.0))

if __name__ == '__main__':
    unittest.main() 