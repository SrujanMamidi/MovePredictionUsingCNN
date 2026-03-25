import unittest
import chess
import numpy as np
from utils.encoding import encode_board
from models.resnet import build_chess_resnet

class TestChessSystem(unittest.TestCase):
    
    def test_encoding_shape(self):
        board = chess.Board()
        encoded = encode_board(board)
        self.assertEqual(encoded.shape, (8, 8, 17))
        
    def test_encoding_values(self):
        board = chess.Board() # Starting position
        encoded = encode_board(board)
        # Check white pawn at rank 1, file 0 (A2)
        # White pawn channel is 0
        self.assertEqual(encoded[1, 0, 0], 1.0)
        # Check turn channel
        self.assertEqual(encoded[0, 0, 16], 1.0)
        
    def test_model_inference_shape(self):
        model = build_chess_resnet(num_moves=100)
        dummy_input = np.zeros((1, 8, 8, 17))
        policy, value = model.predict(dummy_input, verbose=0)
        self.assertEqual(policy.shape, (1, 100))
        self.assertEqual(value.shape, (1, 1))

if __name__ == "__main__":
    unittest.main()
