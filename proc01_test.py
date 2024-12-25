import unittest
import numpy as np
from lib.base import Variable, Function

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):
        return np.exp(self.input.data) * gy

def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.)
        self.assertEqual(y.data, expected)

    def test_backward_square(self):
        x = Variable(np.array(3.))
        y = square(x)
        y.backward()
        expected = np.array(6.)
        self.assertEqual(expected, x.grad)

    def test_backward_exp(self):
        x = Variable(np.array(3.))
        y = exp(x)
        y.backward()
        expected = np.exp(3.)
        self.assertEqual(expected, x.grad)

    def test_backward_deep(self):
        x = Variable(np.array(3.))
        y = square(x)
        z = exp(y)
        z.backward()
        expected = np.exp(9.) * 6
        self.assertEqual(expected, x.grad)