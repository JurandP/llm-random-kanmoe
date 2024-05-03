import unittest

import torch

from research.grad_norm.grad_norm import GradientSTDNormLayer


class TestGradientSTDNormLayer(unittest.TestCase):
    def test_forward_id(self):
        layer = GradientSTDNormLayer(c=1)

        x = torch.randn(3, 4, 4)
        y = layer(x)

        self.assertEqual(x.shape, y.shape)
        self.assertTrue(torch.equal(x, y))

    def test_backward_norm(self):
        c_values = [i / 10 for i in range(0, 11)]  # from 0 to 1

        for c in c_values:
            with self.subTest(c=c):
                layer = GradientSTDNormLayer(c=c)

                x = torch.randn(3, 4, requires_grad=True)
                grad = torch.randn(3, 4)
                y = layer(x)
                y.backward(grad)

                with torch.no_grad():
                    expected_grad = grad / (torch.pow((grad.std()), c) + 1e-8)

                self.assertTrue(torch.allclose(x.grad, expected_grad))

    def test_backward_unary(self):
        layer = GradientSTDNormLayer(c=1)

        x = torch.randn(1, requires_grad=True)
        grad = torch.randn(1)
        y = layer(x)
        y.backward(grad)

        self.assertTrue(torch.equal(x.grad, grad))

    def test_grad_dist_to_1(self):
        c = 1
        l1 = torch.nn.Linear(4, 4)
        l1.requires_grad = False
        norm = GradientSTDNormLayer(c=c)

        x = torch.randn(4, requires_grad=True)
        x.retain_grad()

        # path 1
        x.grad = None
        y = norm(x)
        y = l1(y)
        y.sum().backward()
        grad1 = x.grad.clone()

        # path 2
        x.grad = None
        y = l1(x)
        y.sum().backward()
        grad2 = x.grad.clone()

        self.assertTrue(torch.abs(grad1.std() - 1) < torch.abs(grad2.std() - 1))
