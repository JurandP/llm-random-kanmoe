import torch

from lizrd.core import misc
from lizrd.core.misc import Chunked
from lizrd.support.test_utils import GeneralTestCase


class TestDense(GeneralTestCase):
    def test_basic(self):
        batch, dinp, dout = 4, 32, 64
        layer = misc.DenseEinMix(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_more_dims(self):
        batch, seql, dinp, dout = 4, 8, 32, 64
        layer = misc.DenseEinMix(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, seql, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seql, dout))


class TestLinear(GeneralTestCase):
    def test_basic(self):
        batch, dinp, dout = 4, 32, 64
        layer = misc.Linear(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_more_dims(self):
        batch, seql, dinp, dout = 4, 8, 32, 64
        layer = misc.Linear(dinp, dout)
        input = torch.normal(0.0, 1.0, (batch, seql, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seql, dout))


class TestEinMix(GeneralTestCase):
    def test_no_ellipsis(self):
        batch, dinp, dout = 4, 32, 64
        layer = misc.EinMix(
            "b d -> b f", weight_shape="d f", bias_shape="f", d=dinp, f=dout
        )
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_one_ellipsis(self):
        batch, dinp, dout = 4, 32, 64
        layer = misc.EinMix(
            "... d -> ... f", weight_shape="d f", bias_shape="f", d=dinp, f=dout
        )
        input = torch.normal(0.0, 1.0, (batch, dinp))
        output = layer(input)
        self.assertShape(output, (batch, dout))

    def test_two_ellipsis(self):
        batch, seqlen, dinp, dout = 4, 2, 32, 64
        layer = misc.EinMix(
            "... d -> ... f", weight_shape="d f", bias_shape="f", d=dinp, f=dout
        )
        input = torch.normal(0.0, 1.0, (batch, seqlen, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seqlen, dout))

    def test_two_ellipsis(self):
        batch, seqlen, whatever, dinp, dout = 5, 7, 3, 32, 64
        layer = misc.EinMix(
            "... d -> ... f", weight_shape="d f", bias_shape="f", d=dinp, f=dout
        )
        input = torch.normal(0.0, 1.0, (batch, seqlen, whatever, dinp))
        output = layer(input)
        self.assertShape(output, (batch, seqlen, whatever, dout))


class TestChungus(GeneralTestCase):
    def test_outputs_and_grads(self):
        # create the model inputs
        torch.manual_seed(0)
        # create a simple Sequential model
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 20),
            torch.nn.Linear(20, 20),
            torch.nn.Linear(20, 70),
        )

        model_chunked = torch.nn.Sequential(
            torch.nn.Linear(100, 20),
            Chunked(torch.nn.Linear(20, 20), n_chunks=2),
            torch.nn.Linear(20, 70),
        )

        # clone the model weights
        model_chunked._modules["0"].weight.data = model._modules[
            "0"
        ].weight.data.clone()
        model_chunked._modules["0"].bias.data = model._modules["0"].bias.data.clone()
        model_chunked._modules["1"].module.weight.data = model._modules[
            "1"
        ].weight.data.clone()
        model_chunked._modules["1"].module.bias.data = model._modules[
            "1"
        ].bias.data.clone()
        model_chunked._modules["2"].weight.data = model._modules[
            "2"
        ].weight.data.clone()
        model_chunked._modules["2"].bias.data = model._modules["2"].bias.data.clone()

        x = torch.rand(100, 100)
        y = x.clone()
        ###################### ORIGINAL ###########################
        # get the model output and save it to prevent any modifications
        output_original = model(x)

        model.zero_grad()
        output_original.sum().backward()
        grad_vanilla = {}
        for name, param in model.named_parameters():
            grad_vanilla[name] = param.grad.data.clone()

        ###################### CHUNGUS ############################
        output_chunked = model_chunked(y)
        model_chunked.zero_grad()
        output_chunked.sum().backward()

        grad_chunked = {}
        for name, param in model_chunked.named_parameters():
            grad_chunked[name] = param.grad.data.clone()

        ####################### COMPARE ORIG MODEL AND CHUNGUS #######################
        # compare the output and parameters gradients
        assert torch.isclose(
            output_original, output_chunked
        ).all(), f" output failed, log of difference is: {torch.log10((output_original - output_chunked).abs().max())}, max difference is: {((output_original - output_chunked).abs().max())}, argmax is {((output_original - output_chunked).abs().argmax())}"
        for name in grad_vanilla:
            name_for_chunked = name
            if "1" in name:
                name_for_chunked = name.replace("1", "1.module")

            chung = grad_chunked[name_for_chunked]
            van = grad_vanilla[name]

            assert torch.isclose(
                chung, van
            ).all(), f"parameter {name} failed, log of difference is: {torch.log10((chung - van).abs().max())}"
