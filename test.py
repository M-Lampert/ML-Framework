import numpy as np

import test_utils
from framework.activation_layers import ReLU, Sigmoid, Softmax, Tanh
from framework.input_layer import DefaultInputLayer
from framework.layers import FullyConnectedLayer
from framework.losses import CrossEntropy, MeanSquaredError
from framework.network import NeuralNetwork
from framework.optimizer import SGD
from framework.tensor import Tensor
from framework.utils import classes_to_one_hot_vector, one_hot_vector_to_classes
from test_utils import get_inputs, in_shape, out_shape


def test_mse():
    input, _ = get_inputs()
    target = np.ones(input.shape)
    exp_outs = test_utils.Exp_Outputs_MSE()

    mse_loss = MeanSquaredError()
    output = mse_loss.forward(Tensor(input), Tensor(target))
    assert np.abs(output - exp_outs.get_exp_forward_out()) < 1e-7

    output = mse_loss.backward(Tensor(input), Tensor(target))
    assert np.all(np.abs(output.get_deltas() - exp_outs.get_exp_gradients()) < 1e-7)


def test_CrossEntropy():
    input, _ = get_inputs()
    target = np.zeros(input.shape)
    target[np.argmax(input)] = 1
    exp_outs = test_utils.Exp_Outputs_CrossEntropy()

    mse_loss = CrossEntropy()
    output = mse_loss.forward(Tensor(input), Tensor(target))
    assert np.abs(output - exp_outs.get_exp_forward_out()) < 1e-7

    output = mse_loss.backward(Tensor(input), Tensor(target))
    assert np.all(np.abs(output.get_deltas() - exp_outs.get_exp_gradients()) < 1e-7)


def test_Dense_MSE():
    input, weights = get_inputs()
    exp_outs = test_utils.Exp_Outputs_Dense_MSE()

    layers = [
        FullyConnectedLayer(in_shape, out_shape),
    ]
    layers[0].weights = Tensor(weights)
    nn = NeuralNetwork(DefaultInputLayer(), layers)

    out_tensors = nn.forward(input)
    assert np.all(np.abs(out_tensors[-1].get_elements() - exp_outs.get_exp_forward_out()) < 1e-7)
    print("Forward pass is correct for the given example!")

    SGD.update(
        nn=nn,
        loss=MeanSquaredError(),
        lr=0.01,
        epochs=1,
        data=(np.array([input]), np.ones((1, 1, 8))),
    )

    assert np.all(np.abs(nn.layers[0].bias.get_elements() - exp_outs.get_exp_updated_bias()) < 1e-7)
    assert np.all(np.abs(nn.layers[0].weights.get_elements() - exp_outs.get_exp_updated_weights()) < 1e-7)
    print("Weight and Bias gradients are correct for the given example!")


def test_Dense_Softmax_CrossEntropy():
    input, weights = get_inputs()
    exp_outs = test_utils.Exp_Outputs_Dense_Softmax_CrossEntropy()

    layers = [FullyConnectedLayer(in_shape, out_shape), Softmax()]
    layers[0].weights = Tensor(weights)
    nn = NeuralNetwork(DefaultInputLayer(), layers)

    out_tensors = nn.forward(input)
    print(out_tensors[-1])
    assert np.all(np.abs(out_tensors[-1].get_elements() - exp_outs.get_exp_forward_out()) < 1e-7)
    print("Forward pass is correct for the given example!")

    SGD.update(
        nn=nn,
        loss=CrossEntropy(),
        lr=0.01,
        epochs=1,
        data=(np.array([input]), np.ones((1, 1, 8))),
    )

    assert np.all(np.abs(nn.layers[0].bias.get_elements() - exp_outs.get_exp_updated_bias()) < 1e-7)
    assert np.all(np.abs(nn.layers[0].weights.get_elements() - exp_outs.get_exp_updated_weights()) < 1e-7)
    print("Weight and Bias gradients are correct for the given example!")


def test_Dense_Sigmoid_Dense_Softmax_CrossEntropy():
    input, weights = get_inputs(2)
    exp_outs = test_utils.Exp_Outputs_Dense_Sigmoid_Dense_Softmax_CrossEntropy()

    layers = [FullyConnectedLayer(in_shape, out_shape), Sigmoid(), FullyConnectedLayer(out_shape, out_shape), Softmax()]
    layers[0].weights = Tensor(weights[0])
    layers[2].weights = Tensor(weights[1])
    nn = NeuralNetwork(DefaultInputLayer(), layers)

    out_tensors = nn.forward(input)
    assert np.all(np.abs(out_tensors[-1].get_elements() - exp_outs.get_exp_forward_out()) < 1e-7)
    print("Forward pass is correct for the given example!")

    SGD.update(
        nn=nn,
        loss=CrossEntropy(),
        lr=0.01,
        epochs=1,
        data=(np.array([input]), np.ones((1, 1, 8))),
    )

    assert np.all(np.abs(nn.layers[0].bias.get_elements() - exp_outs.get_exp_updated_bias_1()) < 1e-7)
    assert np.all(np.abs(nn.layers[0].weights.get_elements() - exp_outs.get_exp_updated_weights_1()) < 1e-7)
    assert np.all(np.abs(nn.layers[2].bias.get_elements() - exp_outs.get_exp_updated_bias_2()) < 1e-7)
    assert np.all(np.abs(nn.layers[2].weights.get_elements() - exp_outs.get_exp_updated_weights_2()) < 1e-7)
    print("Weight and Bias gradients are correct for the given example!")


def test_Dense_ReLU_Dense_Softmax_CrossEntropy():
    input, weights = get_inputs(2)
    exp_outs = test_utils.Exp_Outputs_Dense_ReLU_Dense_Softmax_CrossEntropy()

    layers = [FullyConnectedLayer(in_shape, out_shape), ReLU(), FullyConnectedLayer(out_shape, out_shape), Softmax()]
    layers[0].weights = Tensor(weights[0])
    layers[2].weights = Tensor(weights[1])
    nn = NeuralNetwork(DefaultInputLayer(), layers)

    out_tensors = nn.forward(input)
    assert np.all(np.abs(out_tensors[-1].get_elements() - exp_outs.get_exp_forward_out()) < 1e-7)
    print("Forward pass is correct for the given example!")

    SGD.update(
        nn=nn,
        loss=CrossEntropy(),
        lr=0.01,
        epochs=1,
        data=(np.array([input]), np.ones((1, 1, 8))),
    )

    assert np.all(np.abs(nn.layers[0].bias.get_elements() - exp_outs.get_exp_updated_bias_1()) < 1e-7)
    assert np.all(np.abs(nn.layers[0].weights.get_elements() - exp_outs.get_exp_updated_weights_1()) < 1e-7)
    assert np.all(np.abs(nn.layers[2].bias.get_elements() - exp_outs.get_exp_updated_bias_2()) < 1e-7)
    assert np.all(np.abs(nn.layers[2].weights.get_elements() - exp_outs.get_exp_updated_weights_2()) < 1e-7)
    print("Weight and Bias gradients are correct for the given example!")


def test_Dense_Tanh_Dense_Softmax_CrossEntropy():
    input, weights = get_inputs(2)
    exp_outs = test_utils.Exp_Outputs_Dense_Tanh_Dense_Softmax_CrossEntropy()

    layers = [FullyConnectedLayer(in_shape, out_shape), Tanh(), FullyConnectedLayer(out_shape, out_shape), Softmax()]
    layers[0].weights = Tensor(weights[0])
    layers[2].weights = Tensor(weights[1])
    nn = NeuralNetwork(DefaultInputLayer(), layers)

    out_tensors = nn.forward(input)
    assert np.all(np.abs(out_tensors[-1].get_elements() - exp_outs.get_exp_forward_out()) < 1e-7)
    print("Forward pass is correct for the given example!")

    SGD.update(
        nn=nn,
        loss=CrossEntropy(),
        lr=0.01,
        epochs=1,
        data=(np.array([input]), np.ones((1, 1, 8))),
    )

    assert np.all(np.abs(nn.layers[0].bias.get_elements() - exp_outs.get_exp_updated_bias_1()) < 1e-7)
    assert np.all(np.abs(nn.layers[0].weights.get_elements() - exp_outs.get_exp_updated_weights_1()) < 1e-7)
    assert np.all(np.abs(nn.layers[2].bias.get_elements() - exp_outs.get_exp_updated_bias_2()) < 1e-7)
    assert np.all(np.abs(nn.layers[2].weights.get_elements() - exp_outs.get_exp_updated_weights_2()) < 1e-7)
    print("Weight and Bias gradients are correct for the given example!")


def test_one_hot_conversion():
    class_names = ["fish", "frog", "bird", "kangaroo"]
    y_s = np.array(["fish", "bird", "bird", "kangaroo", "frog"])
    exp_result = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
    assert np.all(classes_to_one_hot_vector(y_s, class_names) == exp_result)
    print("Conversion from class_names to one hot vector was successful!")

    assert np.all(one_hot_vector_to_classes(exp_result, class_names, sparse=False) == y_s)
    assert np.all(one_hot_vector_to_classes(np.argmax(exp_result, axis=1), class_names) == y_s)
    print("Conversion from (sparse) one hot vector to class names was successful!")
