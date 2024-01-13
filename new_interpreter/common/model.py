import torch

class MyModel(torch.nn.Module):
    def __init__(self, input_size, widths, activations, output_size, weights=None, bias=None):
        super(MyModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        layers = []
        counter = 0
        for i in range(len(widths)):
            if i == 0:
                layers.append(torch.nn.Linear(input_size, widths[i]))
                if weights != None:
                    layers[-1].weight.data = weights[counter]
                    layers[-1].bias.data = bias[counter]
                    counter += 1
            else:
                layers.append(torch.nn.Linear(widths[i-1], widths[i]))
                if weights != None:
                    layers[-1].weight.data = weights[counter]
                    layers[-1].bias.data = bias[counter]
                    counter += 1

            activation = torch.nn.ReLU()
            if activations[i] == 'tanh':
                activation = torch.nn.Tanh()
            layers.append(activation)

        layers.append(torch.nn.Linear(widths[-1], output_size))
        if weights != None:
            layers[-1].weight.data = weights[counter]
            layers[-1].bias.data = bias[counter]
            counter += 1
        self.layers = torch.nn.Sequential(*layers)