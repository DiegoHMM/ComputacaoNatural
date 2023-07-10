import re
import torch.nn as nn
import torch.nn.functional as F

class DynamicLinear_QNet(nn.Module):
    def __init__(self, layers_description):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers_description)):
            # Extrair informações da descrição da camada usando expressões regulares
            layer_info = re.match(r"Linear\((\d+), (\d+)\) (\w+)", layers_description[i])
            if layer_info is None:
                raise ValueError(f"Could not parse layer description: {layers_description[i]}")
            # Obter o número de neurônios de entrada e saída da camada atual
            in_features = int(layer_info.group(1))
            out_features = int(layer_info.group(2))
            # Criar camada Linear
            linear_layer = nn.Linear(in_features, out_features)
            self.layers.append(linear_layer)
            # Adicionar função de ativação se não for a última camada
            if i != len(layers_description) - 1:
                activation = layer_info.group(3)
                if activation == 'Tanh':
                    self.layers.append(nn.Tanh())
                elif activation == 'ReLU':
                    self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x