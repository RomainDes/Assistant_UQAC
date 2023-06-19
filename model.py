import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    """
    4 couches cachées : meilleur apprentissage
    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, 2*hidden_size)
        self.l3 = nn.Linear(2*hidden_size, 2*hidden_size)
        self.l4 = nn.Linear(2*hidden_size, num_classes)
        self.dropout = nn.Dropout(p=0.5)
        self.lrelu = nn.LeakyReLU(0.1)
        
    def forward(self, x):
        """
        Abscence de softmax et activation car on utilise la fonction de perte crossentropy 
        qui contient un activation softmax 
        """
        out = self.l1(x)
        out = self.lrelu(out)
        out = self.dropout(out)
        out = self.l2(out)
        out = self.lrelu(out)
        out = self.dropout(out)
        out = self.l3(out)
        out = self.lrelu(out)
        out = self.dropout(out)
        out = self.l4(out)
        return out
