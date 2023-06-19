import numpy as np
import random
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet
from nltk.corpus import stopwords
from hyperopt import fmin, tpe, hp, Trials


#Ouverture du json
with open('intentions.json', 'r') as f:
    intentions = json.load(f)
"""
pour chaque objet d'intentions.json :
on apprned le tag et on l'associe aux patterns tokenizee
"""
all_words = []
tags = []
phraseTokenizee_tag = []
stop_words = set(stopwords.words('french'))

for intention in intentions['intentions']:
    tag = intention['tag']
    tags.append(tag)
    for pattern in intention['patterns']:
        # tokenize les mots de pattern
        w = tokenize(pattern)
        # retirer les stopwords de la liste de mots
        w = [word for word in w if word not in stop_words]
        # ajout à la liste complete des mots
        all_words.extend(w)
        # associer la phrase tokenizee et le tag dans xy
        phraseTokenizee_tag.append((w, tag))

# trouver les racines
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# tri par ordre alphabétique + pas de doublons des mots et des tags
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# affichage
print(len(phraseTokenizee_tag), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# trainnnn
x_train = []
y_train = []
for (pattern_sentence, tag) in phraseTokenizee_tag:
    x_train.append(bag_of_words(pattern_sentence, all_words))

    #index car crossentropy en loss et faut des class labels
    y_train.append(tags.index(tag)) 

x_train = np.array(x_train)
y_train = np.array(y_train)


"""
recehrche des best hyperparameters avec hyperopt
"""

space = {
    'num_epochs': hp.choice('num_epochs', range(500, 1000, 100)),
    'batch_size': hp.choice('batch_size', range(16, 129, 16)),
    'learning_rate': hp.choice('learning_rate', [0.1, 0.01, 0.001, 0.0001]),
    'hidden_size': hp.choice('hidden_size', range(8, 64, 8))
}


class ChatDataset(Dataset):
    """
    met les donnees en format facilement manipulable pour PyTorch
    """
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # return une paire de tenseur (entree/sortie) !
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # taille de l'echantillon
    def __len__(self):
        return self.n_samples

def objective(params):

    print("VOICI les PARAMS : ", params)
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    hidden_size = params['hidden_size']

    input_size = len(x_train[0])
    output_size = len(tags)

    dataset = ChatDataset()

    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # perte fonction + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train
    for epoch in range(num_epochs):
        # on itère sur chaque batch dans train_loader
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            outputs = model(words)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print(f'Loss final : {loss.item():.4f}')

    return loss.item()

# on recup les meilleurs hyper_paramètres
trials = Trials()
best = fmin(objective, space, algo=tpe.suggest, max_evals=5, trials=trials)

# ATTENTION C'EST EN INDEX !!!!
print("\nVoici les meilleurs valeurs trouvées par hyperopt sous forme d'index : ")
print(best)

index_num_epochs = best['num_epochs']
index_batch_size = best['batch_size']
index_learning_rate = best['learning_rate']
index_hidden_size = best['hidden_size']

input_size = len(x_train[0])
output_size = len(tags)

# A COMMENTER
print("Voici l'index du best batch_size : ", index_batch_size)
# FIN COMMENTER

# on prend les valeurs aux index données
num_epochs = 500 + (100*index_num_epochs)
batch_size = 16 + (16*index_batch_size)
lr = [0.1, 0.01, 0.001, 0.0001]
learning_rate = lr[index_learning_rate]
hidden_size = 8 + 8 * (index_hidden_size)

batch_int = int(batch_size)

#insérer le nouvel entrainement du modèle avec les best paramètrees
dataset = ChatDataset()

train_loader = DataLoader(dataset=dataset,
                            batch_size=batch_int,
                            shuffle=True,
                            num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# modèle avec les meilleures paramètres !
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# perte fonction + optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train avec toujours les meilleures paramètres !
for epoch in range(num_epochs):
    # on itère sur chaque batch dans train_loader
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Loss final : {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}


FILE = "data.pth"
torch.save(data, FILE)
print(f'train complete, file {FILE}')