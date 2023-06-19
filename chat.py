import random
import json
import time
import torch
import sys
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
from sklearn.metrics.pairwise import cosine_similarity
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intentions.json', 'r') as json_data:
    intentions = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "UQAC Assistant"

def get_response(msg):
    
    # on token la phrase d'entree + ajout dans bag_of_words puis tenseur entree
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)

    # reshape pour avoir une seul phrase à la fois (batch size 1)
    X = X.reshape(1, X.shape[0]) 
    X = torch.from_numpy(X).to(device)

    output = model(X)
    # return valeur max + index du tag
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    # on recupère la proba et on verifie si le seuil de confiance est valdié dans la requete
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.7:
        for intention in intentions['intentions']:
            if tag == intention["tag"]:
                return random.choice(intention['responses'])

    else:
        # on recherche les intentions avec la plus grande similarité cosinus
        similarities = []
        for intention in intentions['intentions']:
            for pattern in intention['patterns']:

                # calcul de la similarité cosinus
                pattern_words = tokenize(pattern)
                pattern_bag = bag_of_words(pattern_words, all_words)
                pattern_bag = pattern_bag.reshape(1, pattern_bag.shape[0])
                print(len(pattern_bag))
                similarity = cosine_similarity(X.cpu().numpy(), pattern_bag)[0][0]
                similarities.append((similarity, intention['tag']))
        
        # on trie les intentions par ordre décroissant de similarité cosinus
        similarities = sorted(similarities, key=lambda x: x[0], reverse=True)

        # on renvoie la suggestion la plus similaire avec une réponse prédéfinie
        if similarities[0][0] > 0:
            for intention in intentions['intentions']:
                if similarities[0][1] == intention['tag']:
                    return f"{bot_name}: Je n'ai pas compris votre question, voulez-vous dire '{intention['patterns'][0]}' ?"
        else : 
            # si la suggestion est trop peu similaire, on renvoie une réponse générique
            return f"{bot_name}: Je n'ai pas compris votre question, je suis désolé..."
        


# # -----------------------------------------------------------------
# # A RETIRER POUR L'APP.PY
# # Cela sert pour les tests dans le terminal
# # Ce code prend en compte la latence du chat bot a répondre
# # afin de le rendre plus humain mais pas appliqué sur l'interface à date :-( 
# # -----------------------------------------------------------------

# import time

# while True:
#     sentence = input("You: ")
#     if sentence == "quit":
#         break

    
#     sentence = tokenize(sentence)
#     X = bag_of_words(sentence, all_words)
#     X = X.reshape(1, X.shape[0]) # reshape pour avoir une seul phrase à la fois (batch size 1)
#     X = torch.from_numpy(X).to(device)


#     output = model(X)
#     # return valeur max + index du tag
#     _, predicted = torch.max(output, dim=1)

#     tag = tags[predicted.item()]

#     # on recupère la proba et on verifie si le seuil de confiance est valdié dans la requete
#     probs = torch.softmax(output, dim=1)
#     prob = probs[0][predicted.item()]

#     if prob.item() > 0.7:

#         for intention in intentions['intentions']:
#             if tag == intention["tag"]:
#                 sys.stdout.write(f"{bot_name}: ")
#                 for char in "...":
#                     sys.stdout.write(char)

#                     # force l'impression immédiate des caractères
#                     sys.stdout.flush()  

#                     # pause de 0.3 seconde entre chaque caractère
#                     time.sleep(0.3)  

#                 print(random.choice(intention['responses']))

#     else:
#         # on recherche les intentions avec la plus grande similarité cosinus
#         similarities = []

#         for intention in intentions['intentions']:
#             for pattern in intention['patterns']:

#                 # calcul de la similarité cosinus
#                 pattern_words = tokenize(pattern)
#                 pattern_bag = bag_of_words(pattern_words, all_words)
#                 similarity = cosine_similarity(X.cpu().numpy(), pattern_bag)[0][0]
#                 similarities.append((similarity, intention['tag']))
        
#         # on trie les intentions par ordre décroissant de similarité cosinus
#         similarities = sorted(similarities, key=lambda x: x[0], reverse=True)

#         # on renvoie la suggestion la plus similaire avec une réponse prédéfinie
#         if similarities[0][0] > 0:
#             for intention in intentions['intentions']:
#                 if similarities[0][1] == intention['tag']:
#                     # print(f"{bot_name}: Je n'ai pas compris votre question, voulez-vous dire '{intention['patterns'][0]}' ?")
#                     sys.stdout.write(f"{bot_name}: ")
#                     for char in "...":
#                         sys.stdout.write(char)

#                         # force l'impression immédiate des caractères
#                         sys.stdout.flush()  

#                         # pause de 0.3 seconde entre chaque caractère
#                         time.sleep(0.3)  
#                     print(f"Je n'ai pas compris votre question, voulez-vous dire '{intention['patterns'][0]}' ?")
#         else : 
#             # si la suggestion est trop peu similaire, on renvoie une réponse générique

#             # print(f"{bot_name}: ", end="")
#             # time.sleep(1) # pause de 1 seconde
#             # print("Je n'ai pas compris votre question, je suis désolé...")
#             sys.stdout.write(f"{bot_name}: ")
#             for char in "...":
#                 sys.stdout.write(char)

#                 # force l'impression immédiate des caractères
#                 sys.stdout.flush()  

#                 # pause de 0.3 seconde entre chaque caractère
#                 time.sleep(0.3)  
#             print("Je n'ai pas compris votre question, je suis désolé...")


# # -----------------------------------------------------------------
# # FIN DE A RETIRER POUR L'APP.PY
# # -----------------------------------------------------------------
