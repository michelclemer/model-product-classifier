
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data = {
    'produto': [
        'Cimento CP-32', 'Prego 18x27', 'Areia Fina', 'Tijolo Baiano', 'Cal Hidratada', 'Martelo de Borracha',
        'Parafuso 10mm', 'Rolo de Pintura', 'Serra Circular', 'Lâmpada LED 12W'
    ],
    'categoria': [
        'Materiais Básicos', 'Ferragens', 'Materiais Básicos', 'Materiais Básicos', 'Materiais Básicos', 
        'Ferragens', 'Ferragens', 'Pintura', 'Ferramentas', 'Iluminação'
    ]
}


df = pd.DataFrame(data)

X = df['produto']
y = df['categoria']

vectorizer = CountVectorizer()
X_vect = vectorizer.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

modelo = MultinomialNB()
modelo.fit(X_train, y_train)


y_pred = modelo.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Precisão do modelo: {accuracy * 100:.2f}%')

novos_produtos = ['Broxa para Pintura', 'Chave de Fenda', 'Bloco de Concreto']
novos_produtos_vect = vectorizer.transform(novos_produtos)
predicoes = modelo.predict(novos_produtos_vect)

for produto, categoria in zip(novos_produtos, predicoes):
    print(f'O produto "{produto}" foi classificado como: {categoria}')
