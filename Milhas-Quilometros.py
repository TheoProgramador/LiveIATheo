#instalações necesarias
#python -m pip install --upgrade pip
#pip install tensorflow
#pip install pandas
#pip install seaborn
#pip install pyparsing==2.4.7

import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#desse jeito tbm
distancia_df = pd.read_csv("comparativo_quilometros_milhas.csv")
distancia_df.reset_index(drop=True, inplace=True)

print (distancia_df)

#pega so os dois ultimos
print (distancia_df.tail(2))

#pega so os cinco primeiros
print (distancia_df.head())

#pega informação
print (distancia_df.info())

#fornece uma descrição do data frame carregado
print (distancia_df.describe())

#sns.scatterplot(distancia_df["Quilometros"], distancia_df["Milhas"])
#erro proposital

sns.scatterplot(data=distancia_df, x="Quilometros", y="Milhas")

#isto mostra o grafico na tela como se fosse no google colab
plt.show()

x_train = distancia_df["Quilometros"]
y_train = distancia_df["Milhas"]

#imprime o que foi carregado
#print(y_train)
#print(x_train)

model = tf.keras.Sequential()
#model.add(tf.keras.layers.Dense(units = 1<=saida , input_shape = [1]<=entrada))
#aqui significa que tera uma saida pra uma entrada
model.add(tf.keras.layers.Dense(units = 100, input_shape = [1]))


model.output_shape # model summary representation
print(model.summary()) # model configuration
print(model.get_config()) # list all weight tensors in the model
print(model.get_weights()) # get weights and biases

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

epochs_hist = model.fit(x_train, y_train, epochs=10) #<= numero de epocas que sera treinado)
#epochs_hist = model.fit(x_train, y_train, epochs=2000)#<=numero de epocas que sera treinado

print(epochs_hist.history.keys())

#no caso de haver outros plots, esta linha faz com que a imagem seja dividida
#sem ela todas as imagens fica na mesma imagem
plt.figure()
plt.plot(epochs_hist.history['loss'])

#enriquecendo a imagem mostrada
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()  # Para mostrar a legenda
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Adiciona uma grade para melhor visualização

# Mostrando o gráfico
plt.show()

print(model.get_weights())
#com isso sera retornado o peso que a IA encontrou no treinamento


#vamos utilizar tudo isso
dist = 158
DistConvertida = model.predict([dist])
print(DistConvertida)

#vamos salvar o treinamento feito para ser reutilizado em outra hora
#aqui se coloca o path completo do treinamento
model_path = "modelDistancia.h5"
model.save(model_path)