import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# Carregando o treinamento
loaded_model = load_model("modelDistancia.h5")

# Suponhamos que você queira prever alguns valores de distancia
sample_distancia_values = np.array([10,200,5])
predicted_distancia = loaded_model.predict(sample_distancia_values)
print(predicted_distancia)

# Se você quiser continuar o treinamento, pode fazer isso também.
# Por exemplo, com novos dados ou talvez os mesmos dados para mais épocas:
# loaded_model.fit(new_x_train, new_y_train, epochs=50)