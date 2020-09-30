#text classificator
#https://www.tensorflow.org/tutorials/keras/text_classification

# Este cuaderno entrena un modelo de análisis de sentimientos para clasificar las reseñas 
# de películas como positivas o negativas, 
# según el texto de la reseña. Este es un ejemplo de clasificación binaria, o de dos clases, 
# un tipo de problema de aprendizaje automático importante y ampliamente aplicable.

# Utilizará el conjunto de datos de reseñas de películas grandes que contiene el texto de 
# 50.000 reseñas de películas de Internet Movie Database. 
# Estos se dividen en 25,000 revisiones para capacitación y 25,000 revisiones para pruebas. 
# Los conjuntos de entrenamiento y prueba están equilibrados, 
# lo que significa que contienen un número igual de críticas positivas y negativas.

import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


#Descarguemos y extraigamos el conjunto de datos, luego exploremos la estructura del directorio.

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

#Para preparar un conjunto de datos para la clasificación binaria, necesitará dos carpetas en el disco, correspondientes a class_a y class_b.
#Estas serán las críticas positivas y negativas de películas, que se pueden encontrar en aclImdb/train/pos y aclImdb/train/neg. 
#Como el conjunto de datos de IMDB contiene carpetas adicionales, las eliminará antes de usar esta utilidad.

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

#A continuación, utilizará la utilidad text_dataset_from_directory para crear un tf.data.Dataset etiquetado. 
#tf.data es una poderosa colección de herramientas para trabajar con datos.

# Al ejecutar un experimento de aprendizaje automático, se recomienda dividir su conjunto de datos en tres divisiones: entrenamiento , validación y prueba .

# El conjunto de datos de IMDB ya se ha dividido en entrenamiento y prueba, pero carece de un conjunto de validación. 
# Creemos un conjunto de validación usando una división 80:20 de los datos de entrenamiento usando el argumento validation_split continuación.

batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='training', 
    seed=seed)


# A continuación, creará un conjunto de datos de validación y prueba. 
# Utilizará las 5.000 revisiones restantes del conjunto de formación para la validación.

raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/train', 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)


raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'aclImdb/test', 
    batch_size=batch_size)


# A continuación, estandarizará, tokenizará y vectorizará los datos utilizando la útil capa de preprocessing.TextVectorization
# Las etiquetas HTML no son eliminadas por el estandarizador predeterminado, y se aplica esta funcion.

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

# A continuación, creará una capa TextVectorization . usará esta capa para estandarizar, tokenizar y vectorizar nuestros datos. 
# Establece output_mode en int para crear índices enteros únicos para cada token.

max_features = 10000
sequence_length = 250

vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# A continuación, llamará a adapt para ajustar el estado de la capa de preprocesamiento al conjunto de datos. 
# Esto hará que el modelo cree un índice de cadenas a números enteros.

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


# Creemos una función para ver el resultado de usar esta capa para preprocesar algunos datos.

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label


 # retrieve a batch (of 32 reviews and labels) from the dataset
text_batch, label_batch = next(iter(raw_train_ds))
first_review, first_label = text_batch[0], label_batch[0]
print("Review", first_review)
print("Label", raw_train_ds.class_names[first_label])
print("Vectorized review", vectorize_text(first_review, first_label))

print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))



# Está casi listo para entrenar su modelo. Como paso final de preprocesamiento, aplicará la capa TextVectorization que creó anteriormente al conjunto de datos de entrenamiento, validación y prueba.

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


#Crea el modelo - Es hora de crear nuestra red neuronal

embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()


# Función de pérdida y optimizador - Un modelo necesita una funcion de perdida y un optimizador para el entrenamineot.
# Al ser un problema de clasificacion binaria y el modelo genera probabilidad, se utiliza
# la funcoin de perdida losses.BinaryCrossentropy
model.compile(loss=losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=tf.metrics.BinaryAccuracy(threshold=0.0))


#Entrenar el modelo
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)

#Evaluar el modelo
loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuracy: ", accuracy)