# src/clasificador.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

class ClasificadorTexto:
    def __init__(self, categorias=None, max_len=50, vocab_size=1000):
        """
        Inicializa el clasificador de texto simple basado en Keras.
        """
        self.categorias = categorias or ["académico", "comercial", "recreativo", "noticia"]
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self.model = None

    def entrenar(self, textos, etiquetas, epochs=10):
        """
        Entrena un modelo simple de clasificación de texto.
        textos: lista de strings
        etiquetas: lista de enteros (índices de categoría)
        """
        self.tokenizer.fit_on_texts(textos)
        secuencias = self.tokenizer.texts_to_sequences(textos)
        secuencias_pad = pad_sequences(secuencias, maxlen=self.max_len, padding="post")

        # Convertir etiquetas a one-hot
        etiquetas_oh = tf.keras.utils.to_categorical(etiquetas, num_classes=len(self.categorias))

        self.model = Sequential([
            Embedding(self.vocab_size, 16, input_length=self.max_len),
            GlobalAveragePooling1D(),
            Dense(24, activation='relu'),
            Dense(len(self.categorias), activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.fit(secuencias_pad, etiquetas_oh, epochs=epochs, verbose=1)

    def predecir(self, textos):
        """
        Predice la categoría de una lista de textos.
        Retorna una lista de strings con el nombre de la categoría.
        """
        secuencias = self.tokenizer.texts_to_sequences(textos)
        secuencias_pad = pad_sequences(secuencias, maxlen=self.max_len, padding="post")
        predicciones = self.model.predict(secuencias_pad)
        indices = np.argmax(predicciones, axis=1)
        return [self.categorias[i] for i in indices]

