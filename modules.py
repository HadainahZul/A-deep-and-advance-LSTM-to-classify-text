# %%
from tensorflow.keras.layers import Dense, LSTM, Dropout, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential
import re


def text_cleaning(text):
    """This function removes texts with anomalies such as 
    URLs, @NAME, WASHINGTON (Reuters) and also to convert 
    text into lowercase.

    Args:
        text (str): Raw text.

    Returns:
        text (str): Cleaned text
    """

    # Have URL (bit.ly) ---> Things to remove
    text = re.sub('bit.ly/\d\w{1,10}', '', text)

    # Have @realDonaldTrump
    text = re.sub('@[^\s]+', '', text)

    # WASHINGTON (Reuters) : New Header
    text = re.sub('^.*?\)\s*-', '', text)
    
    # [1901 EST]
    text = re.sub('\[.*?EST\]', '', text)

    # $number and special chracters and punctuations
    text = re.sub('[^a-zA-Z]', ' ', text).lower()
  
    return text

def lstm_model_creation(num_words, nb_classes, embedding_layer = 64, dropout=0.3, num_neurons=64):
    """This function creates LSTM model with embedding layer, 
    2 LSTM layers and 1 output

    Args:
        num_words (int): number of vocabulary
        nb_classes (int): number of classess
        embedding_layer (int, optional): The number of output ofembedding layer. Defaults to 64.
        dropout (float, optional): The rate dropout. Defaults to 0.3.
        num_neurons (int, optional): Number of brain cells. Defaults to 64.

    Returns:
        model: Returns the model created using sequential API
    """
    model = Sequential()
    model.add(Embedding(num_words, embedding_layer))
    model.add(LSTM(embedding_layer, return_sequences=True))
    model.add(Dropout(dropout))
    model.add(LSTM(num_neurons))
    model.add(Dropout(dropout))
    model.add(Dense(nb_classes, activation='softmax'))
    model.summary()

    plot_model(model, show_shapes=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

    return model

# %%
