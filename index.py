import src.cleaning.cleaning as cleaning
import src.preprocess.preprocess as preprocessing
import src.training.training as training
import src.models.models as models
import tensorflow as tf
import pandas as pd
import src.training.checkpoint as checkpoint

# cleaner = cleaning.Cleanner('./data/silver/**/*.csv')
# cleaner.cleaner_dataframe()

dataset = pd.read_csv("./data/gold/twittes_cleaner.csv", sep=";")

preprocess = preprocessing.Preprocess()
sorted_all = preprocess.preprocess(dataset)
training.Training(models.DCNNBERTEmbedding, checkpoint, sorted_all)
