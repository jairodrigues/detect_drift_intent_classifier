import math
from bs4 import BeautifulSoup
import pandas as pd
import string
import emoji
import glob
import re


class Cleanner():

    def __init__(self, filesPath):
        self.filesPath = filesPath

    def parse_text(self, text):
        return BeautifulSoup(text, 'html.parser').get_text()

    def remove_mention(self, text):
        return re.sub("@[A-Za-z0-9_]+", '', text)

    def remove_hashtag(self, text):
        return re.sub("#[A-Za-z0-9_]+", '', text)

    def remove_caracters(self, text):
        return re.sub(r"[^a-zA-Z.!?']", ' ', text)

    def remove_numbers(self, text):
        return re.sub("^(?<![0-9-])(\d+)(?![0-9-])", "", text)

    def remove_links(self, text):
        return re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)

    def remove_punct(self, text):
        text = "".join(
            [char for char in text if char not in string.punctuation])
        return text

    def remove_emoji(self, text):
        return emoji.replace_emoji(text, replace="")

    def remove_stop_words(self, text):
        text = text.replace("\n", "")
        text = text.replace("RT", "")
        text = " ".join(text.split())
        str_en = text.encode("ascii", "ignore")
        str_de = str_en.decode()
        return str_de

    def remove_space(self, text):
        return re.sub(r" +", ' ', text)

    def cleaner_text(self, text):
        if (text != text):
            return 'nan'
        text = self.parse_text(text)
        text = self.remove_mention(text)
        text = self.remove_hashtag(text)
        text = self.remove_caracters(text)
        text = self.remove_numbers(text)
        text = self.remove_links(text)
        text = self.remove_emoji(text)
        text = self.remove_punct(text)
        text = self.remove_stop_words(text)
        text = self.remove_space(text)
        return text.lower()

    def read_files_input(self):
        all_filenames = [i for i in glob.glob(self.filesPath)]
        return pd.concat([pd.read_csv(f, sep=";")
                          for f in all_filenames])

    def create_dataframe(self, dataset):
        dataframe = pd.DataFrame()
        dataframe['text'] = dataset['tweet']
        dataframe['intent'] = dataset['intent']
        dataframe['length'] = dataset['length']
        dataframe.to_csv('./data/gold/twittes_cleaner.csv', sep=";")
        return

    def cleaner_dataframe(self):
        dataframe = self.read_files_input()
        dataframe.drop(columns=['edit_history_tweet_ids', 'id'], inplace=True)
        dataframe.text.dropna(inplace=True)
        dataframe.text.drop_duplicates(inplace=True)
        dataframe['tweet'] = [self.cleaner_text(
            text) for text in dataframe.text]
        dataframe['length'] = dataframe['tweet'].apply(len)
        self.create_dataframe(dataframe)
        return
