import src.cleaning.cleaning as cleaning


cleaner = cleaning.Cleanner('./data/silver/**/*.csv')
cleaner.cleaner_dataframe()
