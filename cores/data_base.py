from dataloader.imdb import imdb_get_dataloader
from dataloader.snippets import snippets_get_dataloader


class DataBase():
    def __init__(self, type_data):
        self.type_data = type_data

    def get_dataloader(self, df, batch_size, mode, num_workers=0):
        if self.type_data == 'imdb':
            return imdb_get_dataloader(
                df=df,
                batch_size=batch_size,
                mode=mode,
                num_workers=num_workers
            )
        elif self.type_data == 'snippets':
            return snippets_get_dataloader(
                df=df,
                batch_size=batch_size,
                mode=mode,
                num_workers=num_workers
            )
