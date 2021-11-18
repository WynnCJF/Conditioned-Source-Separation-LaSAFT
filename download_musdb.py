import musdb

musdb_data = musdb.DB(root="musdb18/musdb18", download=False, subsets="train")
print(len(musdb_data))