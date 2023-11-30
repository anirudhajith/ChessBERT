import h5py
import numpy as np

train_f = "data/train.hdf5"
test_f = "data/test.hdf5"

with h5py.File("data/data.hdf5", 'r') as f:
    data = f['embeddings']
    test_len = int(data.shape[0] * 0.01)

    train = data[:-test_len]
    test = data[-test_len:]

with h5py.File(train_f, 'a') as p:
    data = p.create_dataset(f"embeddings", shape=(len(train), 5, 73), maxshape=(None, 5, 73), dtype = np.byte, chunks=True, compression='gzip')
    data[:] = train[:]

with h5py.File(test_f, 'a') as p:
    data = p.create_dataset(f"embeddings", shape=(len(test), 5, 73), maxshape=(None, 5, 73), dtype = np.byte, chunks=True, compression='gzip')
    data[:] = test[:]




