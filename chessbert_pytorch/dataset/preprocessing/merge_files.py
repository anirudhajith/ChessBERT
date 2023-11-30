import h5py
import glob
import numpy as np

with h5py.File('data/data.hdf5', 'w') as f:
    size = 0
    final = None
    for file in glob.glob("data/*.hdf5"):
        p = h5py.File(file, 'r')
        data = p['embeddings']

        if size == 0:
            final = f.create_dataset('embeddings', shape=(0, 5, 73), maxshape=(None, 5, 73), dtype=np.byte, chunks=True, compression='gzip')
    
        final.resize((size + len(data), 5, 73))
        final[-len(data):] = data[:]

        size += len(data)

    print(final.shape)
