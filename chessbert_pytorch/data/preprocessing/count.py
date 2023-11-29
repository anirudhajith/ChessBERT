import h5py
import glob

s = 0
for h5_file in glob.glob("data/*.hdf5"):
    with h5py.File(h5_file, 'r') as p:
        data = p['embeddings']
        s += data.shape[0]

print(s)

