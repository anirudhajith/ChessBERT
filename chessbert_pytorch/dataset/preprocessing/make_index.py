import json

d = {}
d[4] = 1
d[2] = 3
d[3] = 5
d[5] = 7
d[6] = 8
d[1] = 9
d[-4] = 17
d[-2] = 19
d[-3] = 21
d[-5] = 23
d[-6] = 24
d[-1] = 25

with open("piece_index.json", 'w') as f:
    json.dump(d, f)
