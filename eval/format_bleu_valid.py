import sys

f = sys.argv[1]

data = [[0 for i in range(5)] for j in range(4)]

i = 0
j = 4
with open(f) as ifp:
    for line in ifp:
        if line.startswith("../data"):
            if "gen" in line:
                i = 0
            if "rank" in line:
                i = 1
            if "top" in line:
                i = 2
            if 'rank2' in line:
                i = 3
            if '1' in line:
                j = 4
            if ".5" in line:
                j = 3
            if ".25" in line:
                j = 2
            if ".1" in line:
                j = 1
            if ".05" in line:
                j = 0
            continue
        if "." in line:
            data[i][j] = str(float(line.strip()) * 100)
for dd in data:
    print('\t'.join(dd))
