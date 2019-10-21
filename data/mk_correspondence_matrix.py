import sys
import numpy as np

'''Take image / word file and make correspondence matrix.
Output matrix with columns in order corresponding to img matrix
USAGE: python3 mk_correspondence_matrix.py file_word_mapping.txt'''

def read_correspondence_file(filename):
    correspondence = {}
    img_ids = []
    f = open(filename,'r')
    for l in f:
        fields = l.rstrip('\n').split()
        img_id = fields[0] 
        predicate = fields[1]
        correspondence[img_id] = predicate
        img_ids.append(img_id)  #Keep img_ids in order
    return correspondence,img_ids

def mk_matrix(correspondence, img_ids):
    rows = list(set(correspondence.values()))
    columns = img_ids
    m = []
    for r in rows:
        a = np.zeros(len(columns))
        for k,v in correspondence.items():
            if v == r:
                a[columns.index(k)]+=1
        #print(r, a)
        m.append(a)
    return np.array(m), rows

def write_matrix(rows,m):
    f = open(sys.argv[1].replace('.txt','.bin'),'w')
    for i in range(len(rows)):
        f.write(rows[i]+' '+' '.join(str(v) for v in m[i])+'\n')
    f.close()

correspondence, img_ids = read_correspondence_file(sys.argv[1])
m, rows = mk_matrix(correspondence, img_ids)
write_matrix(rows,m)

