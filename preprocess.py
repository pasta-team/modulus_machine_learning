#!/home/jjs/miniconda3/bin/python

import os
import numpy as np
import pickle,linecache,math
from pasta.element import Element
from pasta.structure import Structure
from scipy.spatial import Voronoi,Delaunay


def order(n):
    if n == 1:
        return 1
    if n < 10:
        return 2
    if n < 47:
        return 4
    if n < 75:
        return 8
    if n < 83:
        return 4
    if n < 123:
        return 8
    if n < 143:
        return 16
    if n < 147:
        return 3
    if n < 162:
        return 6
    if n < 168:
        return 12
    if n < 175:
        return 6
    if n < 191:
        return 12
    if n < 195:
        return 24
    if n < 200:
        return 12
    if n < 221:
        return 24
    return 48

ld = {}
with open('./ld') as f:
    while True:
        line = f.readline()
        if not line:
            break
        linedata = line.split()
        ld[int(linedata[0])] = float(linedata[1])

def tetravol(a,b,c,d):
    return abs(np.dot((a-d),np.cross((b-d),(c-d))))/6

def vol(vor,p):
    dpoints = [list(vor.vertices[v]) for v in vor.regions[vor.point_region[p]]]
    vol = 0
    tri = Delaunay(np.array(dpoints))
    for simplex in tri.simplices:
        vol += tetravol(np.array(dpoints[simplex[0]]),np.array(dpoints[simplex[1]]),np.array(dpoints[simplex[2]]),np.array(dpoints[simplex[3]]))
    return vol

def vol_atom(s):
    sl = s.multiple((3,3,3))
    vor = Voronoi(sl.cart_positions)
    na = s.number_of_atoms
    return [vol(vor,p) for p in range(na*13,na*14)]

energy = {}
with open('./energy') as f:
    while True:
        line = f.readline()
        if not line:
            break
        datum = line.split()
        energy[datum[0]]=float(datum[1])

pathdir = '../../neural/ceder'
dirs = os.listdir(pathdir)

for i in dirs:
    if int(i) not in ld:
        continue
    with open('{:s}/{:s}/info'.format(pathdir,i),'rb') as fd:
        d = pickle.load(fd)
    if d[0]['elasticity'] is None:
        continue
    C = np.array(d[0]['elasticity']['elastic_tensor'])
    CCheck = np.mat(C)
    val,vec = np.linalg.eig(CCheck)
    for j in val:
        if j < -1e-5:
            break
    else:
        try:
            Spie = np.linalg.inv(C)
        except:
            continue
        A=(C[0][0]+C[1][1]+C[2][2])/3
        B=(C[1][2]+C[0][2]+C[0][1])/3
        CC = (C[3][3]+C[4][4]+C[5][5])/3
        a = (Spie[0][0]+Spie[1][1]+Spie[2][2])/3
        b = (Spie[1][2]+Spie[0][1]+Spie[0][2])/3
        c = (Spie[3][3]+Spie[4][4]+Spie[5][5])/3
        KV = (A+2*B)/3
        KR = 1/(3*a+6*b)
        KH = (KV+KR)/2
        GV = (A-B+3*CC)/5
        GR = 5/(4*a-4*b+3*c)
        GH = (GV+GR)/2
        #K = d[0]['elasticity']['K_Voigt_Reuss_Hill']
        #G = d[0]['elasticity']['G_Voigt_Reuss_Hill']
        if KH <= 0 or GH <= 0:
            continue
        if KH > 1000 or GH > 1000:
            continue
        os.mkdir('./elas_data/{:s}'.format(i))
        print(i)
        volume = d[0]['volume']
        ene = d[0]['energy']

        s = Structure.import_from_vasp('{:s}/{:s}/POSCAR'.format(pathdir,i))
        pg_ord = order(s.spg_number)
        no_data = list(s.num_per_species)
        ele = list(s.element_species)
        no_atoms = s.number_of_atoms
        Ele = [Element(i) for i in ele]
        chi = [i.electronegativity for i in Ele]
        row_no = [i.atomic_coordination[0] for i in Ele]
        col_no = [i.atomic_coordination[1] for i in Ele]

        lg_vpa = math.log(volume/no_atoms)
        vd = math.log(no_atoms*s.min_bond_length**3/volume)
        for l,j in zip(ele,no_data):
            ene -= energy[l]*j
        epa = ene/no_atoms
        #data.append((res,math.log(KH),math.log(GH)))
        res = [lg_vpa,epa,row_no,col_no,chi,vd,pg_ord,int(i),no_data,ld[int(i)],vol_atom(s)]

        with open('./elas_data/{:s}/elasticity'.format(i),'wb') as f:
            pickle.dump([res,math.log(KH),math.log(GH)],f)
        
        with open('./elas_data/{:s}/structure'.format(i),'wb') as f:
            pickle.dump(s,f)

