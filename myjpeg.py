# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:54:34 2022

@author: Cassi
"""
import math 

def ppm_tokenize(stream):
    while True:
        l=stream.readline()
        if l=='':
            break
        new_l=l.split(' ')
        for i in range(len(new_l)):
            if new_l[i][0]=="#":
                break          
            yield new_l[i]

def ppm_load(stream):
    a=ppm_tokenize(stream)
    next(a)
    w=int(next(a))
    h=int(next(a))
    next(a)
    l=[[0 for i in range(len(w))] for j in range(len(h))]
    for j in range(h):
        for i in range(w):
            b=int(next(a))
            c=int(next(a))
            d=int(next(a))
            l[j][i]=(b, c, d)
        
        return (w, h, l)
    
def ppm_save(w, h, img, output):
    text=f'P3\n{w}\n{h}\n255\n'
    for j in range(h):
        for i in range(x):
            for l in range(3):
                text+=f'{img[j][i][l]}\n'
    output.write(text)
    
def RGB2YCbCr(r, g, b):
    a=[round(0.299*r+0.587*g+0.114*b), round(128-0.168736*r-0.331234*g+0.5*b), round(128+0.5*r-0.418688*g-0.081312*b)]
    for i in range(3):
        if a[i]<0:
            a[i]=0
        elif a[i]>255:
            a[i]=255
    return (a[0], a[1], a[2])

def YCbCr2RGB(Y, Cb, Cr):
    a=[round(Y+1.402*(Cr-128)), round(Y-0.344136*(Cb-128)-0.714136*(Cr-128)), round(Y+1.772*(Cb-128))]
    for i in range(3):
        if a[i]<0:
            a[i]=0
        elif a[i]>255:
            a[i]=255
    return (a[0], a[1], a[2])
                
def img_RGB2YCbCr(img):
    w=len(img[0])
    h=len(img)
    Y=[[RGB2YCbCr(img[j][i])[0] for i in range(len(w))] for j in range(len(h))]
    Cb=[[RGB2YCbCr(img[j][i])[1] for i in range(len(w))] for j in range(len(h))]
    Cr=[[RGB2YCbCr(img[j][i])[2] for i in range(len(w))] for j in range(len(h))]
    return (Y, Cb, Cr)

def img_YCbCr2RGB(Y, Cb, Cr):
    w=len(Y[0])
    h=len(Y)
    
    img=[[YCbCr2RGB(Y[j][i], Cb[j][i], Cr[j][i]) for i in range(len(w))] for j in range(len(h))]
    
    return img

def subsampling(w, h, C, a, b):
    X=[[0 for j in range((w+a-1)//a)] for i in range((h+b-1)//b)]
    
    for i in range(0, h, a):
        for j in range(0, w, b):
            h_bound=min(h, i+a)
            w_bound=min(w, j+b)
            sz=(h_bound-i)*(w_bound-j)
            X[i//a][j//b]=round(sum([C[k][m] for k in range(i, h_bound) for m in range(j, w_bound)])/sz)   
    return X
            
def extrapolate(w, h, C, a, b):
    return [[C[i//a][j//b] for j in range(w)] for i in range(h)]
      
def block_splitting(w, h, C):
    L=[]
    if  w%8!=0:
        for j in range(8-w%8):
            for i in range(h):
                C[i].append(C[i][w-1])
    if h%8!=0:
        for i in range(8-h%8):
            C.append(C[h-1])
            
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            L.append([C[k][j:j+8] for k in range(i, i+8)])

    for i in L:
        yield i

def matrix_mult(M, N):
    return [[sum(M[i][k]*N[k][j] for k in range(len(M[0]))) for j in range(len(N[0]))] for i in range(len(M))]
    
def matrix_tran(M):
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]
    
def delta(i):
    if i ==0:
        return math.sqrt(1/2)
    return 1

def C_(n):
    return [[delta(i)*math.sqrt(2/n)*math.cos((math.pi/n)*(j+1/2)*i) for j in range(n)] for i in range(n)]
        
def DCT(v):
    n=len(v)
    return [sum(delta(i)*math.sqrt(2/n)*v[j]*math.cos((math.pi/n)*(j+1/2)*i) for j in range(n)) for i in range(n)]

def IDCT(v):
    n=len(v)
    A=matrix_mult([v], C_(n))
    return A[0]       

def DCT2(m, n, A):
    return matrix_mult(matrix_mult(C_(m), A), matrix_tran(C_(n)))

def IDCT2(m, n, A):
    return matrix_mult(matrix_tran(C_(m)), matrix_mult(A, C_(n)))

def redalpha(i):
    c=0
    while i>8:
        i-=16
        c+=1
    return ((-1)**(c%2), abs(i))

def ncoeff8(i, j):
    if i==0:
        return (1, 4)
    else:
        return redalpha(i*(2*j+1))
def calc():
    return [[(math.cos((ncoeff8(i, j)[1]*math.pi)/16)*ncoeff8(i, j)[0])/2 for j in range(8)] for i in range(8)]

def VDCT_chen(v):
    Coef=calc()
    L=[]
    Sum=sum(v)*Coef[0][0]
    L.append(Sum/2)
    k=1
    while k<8:
        Sum=0
        if k==4:
           Sum=Coef[k][0]*(v[0]+v[3]+v[4]+v[7]-v[1]-v[2]-v[5]-v[6])
           
        elif k%4==2:
            Sum=Coef[k][1]*(v[1]+v[6]-v[5]-v[2])+Coef[k][0]*(v[0]+v[7]-v[3]-v[4])
       
        else:
            i=0
            while i<4:
                Sum=Sum+Coef[k][i]*(v[i]-v[7-i])
                i+=1
        L.append(Sum/2)
        k+=1
    return L

def DCT_Chen(A):
    K=[]
    s=0
    ret_M=[[0 for j in range(8)] for i in range(8)]
    while s<8:
        
        A[s]=VDCT_chen(A[s])
        s+=1
    s=0
    
    while s<8:
        The_list=[]
        for i in range(8):
            The_list.append(A[s][i])
        The_list=VDCT_chen(The_list)
        for i in range(8):
            ret_M[k][i]=The_list[i]
        s+=1
    return matrix_tran(ret_M)
    
                
    
  
    
def IDCT_Chen(A):
    a=1
    
def quantization(A, Q):
    return [[round(A[i][j]/Q[i][j]) for j in range(8)] for i in range(8)]

def quantizationI(A, Q):
    return [[A[i][j]*Q[i][j] for j in range(8)] for i in range(8)]

LQM = [
  [16, 11, 10, 16,  24,  40,  51,  61],
  [12, 12, 14, 19,  26,  58,  60,  55],
  [14, 13, 16, 24,  40,  57,  69,  56],
  [14, 17, 22, 29,  51,  87,  80,  62],
  [18, 22, 37, 56,  68, 109, 103,  77],
  [24, 35, 55, 64,  81, 104, 113,  92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103,  99],
]

CQM = [
  [17, 18, 24, 47, 99, 99, 99, 99],
  [18, 21, 26, 66, 99, 99, 99, 99],
  [24, 26, 56, 99, 99, 99, 99, 99],
  [47, 66, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
  [99, 99, 99, 99, 99, 99, 99, 99],
]

def s(phi):
    if phi>=50:
        return 200-2*phi
    else:
        return round(5000/phi)
      
def Qmatrix(isY, phi):
    if isY:
        return [[math.ceil((50+s(phi)*LQM[i][j])) for j in range(8)] for i in range(8)]
    else:
        return [[math.ceil((50+s(phi)*CQM[i][j])) for j in range(8)] for i in range(8)]
    
def zigzag(A):
    yield A[0][0]
    (i, j)=(0, 1)
    yield A[i][j]
    left=True
    while (i, j)!=(7, 7):
        if left:
            for k in range(j-i):
                i+=1
                j-=1
                yield A[i][j]
            if i<7:
                i+=1
            else:
                j+=1
            
            yield A[i][j]
            left=False
        else:
            for k in range(i-j):
                i-=1
                j+=1
                yield A[i][j]
            if j<7:
                j+=1
            else:
                i+=1
            yield A[i][j]
            left=True
            

def rle0(g):
    k=0
    for i in range(len(g)):
        a=g[i]
        if a!=0:
            yield(k, a)
            k=0
        else:
            k+=1
            
        
        