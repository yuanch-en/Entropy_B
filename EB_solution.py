import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

n = 20 # number of cities
m = 50 # number of population

# Generate cities

City = np.random.rand(n, 2)
for c in City:
    x1 = c[0]
    y1 = c[1]
           
    plt.plot(x1, y1, 'go--', markersize=5)
    plt.title('Cities')
    
plt.show()


# Generate initial population
P = np.mgrid[0:m, 0:n][1]

for p in P:
    x = []
    y = []
    np.random.shuffle(p)
  
    for i in range(n):
        x.append(City[p[i]][0])
        y.append(City[p[i]][1])
        
    plt.plot(x, y, alpha = 0.08)
    plt.title('Initial Population')
    
plt.show()
    
    
def distance(P):
    D = []
    for p in P:
        d = 0
        
        for i in range(n-1):
            d += np.linalg.norm(City[p[i+1]]-City[p[i]])
            
        d += np.linalg.norm(City[p[0]]-City[p[n-1]])    
        D.append(d)
     
    return np.array(D)


def score(D):
    s = []
    for d in D:
        a = math.exp(-d)
        s.append(a)
        
    return np.array(s)

def Rank_score(D):
    temp = ((-1)*D).argsort()

    ranks = np.empty_like(temp)

    ranks[temp] = np.arange(len(D))   
    
    return ranks

def B_score(D):
    s = []
    temp = D.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(D))
    for i in range(len(D)):
        b = np.count_nonzero(D)
        prob = b / m
        B = math.log(b)
        a = math.exp(-(ranks[i] + B))
        s.append(a)
        
    return np.array(s)

# Reproduction Methods
# 1. transposition
def mutation1(p):
    i = np.random.randint(0, n-1)
    
    a = p[i]
    b = p[i+1]
    p2 = p
    p2[i+1] = a
    p2[i] = b
    return p2

# 2. shuffle
def mutation2(p):
    i = np.random.randint(0, n-3)
    
    np.random.shuffle(p[i:i+3])
    return p
    
# 3. inversion
def mutation3(p):
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    k1 = 0
    k2 = 0
    if i <= j:
        k1 = i
        k2 = j
    else:
        k1 = j
        k1 = i
        
    p2 = p
    p2[k1:k2] = np.flip(p[k1:k2])
    return p2


# 4. smart mutation
def mutation4(p):
    p2 = np.concatenate((p, p), axis = None)
    p3 = p
    i = np.random.randint(0, n)
    j = np.random.randint(0, n)
    
        
    X1 = City[p2[i]][0]
    Y1 = City[p2[i]][1]
    X2 = City[p2[j]][0]
    Y2 = City[p2[j]][1]
    X3 = City[p2[i+1]][0]
    Y3 = City[p2[i+1]][1]
    X4 = City[p2[j+1]][0]
    Y4 = City[p2[j+1]][1]
            
    if i != j:
        if (X1 != X2) and (X3 != X4):
                
            if (max(X1,X2) > min(X3,X4)) or (min(X1,X2) > max(X3,X4)):
                A1 = (Y1-Y2)/(X1-X2)  # Pay attention to not dividing by zero
                A2 = (Y3-Y4)/(X3-X4)  # Pay attention to not dividing by zero
                b1 = Y1-A1*X1 
                b2 = Y3-A2*X3
                        
                if A1 != A2:
                    Xa = (b2 - b1) / (A1 - A2) 
                    if ( (Xa > max( min(X1,X2), min(X3,X4) )) and
                        (Xa < min( max(X1,X2), max(X3,X4) )) ):
                        p3[i] = p2[j]
                        p3[j] = p2[i]
                        p2[i] = p3[i]
                        p2[j] = p3[j]
                        
    return p3
    
# immiga\ration
def mutation5(p):
    np.random.shuffle(p)
    return p
    
    
# Selection Methods
# 1. Roulette Wheel
def rw(S):
    a = np.random.uniform(0, sum(S))
    index = 0
    for b in S:
        a -= b
        if a <= 0:
            return index
        
        else:
            index += 1
            
# 2. Tournament
def tornament(S):
    i = np.random.randint(0, m)
    j = np.random.randint(0, m)
    a = np.random.uniform(0, S[i] + S[j])
    if a - S[i] <= 0:
        return i
    else:
        return j
    
# Rank
   

    

def reproduction(P, gen, G):
    P2 = []
    #R = 0.1 *(G + 2 * math.sqrt(gen))/(3 * G)
    R = 0.15 *(G + 2 * math.sqrt(gen))/(3 * G)
    #R = 0.1
    # R = 0.01
    #R = 1
    #R = 0.5 * (G + 2 * math.sqrt(gen))/(3 * G) # mutation probaility
    for i in range(m):
        j = np.random.randint(0, m)
        k = np.random.rand()
        
       
        #if k <= R*1/2:
            #p2 = mutation4(P[j])
        
        #if R* 1/2 < k <= R:
            #p2 = mutation1(P[j])
            
        if k <= R/2:
            p2 = mutation2(P[j])
            
        if R* 1/2 < k <= R:
            p2 = mutation3(P[j])
        else:
            p2 = P[j]
                
        P2.append(p2)
        P2.append(P[i])
    return P2


def select(P2, S):
    P3 = P
    for i in range(m):
        k = np.random.randint(0, 2)
        if k == 3:
            P3[i] = P2[rw(S)]
            
            
        else:
            P3[i] = P2[tornament(S)]
            
    return P3


def main(iter):
    D = []
    S = []
    for j in range(iter): # number of generations
        P2 = reproduction(P, j, iter)        
        
        D2 = distance(P2)
        S2 = B_score(D2)
        P3 = select(P2, S2)
        d = distance(P3)
        s = B_score(d)
        D.append(d)
        S.append(s)
        
        for p in P3:
            x = []
            y = []
          
            for i in range(n):
                x.append(City[p[i]][0])
                y.append(City[p[i]][1])
                
            
            if j == (iter - 1):
                x.append(City[p[0]][0])
                y.append(City[p[0]][1])    
                plt.plot(x, y, alpha = 0.1)
                plt.title('Final iteration')
          
    plt.show()
    return (D, S)
                
     

iter = 1000


R = main(iter)
D = R[0]
S = R[1]

X = np.arange(iter)

D_std = np.empty(iter)
D_mean = np.empty(iter)
S_std = np.empty(iter)
S_mean = np.empty(iter)

for i in range(iter):
    D_std[i] = np.std(D[i])
    D_mean[i] = np.mean(D[i])
    S_std[i] = np.std(S[i])
    S_mean[i] = np.mean(S[i])

plt.plot(X, D_mean)
plt.title('Mean Distance vs Iteration')
plt.xlabel('iteration')
plt.ylabel('distance')
plt.show()


plt.show()