import numpy as np
import matplotlib.pyplot as plt
import random


def generate_index_pairs(L):
    pairs = []
    for j in range(L):
        for i in range(L):
  
            # Every point with the one to its right (i+1%L), at a fixed y-value (j)
            pairs.append([(i,j), ((i+1)%(L),j)])
            
            # If we switch the y and x values of both points, we get the pair for a point and the one under it
            pairs.append([(j,i), (j,(i+1)%(L))])
    
    return pairs


def H(lattice):
    energy = 0
    index_pairs = generate_index_pairs(lattice.shape[0])
    for pair in index_pairs:
        energy = energy + (lattice[pair[0]] * lattice[pair[1]])    
    return -J*energy


def H_neighbors(lattice, point):
    L = lattice.shape[0]
    energy = 0
    np_point = np.array(point)
    points = np.array([np_point + np.array([0,1]), 
                       np_point + np.array([1,0]), 
                       np_point + np.array([0,-1]), 
                       np_point + np.array([-1,0])]) % L
    
    for p in points:
        energy = energy + (lattice[point] * lattice[tuple(p)])
    
    return -J*energy


def metropolis_update(lattice, T):
    L = lattice.shape[0]
    random_spin_location = (random.randint(0, L-1), random.randint(0, L-1))
    energy_change = -2*H_neighbors(lattice, random_spin_location)
    
    if energy_change <= 0:
        lattice[random_spin_location] = lattice[random_spin_location] * -1
    else:
        prob = np.exp(-energy_change/(k*T))
        if random.random() < prob:
            lattice[random_spin_location] = lattice[random_spin_location] * -1
        else:
            pass

    return lattice


# Main code -----------------------

k = 1
L = 20
J = 1
N = L*L
T_values = np.linspace(1.6, 3.2, 21)
e_values = []
esqr_values = []
m_values = []
msqr_values = []
m4_values = []
C_values = []
chi_values = []


equilibration_sweeps = 300
measurement_sweeps = 400

for T in T_values:
    rng = np.random.default_rng()
    lattice = rng.choice([-1,1], (L, L))

    for i in range(equilibration_sweeps):
        for j in range(N):
            metropolis_update(lattice, T)

    e = 0
    esqr = 0
    m = 0
    msqr = 0
    m4 = 0

    for i in range(measurement_sweeps):
        for j in range(N):
            metropolis_update(lattice, T)

        e = e + H(lattice)
        esqr = esqr + H(lattice)**2
        m = m + np.abs(np.sum(lattice))
        msqr = msqr + (np.abs(np.sum(lattice))**2)
        m4 = m4 + (np.abs(np.sum(lattice))**4)

    e = float(e)/measurement_sweeps
    esqr = float(esqr)/measurement_sweeps
    m = float(m)/measurement_sweeps
    msqr = float(msqr)/measurement_sweeps
    m4 = float(m4)/measurement_sweeps

    C = (esqr - e**2) / (k*(T**2)*N)
    chi = (msqr - m**2) / (k*T*N)

    e_values.append(e)
    esqr_values.append(esqr)
    m_values.append(m)
    msqr_values.append(msqr)
    m4_values.append(m4)
    C_values.append(C)
    chi_values.append(chi)
    

plt.figure()
plt.title("m vs T")
plt.scatter(T_values, np.array(m_values)/float(N))

plt.figure()
plt.title("C vs T")
plt.scatter(T_values, C_values)

plt.figure()
plt.title("chi vs T")
plt.scatter(T_values, chi_values)

plt.show()



