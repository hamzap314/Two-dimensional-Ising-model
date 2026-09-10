import numpy as np
from functools import reduce
from scipy.linalg import expm
from numpy import cos,sin
import matplotlib.pyplot as plt

# Pauli matrices
I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


# Return the column vector for a computational basis state |bitstring⟩.
def ket(bitstring):
    n = len(bitstring)
    dim = 2 ** n
    index = int(bitstring, 2)
    vec = np.zeros((dim, 1), dtype=complex)
    vec[index] = 1.0
    return vec


# Return the matrix form of |ket_bits⟩⟨bra_bits|."""
def density_matrix(bra_bits: str, ket_bits: str) -> np.ndarray:
    if len(ket_bits) != len(bra_bits):
        raise ValueError("ket and bra must have the same number of qubits")
    k = ket(ket_bits)   # column vector
    b = ket(bra_bits)   # column vector; bra = b.conj().T
    return k @ b.T

# Tensor product of a sequence of operators.
def kron_n(*ops):
    return reduce(np.kron, ops)


# Embed a single-site operator `op` at site i in an n-qubit chain.
def single_site_op(op, i, n):
    ops = [I] * n
    ops[i] = op
    return kron_n(*ops)


# Embed op1 at site i and op2 at site j in an n-qubit chain.
def two_site_op(op1, i, op2, j, n):
    ops = [I] * n
    ops[i] = op1
    ops[j] = op2
    return kron_n(*ops)


def transverse_ising_1D_H(J, h, Bz, n):
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)

    # ZZ interaction (periodic boundary conditions: i = 0 ... n-2)
    for i in range(n):
        H += two_site_op(Z, i, Z, (i+1)%n, n)

    # Transverse field: h * sum X_i
    for i in range(n):
        H += h * single_site_op(X, i, n)

    # Longitudinal field: Bz * sum Z_i
    for i in range(n):
        H += Bz * single_site_op(Z, i, n)

    return J * H


# def basisket(i, d=2):
#     """Column vector for basis state i in a d-dimensional space."""
#     v = np.zeros((d, 1), dtype=complex)
#     v[i] = 1.0
#     return v

def basis_den_mat(i, j, d=2):
     """Outer product |i><j| in a d-dimensional space."""
     return ket(f"{i}") @ ket(f"{j}").T


def singleQbasisOp(L, whichQ, i, j):
    """
    Mathematica:
        list = Table[PauliMatrix[0], {m, 1, L}]   (* L identity matrices *)
        list[[whichQ]] = basisDenMat[{i}, {j}]    (* replace one with |i><j| *)
        KroneckerProduct @@ list                   (* tensor the whole list *)

    Place the single-qubit operator |i><j| at site whichQ (1-indexed)
    in an L-qubit chain; all other sites get the 2x2 identity.
    """

    return single_site_op(basis_den_mat(i, j), whichQ, L) 


def TcomponentGen(whichQA, whichQB, i, ip, j, jp, U, rho):
    """
    Mathematica:
        Tr[ ρ . singleQbasisOp[log2(dim), whichQA, j, i]
              . U†
              . singleQbasisOp[log2(dim), whichQB, jp, ip]
              . U ]

    Computes a single matrix element of the T-matrix (transfer / process
    tensor component) for a unitary U and state ρ.

    All indices i, ip, j, jp are 0-based to match Python/numpy convention.
    whichQA, whichQB are 1-based to match Mathematica.
    """
    L = int(np.log2(U.shape[0]))                  # number of qubits

    A  = singleQbasisOp(L, whichQA, j,  i )       # |j><i|  at site whichQA
    B  = singleQbasisOp(L, whichQB, jp, ip)       # |jp><ip| at site whichQB
    Ud = U.conj().T

    return np.trace(rho @ A @ Ud @ B @ U)


def TmatrixGen(whichQA, whichQB, U, rho):
    """
    Mathematica:
        ArrayReshape[
            Table[TcomponentGen[whichQA, whichQB, i, ip, j, jp, U, ρ],
                  {i,0,1},{ip,0,1},{j,0,1},{jp,0,1}],
            {dim, dim}
        ]

    Builds the full T-matrix by evaluating TcomponentGen over all
    combinations of (i, ip, j, jp) ∈ {0,1}^4, then reshapes the
    resulting 2x2x2x2 tensor into a (dim x dim) matrix.
    """
    d = U.shape[0]                                 # full Hilbert space dim

    # 4-index tensor T[i, ip, j, jp]
    T = np.zeros((2, 2, 2, 2), dtype=complex)
    for i  in range(2):
        for ip in range(2):
            for j  in range(2):
                for jp in range(2):
                    T[i, ip, j, jp] = TcomponentGen(
                        whichQA, whichQB, i, ip, j, jp, U, rho
                    )

    # ArrayReshape: flatten (i,ip,j,jp) → row index runs over (i,ip),
    # column index over (j,jp)  →  shape (d_local², d_local²) = (4, 4)
    return T.reshape(d, d)


# KroneckerProduct[iketbraj[0,0], iketbraj[0,0]]  →  |00><00|
rho_test = np.kron(basis_den_mat(0, 0), basis_den_mat(0, 0))

# U: use a simple example
th = 3*np.pi/4
U = np.array([[1,0,0,0],[0,cos(th),-sin(th),0],[0,sin(th),cos(th),0],[0,0,0,1]])

result = TmatrixGen(0, 1, U, rho_test)
print("TmatrixGen(1, 2, U, |00><00|):")
print(np.round(result.real, 6))


# def single_site_pauli(pauli, whichQ, n):
#     """
#     Embed a single-site Pauli matrix at site whichQ (1-indexed)
#     in an n-qubit chain; all other sites get the 2x2 identity.
#     Equivalent to Mathematica's sigmayi[whichQ, n] etc.
#     """
#     ops = [I2.copy() for _ in range(n)]
#     ops[whichQ - 1] = pauli
#     return reduce(np.kron, ops)

# def sigmayi(whichQ, n):
#     """Single-site Pauli Y at site whichQ (1-indexed) in an n-qubit chain."""
#     return single_site_pauli(Y, whichQ, n)


# ── In[266-268] ──────────────────────────────────────────────────────────
numSpins = 11

H = transverse_ising_1D_H(J=1, h=-1.05, Bz=0.5, n=numSpins)

# MatrixExp[-(1/100) H]  →  matrix exponential of -H/100
thermalState = expm(-H / 100)


# ── In[271]: blackline[t] ─────────────────────────────────────────────────
def blackline(t):
    """
    Mathematica:
        U = MatrixExp[-i H t]
        Abs[Tr[Comm[sigmayi[1,n], U†.sigmayi[6,n].U] . U.thermalState.U†]]

    The commutator Comm[A, B] = A.B - B.A, so expanding:
        Comm[σY_1, U†σY_6 U] = σY_1.(U†σY_6 U) - (U†σY_6 U).σY_1

    Then traced against U.thermalState.U†.
    """
    U  = expm(-1j * H * t)
    Ud = U.conj().T

    sY1 = single_site_op(Y, 1, numSpins)
    sY6 = single_site_op(Y, 6, numSpins)

    evolved_sY6 = Ud @ sY6 @ U                    # U†.sigmayi[6].U
    commutator  = sY1 @ evolved_sY6 - evolved_sY6 @ sY1   # Comm[σY1, U†σY6 U]
    rho_t       = U @ thermalState @ Ud            # U.thermalState.U†

    return abs(np.trace(commutator @ rho_t))


# ── In[272]: ListPlot over t = 0..20 ─────────────────────────────────────
t_values       = np.arange(0,20,0.5)                    # {t, 0, 20}  integer steps
blackline_vals = [blackline(t) for t in t_values]

plt.figure(figsize=(8, 4))
plt.plot(t_values, blackline_vals, marker='o', markersize=3, linewidth=1)
plt.xlabel("t")
plt.ylabel(r"$|\mathrm{Tr}([Y_1,\, U^\dagger Y_6 U]\, rho)|$")
plt.title("Blackline: OTO-style correlator")
plt.grid(True)
plt.tight_layout()
plt.show()


