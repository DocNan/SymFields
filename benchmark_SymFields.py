# benchmark SymFields module
import sympy
from SymFields import *

pi = sympy.pi

# cylinder coordinates
r, phi, z = sympy.symbols('r, phi, z')
X = [r, phi, z]
U = sympy.Function('U')
U = U(r, phi, z)

grad = Grad(U, X, coordinate='Cylinder')
curl_grad = Curl(grad, X, coordinate='Cylinder')

A_r = sympy.Function('A_r')
A_phi = sympy.Function('A_phi')
A_z = sympy.Function('A_z')
A_r = A_r(r, phi, z)
A_phi = A_phi(r, phi, z)
A_z = A_z(r, phi, z)
A = [A_r, A_phi, A_z]


divergence = Div(A, X, coordinate='Cylinder')
curl = Curl(A, X, coordinate='Cylinder')
div_curl = Div(curl, X, coordinate='Cylinder')
div_curl.doit()

# Laplacian operator
L = Div(grad, X, coordinate='Cylinder', evaluation=1)

# test metric tensor of toroidal coordinate
r, theta, zeta, R0 = sympy.symbols('r, theta, zeta, R0')
# co_M = Metric(coordinate='Toroidal')


# test non-orthogonal shifted cylinder coordinate
r_2, phi_2, z_2, alpha = sympy.symbols('r_2, phi_2, z_2, alpha')
alpha = pi/3
Xi = [r_2, phi_2, z_2]
x = r_2*sympy.cos(phi_2)
y = r_2*sympy.sin(phi_2) + z_2*sympy.sin(alpha)
z = z_2*sympy.cos(alpha)
R = [x, y, z]
M_co = Metric(Xi=Xi, R=R, coordinate='shifted cylinder', contra=0, evaluation=1)
M_co = sympy.simplify(M_co)
M_contra = M_co.inv(method='GE')

U2 = sympy.Function('U2')
U2 = U2(r_2, phi_2, z_2)

grad2 = Grad(U2, Xi, coordinate='shifted cylinder', metric=M_contra, evaluation=1)
curl_grad2 = Curl(grad2, Xi, coordinate='shifted cylinder', metric=M_contra, evaluation=1)
curl2_grad2 = sympy.simplify(curl_grad2[0])


A_r2 = sympy.Function('A_r2')
A_phi2 = sympy.Function('A_phi2')
A_z2 = sympy.Function('A_z2')
A_r2 = A_r2(r_2, phi_2, z_2)
A_phi2 = A_phi2(r_2, phi_2, z_2)
A_z2 = A_z2(r_2, phi_2, z_2)
A2 = [A_r2, A_phi2, A_z2]

curl2 = Curl(A2, Xi, coordinate='shifted cylinder', metric=M_contra, evaluation=1)
div_curl2 = Div(curl2, Xi, coordinate='shifted cylinder', metric=M_contra, evaluation=1)
sympy.simplify(div_curl2)


