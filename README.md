# SymFields
Symbolic vector Fields analysis tool in python.
Detailed introduction can be seen in this ArXiv paper: https://arxiv.org/abs/2012.10723 

Corrections to the introduction manuscipt on ArXiv:

(1) In abstract and introduction sections, sympy module does have some simple vector field analysis function under Cartesian coordinates, but it could not be extened to the more general curvilinear coordinates. SymFields could be a very good compliment to sympy in the fields of vector fields analysis.

(2) In page 4, example of wrong divergence calculation, the Jacobian should be: $J = H_r H_{\phi} H_z = r$ insteady of $r^2$. And the related divergence shall be:

$$\nabla\cdot\vec{A} = \frac{1}{r}\frac{\partial}{\partial r}(r A_r) + \frac{\partial A_{\phi}}{\partial \phi} + \frac{\partial A_z}{\partial z}$$

Which is also an example of wrong experssion of divergence.  
