# SymFields
Symbolic vector Fields analysis tool in python.
Detailed introduction can be seen in this ArXiv paper: https://arxiv.org/abs/2012.10723 

Corrections to the introduction manuscript on ArXiv:

(1) In abstract and introduction sections, sympy module does have some simple vector field analysis function under Cartesian coordinates, but it could not be extened to the more general curvilinear coordinates. SymFields could be a very good compliment to sympy [Meurer et al. (2017), SymPy: symbolic computing in Python. PeerJ Comput. Sci. 3:e103; DOI 10.7717/peerj- cs.103] in the fields of vector fields analysis.

(2) In page 4, example of wrong divergence calculation, the Jacobian should be: $J = H_r H_{\phi} H_z = r$ insteady of $r^2$. And the related divergence shall be:

$$\nabla\cdot\vec{A} = \frac{1}{r}\frac{\partial}{\partial r}(r A_r) + \frac{\partial A_{\phi}}{\partial \phi} + \frac{\partial A_z}{\partial z}$$

Which is also an example of wrong experssion of divergence.  

# updates to Symfields
In version 2, addition features are added to SymFields, the includes:
## Additional functions
Plus()         # plus two vectors
Minus()        # minus two vectors
Multi_Scalar() # multiply vector with scalar

## wave vector format of outputs
Using optional input (wavevector = 1), the nabla operator will be converted to wave vector format as $i\vec{k}$.

## un-normalized units outputs
Using optinal input (normal=0), the normalization of the units and physical values will be skipped in the calculation. This is useful to make some theoretical comparison. But to get real meaningful physical outputs, the default normalized output (normal=1) should be used.

## Publication:
And the ArXiv preprint manuscript is formally published on the Future Technology Conference (FTC) in Springer FTC collcection [https://link.springer.com/chapter/10.1007/978-3-030-89906-6_45]. The previous errors in ArXiv preprint are corrected in this finally publication. And it can be formally cited as: 
[Chu, N. (2022). SymFields: An Open Source Symbolic Fields Analysis Tool for General Curvilinear Coordinates in Python. In: Arai, K. (eds) Proceedings of the Future Technologies Conference (FTC) 2021, Volume 1. FTC 2021. Lecture Notes in Networks and Systems, vol 358. Springer, Cham. https://doi.org/10.1007/978-3-030-89906-6_45]

