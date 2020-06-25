# this module is focused on symbolic calculation for vector/tensor analysis with differential geometry operations.
import sympy
import warnings
pi = sympy.pi
print('------------------------------------------------------')
print('symbolic calculation for Fields analysis is loaded   !')
print('The default differential operators are not evaluated !')
print('To evalute the expressions:                          !')
print('method 1: use expression.doit()                      !')
print('method 2: set evaluation=1 for nabla operators       !')
print('------------------------------------------------------')
'''
SymFields.py module to solve the Fields analysis problems especially in MHD thoery.
chunan@ipp.ac.cn 2020.06.05 last modified on 2020.06.17
'''




# ------------------------------------------------------------------------------
# calculate nabla operators with Lame coefficients in orthogonal coordinates.
# reference: 中科大数学教究室，高等数学导论，中科大出版社，合肥。
def Metric(Xi=0, R=0, coordinate='Cartesian', evaluation=0, contra=1):
    '''
    calculate controvariant matric tensor g^{ij} = grad(X^i)\dot grad(X^j) 
    coordinate == 'Cartesian', 'Cylinder', 'Sphere' and 'Toroidal'
    where R = (x, y, z) is the displacement in Cartesian coordinate, 
    where X = (xi^1, xi^2, xi^3) is the displacement in the general Curvilinear coordinate.
    chunan@ipp.ac.cn 2020.06.08
    '''
    if coordinate == 'Cartesian':
        print('Coordinate Cartesian: (x1, x2, x3) = (x, y, z)')
        contra_metric = sympy.Matrix([[sympy.S(1), 0, 0], [0, sympy.S(1), 0], [0, 0, sympy.S(1)]])
        co_metric = contra_metric.inv('GE')
        # H = [sympy.S(1), sympy.S(1), sympy.S(1)]
    elif coordinate == 'Cylinder':
        print('Coordinate Cylinder: (x1, x2, x3) = (r, phi, z)')
        contra_metric = sympy.Matrix([[sympy.S(1), 0, 0], [0, Xi[0]**-2, 0], [0, 0, sympy.S(1)]])
        co_metric = contra_metric.inv('GE')
        # H = [sympy.S(1), x[0], sympy.S(1)]
    elif coordinate == 'Sphere':
        print('Coordinate Sphere: (x1, x2, x3) = (r, theta, phi)')
        contra_metric = sympy.Matrix([[sympy.S(1), 0, 0], [0, Xi[0]**-2, 0], [0, 0, (Xi[0]*sympy.sin(Xi[1]))**-2]])
        co_metric = contra_metric.inv('GE')
        # H = [sympy.S(1), x[0], x[0].sympy.sin(x[1])]
    elif coordinate == 'Toroidal':
        print('Coordinate Toroidal: (x1, x2, x3) = (r, theta, zeta)')
        R0 = sympy.symbols('R0')
        contra_metric = sympy.Matrix([[sympy.S(1), 0, 0], [0, Xi[0]**-2, 0], [0, 0, (R0 + Xi[0]*sympy.sin(Xi[1]))**-2]])
        co_metric = contra_metric.inv('GE')
    else:
        print('New coordinate: ', coordinate)
        x = R[0]
        y = R[1]
        z = R[2]
        Xi_1 = Xi[0]
        Xi_2 = Xi[1]
        Xi_3 = Xi[2]
        if contra == 1:
            G1 = Grad(Xi_1, [x,y,z], coordinate='Cartesian', evaluation=evaluation)
            G2 = Grad(Xi_2, [x,y,z], coordinate='Cartesian', evaluation=evaluation)
            G3 = Grad(Xi_3, [x,y,z], coordinate='Cartesian', evaluation=evaluation)
            contra_metric = sympy.Matrix([[Dot(G1,G1), Dot(G1,G2), Dot(G1,G3)], [Dot(G2,G1), Dot(G2,G2), Dot(G2,G3)], [Dot(G3,G1), Dot(G3,G2), Dot(G3,G3)]])    
        elif contra == 0:
            if evaluation == 1:
                g1 = [x.diff(Xi_1), y.diff(Xi_1), z.diff(Xi_1)]
                g2 = [x.diff(Xi_2), y.diff(Xi_2), z.diff(Xi_2)]
                g3 = [x.diff(Xi_3), y.diff(Xi_3), z.diff(Xi_3)]
            elif evaluation == 0:
                g1 = [sympy.Derivative(x, Xi_1), sympy.Derivative(y, Xi_1), sympy.Derivative(z, Xi_1)]
                g2 = [sympy.Derivative(x, Xi_2), sympy.Derivative(y, Xi_2), sympy.Derivative(z, Xi_2)]
                g3 = [sympy.Derivative(x, Xi_3), sympy.Derivative(y, Xi_3), sympy.Derivative(z, Xi_3)]                
            co_metric = sympy.Matrix([[Dot(g1,g1), Dot(g1,g2), Dot(g1,g3)], [Dot(g2,g1), Dot(g2,g2), Dot(g2,g3)], [Dot(g3,g1), Dot(g3,g2), Dot(g3,g3)]])

    if contra == 1:
        # convert the 'Mul' object metric tensor to sympy (dense) Matrix type
        M = contra_metric
    elif contra == 0:
        # calculate covariant matric tensor using symbolic matrix inverse
        M = co_metric 
    return M




def Jacobian(metric, evaluation=0, contra=1):
    '''
    calculate Jacobian from contravariant metric tensor g^{ij}
    where J = sqrt(|g_{ij}}) = 1/sqrt(|g^{ij}|)
    contra == 1: J = sqrt(|g_{ij}|)
    contra == 0: J = sqrt(|g^{ij}|)
    chunan@ipp.ac.cn 2020.06.08
    '''
    # Jacob = sympy.powdenest(sympy.sqrt(metric[0,0]*(metric[1,1]*metric[2,2] - metric[1,2]*metric[2,1]) - metric[0,1]*(metric[1,0]*metric[2,2] - metric[1,2]*metric[2,0]) + metric[0,2]*(metric[1,0]*metric[2,1] - metric[1,1]*metric[2,0])), force=True)
    Jacob = sympy.powdenest(sympy.sqrt(metric.det()), force=True)
    
    if contra == 1:
        J = Jacob**-1
    else:
        J = Jacob
    
    return J




def Lame(metric, evalutation=0, contra=1):
    '''
    calculate Lame coefficients from contravariant metric tensor g^{ij}
    where Hi = sqrt(g_{ii}) = 1/sqrt(g^{ii})
    contra == 1: J = sqrt(|g_{ij}|)
    contra == 0: J = sqrt(|g^{ij}|)
    chunan@ipp.ac.cn 2020.06.17
    '''    
    if contra == 1:
        # calculate Lame coefficients from cotra-variant metric tensor
        co_metric = metric.inv(method='GE')
        H = [sympy.powdenest(sympy.sqrt(co_metric[0,0]), force=True), sympy.powdenest(sympy.sqrt(co_metric[1,1]), force=True), sympy.powdenest(sympy.sqrt(co_metric[2,2]), force=True)]
    elif contra == 0:
        # calculate Lame coefficients from co-variant metric tensor
        H = [sympy.powdenest(sympy.sqrt(metric[0,0]), force=True), sympy.powdenest(sympy.sqrt(metric[1,1]), force=True), sympy.powdenest(sympy.sqrt(metric[2,2]), force=True)]
        
    return H




def Dot(A, B, X=[sympy.symbols('x'), sympy.symbols('y'), sympy.symbols('z')], coordinate='Cartesian', metric=[[sympy.S(1), 0, 0], [0, sympy.S(1), 0], [0, 0, sympy.S(1)]]):
    '''
    Calculate dot product in general curvilinear coordinates (orthogonal or non-orthogonal).
    chunan@ipp.ac.cn 2020.06.17
    '''
    if (coordinate == 'Cartesian') or (coordinate == 'Cylinder') or (coordinate == 'Sphere') or (coordinate == 'Toroidal'):
        metric = Metric(coordinate=coordinate, contra=1)
    else:
        print('Coordinate: '+coordinate+' is used, you should be explictly given at input !')
        metric = metric
        
    H = Lame(metric, contra=1)
        
    D = A[0]*B[0]*metric[0,0]/(H[0]*H[0]) + A[0]*B[1]*metric[0,1]/(H[0]*H[1]) + A[0]*B[2]*metric[0,2]/(H[0]*H[2]) \
      + A[1]*B[0]*metric[1,0]/(H[1]*H[0]) + A[1]*B[1]*metric[1,1]/(H[1]*H[1]) + A[1]*B[2]*metric[1,2]/(H[1]*H[2]) \
      + A[2]*B[0]*metric[2,0]/(H[2]*H[0]) + A[2]*B[1]*metric[2,1]/(H[2]*H[1]) + A[2]*B[2]*metric[2,2]/(H[2]*H[2])
      
    return D




def Cross(A, B, X=[sympy.symbols('x'), sympy.symbols('y'), sympy.symbols('z')], coordinate='Cartesian', metric=[[sympy.S(1), 0, 0], [0, sympy.S(1), 0], [0, 0, sympy.S(1)]], evaluation=0):
    '''
    Calculate cross product in general curvilinear coordinates (orthogonal or non-orthogonal).
    chunan@ipp.ac.cn 2020.06.17    
    '''
    if (coordinate == 'Cartesian') or (coordinate == 'Cylinder') or (coordinate == 'Sphere') or (coordinate == 'Toroidal'):
        metric = Metric(X=X, coordinate=coordinate, contra=1)
    else:
        print('Coordinate: '+coordinate+' is used, you should be explictly given at input !')
        metric = metric
        
    J = Jacobian(metric)
    H = Lame(metric)
    # get the covariant metric tensor with matrix inverse
    g_co = metric.inv(method='GE')
    
    # get the normalized cotravariant component of vector fields A and B
    A1 = g_co[0,0]*A[0]/H[0] + g_co[0,1]*A[1]/H[1] + g_co[0,2]*A[2]/H[2]
    A2 = g_co[1,0]*A[0]/H[0] + g_co[1,1]*A[1]/H[1] + g_co[1,2]*A[2]/H[2]
    A3 = g_co[2,0]*A[0]/H[0] + g_co[2,1]*A[1]/H[1] + g_co[2,2]*A[2]/H[2]
    B1 = g_co[0,0]*B[0]/H[0] + g_co[0,1]*B[1]/H[1] + g_co[0,2]*B[2]/H[2]
    B2 = g_co[1,0]*B[0]/H[0] + g_co[1,1]*B[1]/H[1] + g_co[1,2]*B[2]/H[2]
    B3 = g_co[2,0]*B[0]/H[0] + g_co[2,1]*B[1]/H[1] + g_co[2,2]*B[2]/H[2]
    
    D1 = J**-1*H[0]*(A2*B3 - A3*B2)
    D2 = -J**-1*H[1]*(A1*B3 - A3*B1)
    D3 = J**-1*H[2]*(A1*B2 - A2*B1)
    D = [D1, D2, D3]
    
    return D




# ------------------------------------------------------------------------------
# calculation Nabla operators with metric tensor for general curvilinear coordinates
# reference: 
# [1] L. P. Lebedev, et al, 2010, Tensor Analysis with Applications in mechanics, world scientific press.
# [2] V. I. Piercey, 2007, The Lame and metric coefficients for curvilinear coordinates in R^3, university of Arizona, lecture notes.            




def Grad(U, X, coordinate='Cartesian', metric=[[sympy.S(1), 0, 0], [0, sympy.S(1), 0], [0, 0, sympy.S(1)]], evaluation=0):
    '''
    calculate gradient with metric tensor g^{ij}
    chunan@ipp.ac.cn 2020.06.08
    '''
    D1, D2, D3 = sympy.symbols('D1, D2, D3')
    if (coordinate == 'Cartesian') or (coordinate == 'Cylinder') or (coordinate == 'Sphere') or (coordinate == 'Toroidal'):
        metric = Metric(Xi=X, coordinate=coordinate, contra=1)
    else:
        print('Coordinate: '+coordinate+' is used, you should be explictly given at input !')
        metric = metric

    # obtain the Lame coefficients from metric tensor: H = [H1, H2, H3]
    H = Lame(metric=metric, contra=1)
    print('Lame coefficients of '+coordinate+' H: ', H)

    if evaluation == 1:
        D1 = H[0]*(metric[0,0]*U.diff(X[0]) + metric[0,1]*U.diff(X[1]) + metric[0,2]*U.diff(X[2]))
        D2 = H[1]*(metric[1,0]*U.diff(X[0]) + metric[1,1]*U.diff(X[1]) + metric[1,2]*U.diff(X[2]))
        D3 = H[2]*(metric[2,0]*U.diff(X[0]) + metric[2,1]*U.diff(X[1]) + metric[2,2]*U.diff(X[2]))
    else:
        D1 = H[0]*(metric[0,0]*sympy.Derivative(U, X[0]) + metric[0,1]*sympy.Derivative(U, X[1]) + metric[0,2]*sympy.Derivative(U, X[2]))
        D2 = H[1]*(metric[1,0]*sympy.Derivative(U, X[0]) + metric[1,1]*sympy.Derivative(U, X[1]) + metric[1,2]*sympy.Derivative(U, X[2]))
        D3 = H[2]*(metric[2,0]*sympy.Derivative(U, X[0]) + metric[2,1]*sympy.Derivative(U, X[1]) + metric[2,2]*sympy.Derivative(U, X[2]))        
    D = [D1, D2, D3]
    
    return D    




def Div(U, X, coordinate='Cartesian', metric=[[sympy.S(1), 0, 0], [0, sympy.S(1), 0], [0, 0, sympy.S(1)]], evaluation=0):
    '''
    calculate divergence with metric tensor.
    The default input metric tensor is contra-variant metric tensor.
    chunan@ipp.ac.cn 2020.06.08
    '''
    U1 = U[0]
    U2 = U[1]
    U3 = U[2]
    if (coordinate == 'Cartesian') or (coordinate == 'Cylinder') or (coordinate == 'Sphere') or (coordinate == 'Toroidal'):
        metric = Metric(Xi=X, coordinate=coordinate, contra=1)
    else:
        print('Coordinate: '+coordinate+' is used, you should be explictly given at input !')
        metric = metric

    # obtain the Lame coefficients from metric tensor: H = [H1, H2, H3]
    H = Lame(metric=metric, contra=1)
    print('Lame coefficients of '+coordinate+' H: ', H)

    J = Jacobian(metric, contra=1)
    print('contra = 1 , Jacobian = ', J)
    
    if evaluation == 1:
        D = J**-1*((J*U1/H[0]).diff(X[0]) + (J*U2/H[1]).diff(X[1]) + (J*U3/H[2]).diff(X[2]))
    else:
        D = J**-1*(sympy.Derivative(J*U1/H[0], X[0]) + sympy.Derivative(J*U2/H[1], X[1]) + sympy.Derivative(J*U3/H[2], X[2]))
     
    return D




def Curl(U, X, coordinate='Cartesian', metric=[[sympy.S(1), 0, 0], [0, sympy.S(1), 0], [0, 0, sympy.S(1)]], evaluation=0):
    '''
    calculate divergence with metric tensor.
    chunan@ipp.ac.cn 2020.06.08
    '''
    if (coordinate == 'Cartesian') or (coordinate == 'Cylinder') or (coordinate == 'Sphere') or (coordinate == 'Toroidal'):
        metric = Metric(Xi=X, coordinate=coordinate, contra=1)
    else:
        print('Coordinate: '+coordinate+' is used, you should be explictly given at input !')
        metric = metric

    print('contra-variant metric tensor: ', metric)
    H = Lame(metric=metric, contra=1)
    print('Lame coefficients of '+coordinate+' H: ', H)
    J = Jacobian(metric)
    print('Jacobian = ', J)
    
    # get the covariant metric tensor using matrix inverse method
    g_co = metric.inv(method='GE')
    
    # get the normalized cotravariant component of vector fields A and B
    U1 = g_co[0,0]*U[0]/H[0] + g_co[0,1]*U[1]/H[1] + g_co[0,2]*U[2]/H[2]
    U2 = g_co[1,0]*U[0]/H[0] + g_co[1,1]*U[1]/H[1] + g_co[1,2]*U[2]/H[2]
    U3 = g_co[2,0]*U[0]/H[0] + g_co[2,1]*U[1]/H[1] + g_co[2,2]*U[2]/H[2]
    
    if evaluation == 1:
        D1 = (U3.diff(X[1]) -U2.diff(X[2]))*H[0]/J
        D2 = -(U3.diff(X[0]) - U1.diff(X[2]))*H[1]/J
        D3 = (U2.diff(X[0]) - U1.diff(X[1]))*H[2]/J
    else:
        D1 = (sympy.Derivative(U3, X[1]) - sympy.Derivative(U2, X[2]))*H[0]/J
        D2 = -(sympy.Derivative(U3, X[0]) - sympy.Derivative(U1, X[2]))*H[1]/J
        D3 = (sympy.Derivative(U2, X[0]) - sympy.Derivative(U1, X[1]))*H[2]/J
    D = [D1, D2, D3]
    
    return D



