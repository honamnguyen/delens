'''
Following Namikawa's formulation for RD estimator (1703.00169)

Lens function: need noise for phiphi, EE.
Recons function: 
'''

import numpy as np

def RDEstimator_N4(X,Y,Z,W):
    '''
    Eq C5
    Need X012, Y012, Z012, W01
    Y,Z are used to reconstruct phi -> need phi(0,0), phi(0-2,1), phi(1,0-2)
    Return 4-pt bias term before ensemble average
    '''
    X012 = X[0]+X[1]+X[2]
    phi = rPhi(Y[0],Z[0])
    phi021 = rPhi(Y[0]-Y[2],Z[1])
    phi102 = rPhi(Y[1],Z[0]-Z[2])
    W01 = conjugate(W[0]+W[1])

    term1 = lens(X[1],phi)
    term2 = lens(X012,phi021+phi102)

    return -2.*(term1+term2)*W01

    
def RDEstimator_N6_sub(X,Y,Z,W):
    '''
    optimal: Eq C18, C22
    suboptimal: Eq C30, C31
    '''

    X012 = X[0]+X[1]+X[2]
        
    # RD B-mode
    phi = rPhi(Y[0],Z[0])
    phi11 = rPhi(Y[1],Z[1])
    phi33 = rPhi(Y[3],Z[3])
    phi01 = 0.5*( rPhi(Y[0],Z[1]) + rPhi(Y[1],Z[0]) )
    phi02 = 0.5*( rPhi(Y[0],Z[2]) + rPhi(Y[2],Z[0]) )
    phi03 = 0.5*( rPhi(Y[0],Z[3]) + rPhi(Y[3],Z[0]) )
    phi13 = 0.5*( rPhi(Y[1],Z[3]) + rPhi(Y[3],Z[1]) )
    phi23 = 0.5*( rPhi(Y[2],Z[3]) + rPhi(Y[3],Z[2]) )

    if not suboptimal:
        term1 = lens( X012 , 2.*phi - 2.*phi02 - 4.*phi03 + 2.*phi23)
        term2 = -2.*lens( X[3] , phi)
        term3 = 2.*lens( X[0]-X[2] , 2.*phi01 - phi13)
    else: 
        term1 = lens( X012 , 2.*phi - 4.*phi02 - 8.*phi03 + 8.*phi23 + 2.*phi11 )
        term2 = -2.*lens( X[3] , 2.*phi + phi11 + phi33 )
        term3 = 2.*lens( X[0]+X[1]-2.*X[2] , phi01 - phi13)
 
    B0123 = conjugate(term1+term2+term3)

    # N6 term
    term0 = lens( X012 , 2.*phi01 + 2.*phi13 )

    N6 = real(term0*B0123)
    if not suboptimal:
        N6 += lens(X[0]-X[3],phi11)*conjugate(lens(X[0]+X[3],phi11))

    return N6
