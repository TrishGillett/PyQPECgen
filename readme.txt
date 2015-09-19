#######################################################################
## Houyuan Jiang, Daniel Ralph, copyright 1997
## Matlab code accompanied the paper: 
##   Jiang, H., Ralph D. 
##   QPECgen, a MATLAB generator for mathematical programs with 
##   Computational Optimization and Applications 13 (1999), 25–59.
##
## Python implementation coded by Patricia Gillett, 2013
#######################################################################
##
## This code generates random test problems of MPEC with quadratic objective
## functions and affine variational inequality constraints, and certain special
## cases.
##
## The MPEC problem is defined as:
##        #############################################################
##        ##   min   f(x,y)                                          ##
##        ##   s.t.  (x,y) in Z                                      ##
##        ##         y in S(x), S(x) solves AVI(F(x,y), C(x))        ##
##        ##         F(x,y) is linear with respect to both x and y   ##
##        ##         C(x) is polyhedral in y-space                   ## 
##        #############################################################
##
## x:      n dimensional first level variable.
## y:      m dimensional second level variable.
## P:      P=[Px Pxy^T; Pxy Py] -- Hessian of the objective function.
## c, d:   coefficient vectors associated with x and y, respectively.
## A, a:   A is an l by (m+n) matrix, a is an l dimensional vector matrix.
##         Used in the upper level constraints A*[x;y] + a <= 0
##         (* models are described in paper in terms of G and H where A=[G, H])
## F, q, N, M: N is an m by n matrix, M is an m by m matrix, and q is
##             an m by 1 vector.
##             These define F linearly in terms of x and y: F=N*x+M*y+q.
## D, E, b: D is a p by n matrix, E is a p by m matrix, b is an m dimensional
##          vector.  Used in the lower level constraints for type 100 problems.
## u:      m dimensional vector used in lower level constraints for type 200
##         problem.
##
##        ############################################
##        ##           AVI-QPEC (type 100)          ##
##        ############################################
##
##           min   0.5*[x;y]^T*P*[x;y] + [c;d]^T*[x;y]
##           s.t.  A*[x;y] + a <= 0
##                 D*x + E*y + b <= 0
##                 lambda >= 0
##                 (D*x + E*y + b)^Tlambda = 0
##                 N*x + M*y + E^T*lambda + q = 0
##                 
##
##        ############################################
##        ##           BOX-MPEC (type 200)          ##
##        ############################################
##
##        For this case, let y=[y1;y2]　where variables y1 have both upper and
##        lower bounds and y2 variables only have lower bounds.  Because there
##        are no other lower level constraints, the case simplifies and there
##        are no lambda variables.
##
##           min   0.5*[x;y]^T*P*[x;y] + [c;d]^T*[x;y]
##           s.t.  A*[x;y] + a <= 0
##                 0 <= y1 <= u 
##                 0 <= y2
##                 N2*x + M2*y + q2 >= 0    complements    0 <= y2
##                 N1*x + M1*y + q1    complements    0 <= y1 <= u
##                 
## 
##        #####################################################################
##        ##         PATRICIA'S ADDITION: SPECIAL BOX-MPEC (type 201)        ##
##        #####################################################################
##
##        It is convenient for us to have one more type where all second
##        level variables have both lower and upper bounds and all first level
##        variables are bounded above and below as well.
##
##           min   0.5*[x;y]^T*P*[x;y] + [c;d]^T*[x;y]
##           s.t.  A*[x;y] + a <= 0
##                 0 <= x <= ux
##                 0 <= y <= uy
##                 N1*x + M1*y + q1    complements    0 <= y <= uy
##                 
## 
##        ############################################
##        ##          LCP-MPEC type(300)            ##
##        ############################################
##
##           min   0.5*[x;y]^T*P*[x;y] + [c;d]^T*[x;y]
##           s.t.  A*[x;y] + a <= 0
##                 0 <= y
##                 N*x + M*y + q >= 0
##                 (N*x + M*y + q)^Ty = 0
##        
##
##        ############################################
##        ##           GOOD-LCP (type 800)          ##
##        ############################################
##        
##        The objective function is equivalent to sum((x-1)^2) + sum((y+2)^2)
##        shifted by a constant.  It is minimized by the point closest
##        to (1 ... 1,-2 ... -2), which is the origin.
##        
##           min   x^Tx + y^Ty - 2*sum(x) + 4*sum(y)
##           s.t.  x <= y
##                 0 <= y
##                 (y-x)^Ty = 0
##                 
## 
##        ###########################################
##        ##          BAD-LCP type(900)            ##
##        ###########################################
##        
##        This problem has multiple local minima.
##        The objective function is equivalent to sum((x+1)^2) + sum((y-2)^2)
##        shifted by a constant.  It is minimized by the feasible point closest
##        to (-1 ... -1, 2 ... 2), which is (-1 ... -1, 0 ... 0).
##        
##           min   x^Tx + y^Ty + 2*sum(x) - 4*sum(y)
##           s.t.  x <= y
##                 0 <= y
##                 (y-x)^Ty = 0
"""