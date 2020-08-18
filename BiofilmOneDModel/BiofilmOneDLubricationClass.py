"""
A helper class for solving the non-linear time dependent equations 
of biofilm growth which includes models of the cell concentration 
and also nutrient concentrations in both the substrate and biofilm.
All of these are assumed to be radially symmetric and depend on only 
r and t. The specific equations solved by this class are described in the
publication:

A Thin-Film Lubrication Model for Biofilm Expansion Under Strong Adhesion,
A. Tam, B. Harding, J.E.F. Green, S. Balasuriya, and B.J. Binder
To be submitted soon, 2020.

This works builds upon the model developed by Alex Tam in his PhD thesis:

Mathematical Modelling of Pattern Formation in Yeast Biofilms, 
Alex Tam,
The University of Adelaide, 2019.

Two solvers are currently implemented within the class.
The first, a "decoupled" Crank-Nicolson implementation, denoted DCN, 
solves the non-linear system of equations in a weakly coupled manner. 
Each equation is solved one at a time (via Newton iterations where 
applicable) using the last known/computed solution of any other variables.
The second, a fully coupled Crank-Nicolson implementation, denoted FCN,
solves the complete non-linear system of equations using Newton iterations.
Both use the scipy sparse LU solver to solve the discretised systems 
of equations that result from a compact finite difference discretisation.
Both methods can be expected to achieve 2nd order convergence 
in both space and time.

Compatibility notes: 
The code was written in Python3 (3.7.3 specifically) although it 
should also work in 2.7.x releases that are not to old as well.
The scientific computing packages numpy and scipy are required.
Again, any version that is not unreasonably old should be fine.
You will probably also want matplotlib for plotting.

Maintainer: Brendan Harding 
Initial development: Jan-May 2020
Last updated: August 2020
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import diags,bmat

class BiofilmOneDLubricationModel(object):
    """
    Helper class for solving the PDEs describing the development of 
    a radially symmetric and thin yeast biofilm over time.
    The model/system that is solved includes the biofilm height,
    the cell concentration, and the nutrient concentrations in both
    the biofilm and the substrate.
    """
    def __init__(self,R=2.0,dr=0.5**7,dt=None,params=None,solver='DCN',verbose=False):
        """
        Initialise the class
        
        With no arguments a default problem set up is initialised.
        
        Optionally you may pass the following:
        R: The radius of the domain (or petri dish). If not specified
        a default value of 2 is used.
        dr: The grid spacing used for the discretisation of the domain.
        If not specified a default value of 0.5**7 is used.
        dt: The time step size, if not specified 0.25*dr is used.
        params: Parameters for the system of equations. These should 
        be passed as a dictionary. Any which are not specified will
        be set to a default value (specifically corresponding to 
        Table 6.1 in Alex's thesis).
        solver: specify which solver to use.
        verbose: Set to True to output convergence information when solving
        """
        # Set up the radial coordinate array
        self._r = np.arange(0.0,R+0.5*dr,dr)
        self._r_half = self._r[:-1]+0.5*dr
        # Set up the parameters
        if dt is None:
            self._dt = 0.25*dr # this is quite conservative...
        else:
            self._dt = dt
        if type(params)==dict:
            # Set various parameters depending what is passed,
            # those not specified will be set to those Alex used 
            # in his thesis (Table 6.1 specifically)
            self._b = params.get("b",0.0001)
            self._H0 = params.get("H0",0.1)
            self._Psi_m = params.get("Psi_m",0.111)
            self._Psi_d = params.get("Psi_d",0.0)
            #self._R = params.get("R",10.0)
            #self._T = params.get("T",50.0)
            self._gamma_ast = params.get("gamma_ast",1.0)
            self._D  = params.get("D",1.05)
            self._Pe = params.get("Pe",3.94)
            self._Upsilon = params.get("Upsilon",3.15)
            self._Q_b = params.get("Q_b",8.65)
            self._Q_s = params.get("Q_s",2.09)
            self._h_ast = params.get("h_ast",0.002)
            self._lambda_ast = params.get("lambda_ast",np.inf)
        else:
            if params is not None:
                print("Setting parameters is currently only supported through a dictionary, default values will be used")
            # Set various parameters to those Alex used in 
            # his thesis (Table 6.1 specifically)
            self._b  = 0.0001
            self._H0 = 0.1
            self._Psi_m = 0.111
            self._Psi_d = 0.0
            #self._R = 10.0
            #self._T = 50.0 
            self._gamma_ast = 1.0
            self._D  = 1.05
            self._Pe = 3.94
            self._Upsilon = 3.15
            self._Q_b = 8.65
            self._Q_s = 2.09
            self._h_ast = 0.002
            self._lambda_ast = np.inf
        self.set_solver(solver)
        self._verbose = verbose
        # Set up the solution arrays with default initial conditions
        self._set_default_initial_conditions()
        # Anything else???
        
        # done
    def _set_default_initial_conditions(self):
        """
        Sets the initial conditions to be those described by 
        equation 6.22 of Alex Tam's thesis.
        """
        self._t = 0
        r = self._r
        self._h = self._b + (self._H0-self._b)*(r<1)*(1-r**2)**4
        self._Phi_n = (r<1)*(1-3*r**2+2*r**3)*self._h
        self._g_s = np.ones(len(self._r))
        self._g_b = np.zeros(len(self._r))
        # done
    # add getters and setters
    def set_parameters(self,params):
        """
        Set the current problem parameters.
        Parameters should be passed using a dictionary.
        """
        if type(params)==dict:
            # Set various parameters depending what is passed,
            # those not specified will be set to those Alex used 
            # in his thesis (Table 6.1 specifically)
            self._b = params.get("b",self._b)
            self._H0 = params.get("H0",self._H0)
            self._Psi_m = params.get("Psi_m",self._Psi_m)
            self._Psi_d = params.get("Psi_d",self._Psi_d)
            #self._R = params.get("R",self._R)
            #self._T = params.get("T",self._T)
            self._gamma_ast = params.get("gamma_ast",self._gamma_ast)
            self._D  = params.get("D",self._D )
            self._Pe = params.get("Pe",self._Pe)
            self._Upsilon = params.get("Upsilon",self._Upsilon)
            self._Q_b = params.get("Q_b",self._Q_b)
            self._Q_s = params.get("Q_s",self._Q_s)
            self._h_ast = params.get("h_ast",self._h_ast)
            self._lambda_ast = params.get("lambda_ast",self._lambda_ast)
        else:
            print("Setting parameters is currently only supported through a dictionary, existing values will be used")
        # done
    def get_parameters(self,param=None):
        """
        Get the current problem parameters.
        If a specific parameter is not requested 
        then all are returned in a dictionary.
        """
        params_dict = {"b":self._b,"H0":self._H0,"Psi_m":self._Psi_m,"Psi_d":self._Psi_d,\
                       "gamma_ast":self._gamma_ast,"D":self._D,"Pe":self._Pe,\
                       "Upsilon":self._Upsilon,"Q_b":self._Q_b,"Q_s":self._Q_s,\
                       "h_ast":self._h_ast,"lambda_ast":self._lambda_ast}
        #params_dict["R"] = self._R
        #params_dict["T"] = self._T
        if param is None:
            # return dictionary with all parameters
            return params_dict
        elif param in params_dict.keys():
            return params_dict[param]
        else:
            print("Requested parameter does not exist")
        # done
    def get_r(self):
        """
        Returns the array for the radial coordinates.
        """
        return self._r
    def set_verbosity(self,verbose):
        """
        Set the verbosity for the solvers (True or False).
        """
        self._verbose = verbose
        # done
    def set_h(self,h):
        """
        Update the biofilm height h. 
        For example, use this to set the initial condition.
        (Note this over-writes the current solution in the class.)
        Accepts a callable function h(r), or an array (with correct length).
        Note: This will not alter Phi_n=phi_n*h, if it is desired that this 
        too be changed it should be done separately via set_Phi_n or set_phi_n.
        """
        if callable(h):
            self._h[:] = h(self._r)
        else:
            assert len(h)==len(self._r)
            self._h[:] = h
        # done
    def get_h(self):
        """
        Returns the current biofilm height h.
        """
        return self._h
    def set_Phi_n(self,Phi_n):
        """
        Update the cumulative cell volume fraction Phi_n (=phi_n*h). 
        For example, use this to set the initial condition.
        (Note this over-writes the current solution in the class.)
        Accepts a callable function phi_n(r), or an array (with correct length).
        """
        if callable(Phi_n):
            self._Phi_n[:] = Phi_n(self._r)
        else:
            assert len(Phi_n)==len(self._r)
            self._Phi_n[:] = Phi_n
        # done
    def get_Phi_n(self):
        """
        Returns the current cumulative cell volume fraction Phi_n (=phi_n*h). 
        """
        return self._Phi_n
    def set_phi_n(self,phi_n):
        """
        Update the cell volume fraction phi_n. 
        For example, use this to set the initial condition.
        (Note this over-writes the current solution in the class.)
        Accepts a callable function phi_n(r), or an array (with correct length).
        Note: This internally updates Phi_n=phi_n*h using the existing h.
        If h is also to be updated, it should be done first!
        """
        if callable(phi_n):
            self._Phi_n[:] = phi_n(self._r)*self._h
        else:
            assert len(phi_n)==len(self._r)
            self._Phi_n[:] = phi_n*self._h
        self._Phi_n[self._h==self._b] = 0
        # done
    def get_phi_n(self):
        """
        Returns the current cell volume fraction phi_n. 
        """
        return self._Phi_n/self._h
    def set_g_s(self,g_s):
        """
        Update the substrate nutrient concentration g_s. 
        For example, use this to set the initial condition.
        (Note this over-writes the current solution in the class)
        Accepts a callable function g_s(r), or an array (with correct length).
        """
        if callable(g_s):
            self._g_s[:] = g_s(self._r)
        else:
            assert len(g_s)==len(self._r)
            self._g_s[:] = g_s
        # done
    def get_g_s(self):
        """
        Returns the substrate nutrient concentration g_s.
        """
        return self._g_s
    def set_g_b(self,g_b):
        """
        Update the biofilm nutrient concentration g_b. 
        For example, use this to set the initial condition.
        (Note this over-writes the current solution in the class)
        Accepts a callable function g_b(r), or an array (with correct length).
        """
        if callable(g_b):
            self._g_b[:] = g_b(self._r)
        else:
            assert len(g_b)==self._nr
            self._g_b[:] = g_b
        # done
    def get_g_b(self):
        """
        Returns the biofilm nutrient concentration g_b.
        """
        return self._g_b
    def set_dt(self,dt):
        """
        Set/change the time step size (dt) which is used by default
        (i.e. if dt is not specified when solve is called then this value is used)
        """
        self._dt = dt
        # done
    def get_dt(self):
        """
        Get the current time step size (dt) which is used by default
        (i.e. if dt is not specified when solve is called then this value is used)
        """
        return self._dt
    def set_t(self,t):
        """
        Set/change the current solution time t.
        """
        self._t = t
        # done
    def get_t(self):
        """
        Get the current solution time T.
        """
        return self._t
        # done
    # Add private methods relating to the discretisation of the fourth order 'advective' term
    def _advective_term(self,r,h,p=3,f=None,near_boundary=True,prefix=None):
        """
        Finite difference discretisation of:
        prefix * (d/dr)[ r h^p f (d/dr)[ (1/r) (d/dr)[ r (dh/dr) ] ] ]
        """
        r_half = 0.5*(r[1:]+r[:-1])
        dr = r[1]-r[0]
        h_half = 0.5*(h[1:]+h[:-1])
        D_half =  (r_half[2:  ]*(h[3:  ]-h[2:-1])-r_half[1:-1]*(h[2:-1]-h[1:-2]))/r[2:-1]\
                 -(r_half[1:-1]*(h[2:-1]-h[1:-2])-r_half[ :-2]*(h[1:-2]-h[ :-3]))/r[1:-2]
        if f is None:
            f = np.ones(len(r))
            f_half = f[:-1]
        else:
            f_half = 0.5*(f[1:]+f[:-1])
        res = np.empty(len(r))
        res[[0,1,-2,-1]] = 0
        res[2:-2] =  r_half[2:-1]*h_half[2:-1]**p*f_half[2:-1]*D_half[1:] \
                    -r_half[1:-2]*h_half[1:-2]**p*f_half[1:-2]*D_half[:-1]
        if near_boundary:
            # At index one we expoloit that 0 = (d/dr)[ (1/r) (d/dr)[ r (dh/dr) ] ] for r=0
            D_p2 = 0.5*(D_half[ 0]+D_half[ 1])
            res[1] = 0.5*r[2]*h[2]**p*f[2]*D_p2
            # At index -2 we can exploit that 0 = (dh/dr)
            # The width of the stencil is widened to achieve this though...
            D_m5o2 =  0.25*(r[-2]*(h[-1]-h[-3])-r[-4]*(h[-3]-h[-5]))/r[-3]\
                     -0.25*(r[-3]*(h[-2]-h[-4])-r[-5]*(h[-4]-h[-6]))/r[-4]
            D_m3o2 =  0.25*(                   -r[-3]*(h[-2]-h[-4]))/r[-2]\
                     -0.25*(r[-2]*(h[-1]-h[-3])-r[-4]*(h[-3]-h[-5]))/r[-3]
            res[-2] =  r_half[-1]*h_half[-1]**p*f_half[-1]*D_m3o2 \
                      -r_half[-2]*h_half[-2]**p*f_half[-2]*D_m5o2
        if prefix is not None:
            res[1:-1] *= prefix[1:-1]
        return res/dr**4
    def _advective_term_h_gradient(self,r,h,p=3,f=None,near_boundary=True,prefix=None):
        """
        Finite difference discretisation of the gradient of
        prefix * (d/dr)[ r h^p f (d/dr)[ (1/r) (d/dr)[ r (dh/dr) ] ] ]
        with respect to h.
        
        Note: the caller is responsible for enforcing boundary conditions
        """
        r_half = 0.5*(r[1:]+r[:-1])
        dr = r[1]-r[0]
        h_half = 0.5*(h[1:]+h[:-1])
        D_half =  (r_half[2:  ]*(h[3:  ]-h[2:-1])-r_half[1:-1]*(h[2:-1]-h[1:-2]))/r[2:-1]\
                 -(r_half[1:-1]*(h[2:-1]-h[1:-2])-r_half[ :-2]*(h[1:-2]-h[ :-3]))/r[1:-2]
        if f is None:
            f = np.ones(len(r))
            f_half = f[:-1]
        else:
            f_half = 0.5*(f[1:]+f[:-1])
        Dh_diag_p2 = np.empty((len(r)))
        Dh_diag_p1 = np.empty((len(r)))
        Dh_diag_p0 = np.empty((len(r)))
        Dh_diag_m1 = np.empty((len(r)))
        Dh_diag_m2 = np.empty((len(r)))
        Dh_diag_p2[[0,1,-2,-1]] = 0
        Dh_diag_p1[[0,1,-2,-1]] = 0
        Dh_diag_p0[[0,1,-2,-1]] = 0
        Dh_diag_m1[[0,1,-2,-1]] = 0
        Dh_diag_m2[[0,1,-2,-1]] = 0
        Dh_diag_p1[2:-2] =  r_half[2:-1]*0.5*p*h_half[2:-1]**(p-1)*f_half[2:-1]*D_half[1:]/dr**4
        Dh_diag_p0[2:-2] =  r_half[2:-1]*0.5*p*h_half[2:-1]**(p-1)*f_half[2:-1]*D_half[1:]/dr**4 \
                           -r_half[1:-2]*0.5*p*h_half[1:-2]**(p-1)*f_half[1:-2]*D_half[:-1]/dr**4
        Dh_diag_m1[2:-2] = -r_half[1:-2]*0.5*p*h_half[1:-2]**(p-1)*f_half[1:-2]*D_half[:-1]/dr**4
        # I think the following 5 are okay...
        Dh_diag_p2[2:-2]  =  r_half[2:-1]*h_half[2:-1]**p*f_half[2:-1]*(r_half[3:  ]/r[3:-1])/dr**4
        Dh_diag_p1[2:-2] += -r_half[2:-1]*h_half[2:-1]**p*f_half[2:-1]*(r_half[2:-1]/r[2:-2]+2)/dr**4 \
                            -r_half[1:-2]*h_half[1:-2]**p*f_half[1:-2]*(r_half[2:-1]/r[2:-2])/dr**4  
        Dh_diag_p0[2:-2] +=  r_half[2:-1]*h_half[2:-1]**p*f_half[2:-1]*(r_half[2:-1]/r[3:-1]+2)/dr**4 \
                            +r_half[1:-2]*h_half[1:-2]**p*f_half[1:-2]*(r_half[1:-2]/r[1:-3]+2)/dr**4  
        Dh_diag_m1[2:-2] += -r_half[2:-1]*h_half[2:-1]**p*f_half[2:-1]*(r_half[1:-2]/r[2:-2])/dr**4 \
                            -r_half[1:-2]*h_half[1:-2]**p*f_half[1:-2]*(r_half[1:-2]/r[2:-2]+2)/dr**4 
        Dh_diag_m2[2:-2]  =  r_half[1:-2]*h_half[1:-2]**p*f_half[1:-2]*(r_half[ :-3]/r[1:-3])/dr**4
        if near_boundary:
            # Pre-allocate additional diagonals for the boundary terms
            Dh_diag_p3 = np.zeros((len(r)))
            Dh_diag_m3 = np.zeros((len(r)))
            Dh_diag_m4 = np.zeros((len(r)))
            # At index one we expoloit that 0 = (d/dr)[ (1/r) (d/dr)[ r (dh/dr) ] ]
            D_p2 = 0.5*(D_half[ 0]+D_half[ 1])
            Dh_diag_p1[1]  =  0.5*r[2]*p*h[2]**(p-1)*f[2]*D_p2/dr**4
            Dh_diag_p3[1]  =  0.5*r[2]*h[2]**p*f[2]*0.5*(r_half[3]/r[3])/dr**4
            Dh_diag_p2[1]  = -0.5*r[2]*h[2]**p*f[2]/dr**4
            Dh_diag_p1[1] +=  0.5*r[2]*h[2]**p*f[2]*0.5*(r_half[2]/r[3]-r_half[1]/r[1])/dr**4  
            Dh_diag_p0[1]  = -0.5*r[2]*h[2]**p*f[2]/dr**4  
            Dh_diag_m1[1]  =  0.5*r[2]*h[2]**p*f[2]*0.5*(r_half[0]/r[1])/dr**4
            # At index -2 we can exploit that 0 = (dh/dr)
            # The width of the stencil is widened to achieve this though...
            D_m5o2 =  0.25*(r[-2]*(h[-1]-h[-3])-r[-4]*(h[-3]-h[-5]))/r[-3]\
                     -0.25*(r[-3]*(h[-2]-h[-4])-r[-5]*(h[-4]-h[-6]))/r[-4]
            D_m3o2 =  0.25*(                   -r[-3]*(h[-2]-h[-4]))/r[-2]\
                     -0.25*(r[-2]*(h[-1]-h[-3])-r[-4]*(h[-3]-h[-5]))/r[-3]
            Dh_diag_p1[-2] =  r_half[-1]*0.5*p*h_half[-1]**(p-1)*f_half[-1]*D_m3o2 
            Dh_diag_p0[-2] =  r_half[-1]*0.5*p*h_half[-1]**(p-1)*f_half[-1]*D_m3o2 \
                             -r_half[-2]*0.5*p*h_half[-2]**(p-1)*f_half[-2]*D_m5o2
            Dh_diag_m1[-2] = -r_half[-2]*0.5*p*h_half[-2]**(p-1)*f_half[-2]*D_m5o2
            # I think the following are okay...
            Dh_diag_p1[-2] +=  r_half[-1]*h_half[-1]**p*f_half[-1]*( r[-2]/r[-3])*0.25/dr**4 \
                              -r_half[-2]*h_half[-2]**p*f_half[-2]*(-r[-2]/r[-3])*0.25/dr**4
            Dh_diag_p0[-2] +=  r_half[-1]*h_half[-1]**p*f_half[-1]*(-r[-3]/r[-4])*0.25/dr**4 \
                              -r_half[-2]*h_half[-2]**p*f_half[-2]*(-r[-3]/r[-2])*0.25/dr**4
            Dh_diag_m1[-2] +=  r_half[-1]*h_half[-1]**p*f_half[-1]*(-2)*0.25/dr**4 \
                              -r_half[-2]*h_half[-2]**p*f_half[-2]*( 2)*0.25/dr**4
            Dh_diag_m2[-2]  =  r_half[-1]*h_half[-1]**p*f_half[-1]*( 2)*0.25/dr**4 \
                              -r_half[-2]*h_half[-2]**p*f_half[-2]*( r[-3]/r[-2])*0.25/dr**4
            Dh_diag_m3[-2]  =  r_half[-1]*h_half[-1]**p*f_half[-1]*( r[-4]/r[-3])*0.25/dr**4 \
                              -r_half[-2]*h_half[-2]**p*f_half[-2]*( r[-4]/r[-3])*0.25/dr**4
            Dh_diag_m4[-2]  =  r_half[-1]*h_half[-1]**p*f_half[-1]*(-r[-5]/r[-4])*0.25/dr**4
            #Dh = diags([Dh_diag_m4[4:],Dh_diag_m3[3:],Dh_diag_m2[2:],Dh_diag_m1[1:],\
            #            Dh_diag_p0,Dh_diag_p1[:-1],Dh_diag_p2[:-2],Dh_diag_p3[:-3]],\
            #           [-4,-3,-2,-1,0,1,2,3])
            diagonals = [Dh_diag_m4,Dh_diag_m3,Dh_diag_m2,Dh_diag_m1,\
                         Dh_diag_p0,Dh_diag_p1,Dh_diag_p2,Dh_diag_p3]
            offsets = [-4,-3,-2,-1,0,1,2,3]
        else:
            #Dh = diags([Dh_diag_m2[2:],Dh_diag_m1[1:],Dh_diag_p0,Dh_diag_p1[:-1],Dh_diag_p2[:-2]],\
            #           [-2,-1,0,1,2])
            diagonals = [Dh_diag_m2,Dh_diag_m1,Dh_diag_p0,Dh_diag_p1,Dh_diag_p2]
            offsets = [-2,-1,0,1,2]
        if prefix is not None:
            for diagonal in diagonals:
                diagonal[1:-1] *= prefix[1:-1]
        return diagonals,offsets
    def _advective_term_f_gradient(self,r,h,p=3,f=None,near_boundary=True,prefix=None):
        """
        Finite difference discretisation of the gradient of
        prefix * (d/dr)[ r h^p f (d/dr)[ (1/r) (d/dr)[ r (dh/dr) ] ] ]
        with respect to f.
        """
        if f is None:
            return None
        r_half = 0.5*(r[1:]+r[:-1])
        dr = r[1]-r[0]
        h_half = 0.5*(h[1:]+h[:-1])
        D_half =  (r_half[2:  ]*(h[3:  ]-h[2:-1])-r_half[1:-1]*(h[2:-1]-h[1:-2]))/r[2:-1]\
                 -(r_half[1:-1]*(h[2:-1]-h[1:-2])-r_half[ :-2]*(h[1:-2]-h[ :-3]))/r[1:-2]
        #f_half = 0.5*(f[1:]+f[:-1])
        Df_diag_p1 = np.empty((len(r)))
        Df_diag_p0 = np.empty((len(r)))
        Df_diag_m1 = np.empty((len(r)))
        Df_diag_p1[[0,1,-2,-1]] = 0
        Df_diag_p0[[0,1,-2,-1]] = 0
        Df_diag_m1[[0,1,-2,-1]] = 0
        Df_diag_p1[2:-2] =  r_half[2:-1]*h_half[2:-1]**p*0.5*D_half[1:]/dr**4
        Df_diag_p0[2:-2] =  r_half[2:-1]*h_half[2:-1]**p*0.5*D_half[1:]/dr**4 \
                           -r_half[1:-2]*h_half[1:-2]**p*0.5*D_half[:-1]/dr**4
        Df_diag_m1[2:-2] = -r_half[1:-2]*h_half[1:-2]**p*0.5*D_half[:-1]/dr**4
        if near_boundary:
            D_p2 = 0.5*(D_half[ 0]+D_half[ 1])
            Df_diag_p1[1] = 0.5*r[2]*h[2]**p*D_p2/dr**4
            D_m5o2 =  0.25*(r[-2]*(h[-1]-h[-3])-r[-4]*(h[-3]-h[-5]))/r[-3]\
                     -0.25*(r[-3]*(h[-2]-h[-4])-r[-5]*(h[-4]-h[-6]))/r[-4]
            D_m3o2 =  0.25*(                   -r[-3]*(h[-2]-h[-4]))/r[-2]\
                     -0.25*(r[-2]*(h[-1]-h[-3])-r[-4]*(h[-3]-h[-5]))/r[-3]
            Df_diag_p1[-2] =  r_half[-1]*h_half[-1]**p*D_m3o2/dr**4
            Df_diag_p0[-2] = -r_half[-2]*h_half[-2]**p*D_m5o2/dr**4
        #Df = diags([Df_diag_m1[1:],Df_diag_p0,Df_diag_p1[:-1]],[-1,0,1])#,format="csr")
        diagonals = [Df_diag_m1,Df_diag_p0,Df_diag_p1]
        offsets = [-1,0,1]
        if prefix is not None:
            for diagonal in diagonals:
                diagonal[1:-1] *= prefix[1:-1]
        return diagonals,offsets
    # Add 'private' methods related to the solvers
    def _h_equation_RHS(self,v_old,v_new,dt=None):
        """
        Calculate the RHS vector component corresponding to the height equation.
        The internal time step dt is used if one is not provided.
        """
        r = self._r    
        nr = len(r)
        dr = r[1]
        b = self._b
        h_ast = self._h_ast
        g_ast = self._gamma_ast
        Psi_m = self._Psi_m
        lambda_ast = self._lambda_ast
        h_old,Phi_n_old,g_s_old,g_b_old = v_old
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        # Initialise rhs vector
        rhs = np.empty(nr)
        rhs[2:-2] = -(h_new[2:-2]-h_old[2:-2])
        # Calculate spatial stencil and add to the rhs
        adv_old = self._advective_term(r,h_old,near_boundary=False)
        adv_new = self._advective_term(r,h_new,near_boundary=False)
        rhs[2:-2] -= 0.5*dt*g_ast/3.0*(adv_old[2:-2]+adv_new[2:-2])/r[2:-2]
        if np.isfinite(lambda_ast): # add slip term if lambda_ast is finite
            adv_old = self._advective_term(r,h_old,p=2,near_boundary=False)
            adv_new = self._advective_term(r,h_new,p=2,near_boundary=False)
            rhs[2:-2] -= 0.5*dt*g_ast/lambda_ast*(adv_old[2:-2]+adv_new[2:-2])/r[2:-2]
        # Add the forcing term
        forcing_old = (h_old>h_ast)*(1.0+Psi_m)*Phi_n_old*g_b_old
        forcing_new = (h_new>h_ast)*(1.0+Psi_m)*Phi_n_new*g_b_new
        rhs[2:-2] += 0.5*dt*(forcing_old[2:-2]+forcing_new[2:-2])
        # Set RHS entries relating to boundary conditions
        rhs[ 0] =  3.0*h_new[ 0]- 4.0*h_new[ 1]+     h_new[ 2]
        rhs[ 1] =  5.0*h_new[ 0]-18.0*h_new[ 1]+24.0*h_new[ 2]-14.0*h_new[ 3]+3.0*h_new[ 4]
        rhs[-2] = -3.0*h_new[-1]+ 4.0*h_new[-2]-     h_new[-3]
        rhs[-1] =  b-h_new[-1] 
        # done
        return rhs
    def _h_equation_LHS0(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        h dependence in the height equation.
        The internal time step dt is used if one is not provided.
        """
        r = self._r
        nr = len(r)
        dr = r[1]
        r_half = self._r_half
        g_ast = self._gamma_ast
        h_ast = self._h_ast
        Psi_m = self._Psi_m
        lambda_ast = self._lambda_ast
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        # Construct/fetch the diagonal components from the gradient of the fourth order "advective" term
        diagonals,offsets = self._advective_term_h_gradient(r,h_new,near_boundary=False)
        for i in range(len(diagonals)):
            assert offsets[i]==i-2 # sanity check
            diagonals[i][2:-2] *= (0.5*dt*g_ast/3.0)*r[2:-2]**(-1)
        if np.isfinite(lambda_ast): # add slip term if lambda_ast is finite
            diagonals2,offsets2 = self._advective_term_h_gradient(r,h_new,p=2,near_boundary=False)
            for i in range(len(diagonals2)):
                assert offsets2[i]==offsets[i]
                diagonals[i][2:-2] += (0.5*dt*g_ast/lambda_ast)*r[2:-2]**(-1)*diagonals2[i][2:-2]
        # Add to the main diagonal
        diagonals[2][2:-2] += 1.0
        # Note: there is no longer a 'forcing term' since h is absorbed into Phi_n
        # Enforce the boundary conditions
        diagonals.append(np.zeros(nr))
        offsets.append(3)
        diagonals[2][ 0] = -3 # first order BC at r=0
        diagonals[3][ 0] =  4
        diagonals[4][ 0] = -1
        diagonals[1][ 1] = -5  # third order BC at r=0
        diagonals[2][ 1] =  18
        diagonals[3][ 1] = -24
        diagonals[4][ 1] =  14
        diagonals[5][ 1] = -3
        diagonals[1][-2] =  1  # first order BC at r=R
        diagonals[2][-2] = -4
        diagonals[3][-2] =  3
        diagonals[2][-1] =  1  # Dirichlet BC at r=R
        # Final construction
        A_00 = diags([diagonals[0][2:],diagonals[1][1:],diagonals[2],diagonals[3][:-1],\
                      diagonals[4][:-2],diagonals[5][:-3]],\
                      offsets)#,format="csr")
        return A_00
    def _h_equation_LHS1(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        Phi_n dependence in the height equation (Phi_n = phi_n*h).
        The internal time step dt is used if one is not provided.
        """
        h_ast = self._h_ast
        Psi_m = self._Psi_m
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        A_diag = -0.5*dt*(1.0+Psi_m)*(h_new>h_ast)*g_b_new
        A_diag[[0,1,-2,-1]] = 0
        return diags(A_diag)#,format="csr")
    def _h_equation_LHS2(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        g_s dependence in the height equation.
        The internal time step dt is used if one is not provided.
        """
        # Note: there is no g_s dependence
        return None  
    def _h_equation_LHS3(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        g_b dependence in the height equation.
        The internal time step dt is used if one is not provided.
        """
        h_ast = self._h_ast
        Psi_m = self._Psi_m
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        A_diag = -0.5*dt*(1.0+Psi_m)*(h_new>h_ast)*Phi_n_new
        A_diag[[0,1,-2,-1]] = 0
        return diags(A_diag)#,format="csr")
    def _Phi_n_equation_RHS(self,v_old,v_new,dt=None):
        """
        Calculate the RHS vector component corresponding to the Phi_n equation.
        (Here Phi_n = phi_n*h, the input v_old,v_new must contain Phi_n rather than phi_n)
        The internal time step dt is used if one is not provided.
        """
        r = self._r
        nr = len(r)
        dr = r[1]
        g_ast = self._gamma_ast
        Psi_d = self._Psi_d
        lambda_ast = self._lambda_ast
        h_old,Phi_n_old,g_s_old,g_b_old = v_old
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        # Initialise the rhs term with the d/dt part
        rhs = -(Phi_n_new-Phi_n_old)
        # Get the discretisation of the advective term
        prefix = np.empty(nr)
        prefix[[0,-1]] = 0
        prefix[1:-1] = (-0.5*dt*g_ast/3.0)/r[1:-1]
        adv_old = self._advective_term(r,h_old,2,Phi_n_old,True,prefix)
        adv_new = self._advective_term(r,h_new,2,Phi_n_new,True,prefix)
        rhs += (adv_old+adv_new)
        if np.isfinite(lambda_ast):
            prefix[1:-1] *= 3.0/lambda_ast
            adv_old = self._advective_term(r,h_old,1,Phi_n_old,True,prefix)
            adv_new = self._advective_term(r,h_new,1,Phi_n_new,True,prefix)
            rhs += (adv_old+adv_new)
        # Add the forcing term (note Phi_n includes the h factor)
        rhs[1:-1] += 0.5*dt*( Phi_n_old*(g_b_old-Psi_d)
                             +Phi_n_new*(g_b_new-Psi_d))[1:-1]
        # Set RHS entry relating to boundary condition at r=0
        rhs[ 0] = -3.0*Phi_n_new[0]+4.0*Phi_n_new[1]-Phi_n_new[2] # (d/dr)Phi_n=0
        if False: # higher order stencil
            rhs[ 0] = -11*Phi_n_new[0]+18*Phi_n_new[1]-9*Phi_n_new[2]+2*Phi_n_new[3]
        if True:
            # additing this 'artificial' BC is the easiest way to fix the lhs error
            rhs[ 1] = 4*Phi_n_new[0]-7*Phi_n_new[1]+4*Phi_n_new[2]-Phi_n_new[3]
        # Technically need to use a one sided stencil at r=r_max
        # However, since h=b~0 is enforced, I will just enforce Phi_n=0
        rhs[-1] = 0
        # done
        return rhs
    def _Phi_n_equation_LHS0(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        h dependence in the Phi_n equation.
        (Here Phi_n = phi_n*h, the input v_old,v_new must contain Phi_n rather than phi_n)
        """
        r = self._r
        nr = len(r)
        g_ast = self._gamma_ast
        lambda_ast = self._lambda_ast
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        # Get the diagonals and offsets for the advective term
        prefix = np.empty(nr)
        prefix[[0,-1]] = 0
        prefix[1:-1] = (0.5*dt*g_ast/3.0)*r[1:-1]**(-1)
        if True:
            prefix[1] = 0 # Need to add this in conjunction with the 'articicial' BC
        diagonals,offsets = self._advective_term_h_gradient(r,h_new,2,Phi_n_new,True,prefix)
        if np.isfinite(lambda_ast): # add slip terms
            prefix[1:-1] *= 3.0/lambda_ast
            diagonals2,offsets2 = self._advective_term_h_gradient(r,h_new,1,Phi_n_new,True,prefix)
            for i in range(len(diagonals2)):
                assert offsets2[i]==offsets[i]
                diagonals[i][1:-1] += diagonals2[i][1:-1]
        # Note: there is nothing to add or modify in this case.
        # Final construction
        A_10 = diags([diagonals[0][4:],diagonals[1][3:],diagonals[2][2:],diagonals[3][1:],
                      diagonals[4],diagonals[5][:-1],diagonals[6][:-2],diagonals[7][:-3]],\
                     offsets)#,format="csr")
        return A_10
    def _Phi_n_equation_LHS1(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        phi_n dependence in the Phi_n equation.
        (Here Phi_n = phi_n*h, the input v_old,v_new must contain Phi_n in place of phi_n)
        """
        r = self._r
        r_half = self._r_half
        nr = len(r)
        dr = r[1]
        g_ast = self._gamma_ast
        Psi_d = self._Psi_d
        lambda_ast = self._lambda_ast
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        # Get the diagonals for the gradient of the advective term
        prefix = np.empty(nr)
        prefix[[0,-1]] = 0
        prefix[1:-1] = (0.5*dt*g_ast/3.0)/r[1:-1]
        diagonals,offsets = self._advective_term_f_gradient(r,h_new,2,Phi_n_new,True,prefix)
        assert offsets[1]==0 # sanity check
        if np.isfinite(lambda_ast): # add slip terms
            prefix[1:-1] *= 3.0/lambda_ast
            diagonals2,offsets2 = self._advective_term_f_gradient(r,h_new,1,Phi_n_new,True,prefix)
            for i in range(len(diagonals2)):
                assert offsets2[i]==offsets[i]
                diagonals[i][1:-1] += diagonals2[i][1:-1]
        # Now add the additional main diagonal bit (including the forcing component)
        diagonals[1][1:-1] += 1
        diagonals[1][1:-1] -= 0.5*dt*(g_b_new[1:-1]-Psi_d)
        # Dirichlet BC (avoid using a one sided stencil here, just set Phi_n=0 since h=b~0 is enforced)
        diagonals[1][-1] = 1
        # Allocate an additional diagonal for Neumann BC enforcement
        diagonals.append(np.zeros(nr))
        offsets.append(2)
        # Neumann BC
        diagonals[1][ 0] =  3  
        diagonals[2][ 0] = -4
        diagonals[3][ 0] =  1
        if True:
            # additing this 'artificial' BC is the easiest way to fix the lhs error
            diagonals[0][ 1] = -4  
            diagonals[1][ 1] =  7  
            diagonals[2][ 1] = -4
            diagonals[3][ 1] =  1
        # Final construction 
        A_11 = diags([diagonals[0][1:],diagonals[1],diagonals[2][:-1],diagonals[3][:-2]],\
                     offsets)#,format="csr")
        if False:
            diagonals.append(np.zeros(nr))
            offsets.append(3)
            diagonals[1][ 0] =  11
            diagonals[2][ 0] = -18
            diagonals[3][ 0] =  9
            diagonals[4][ 0] = -2
            A_11 = diags([diagonals[0][1:],diagonals[1],diagonals[2][:-1],diagonals[3][:-2],diagonals[4][:-3]],\
                         offsets)#,format="csr")
        return A_11
    def _Phi_n_equation_LHS2(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        g_s dependence in the Phi_n equation.
        (Here Phi_n = phi_n*h, the input v_new must contain Phi_n rather than phi_n)
        """
        # Note: there is no dependence on g_s
        return None 
    def _Phi_n_equation_LHS3(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        g_b dependence in the Phi_n equation.
        (Here Phi_n = phi_n*h, the input v_new must contain Phi_n rather than phi_n)
        """
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        # Construct the diagonal, straightforward in this case 
        A_diag = -0.5*dt*Phi_n_new
        A_diag[[0,-1]] = 0 # set to zero at the two ends
        if True:
            # Need to add this in conjunction with the 'articicial' BC
            A_diag[1] = 0 
        return diags(A_diag)#,format="csr")
    def _g_s_equation_RHS(self,v_old,v_new,dt=None):
        """
        Calculate the RHS vector component corresponding to the g_s equation.
        """
        r = self._r
        nr = len(r)
        dr = r[1]
        r_half = self._r_half
        h_ast = self._h_ast
        D = self._D
        Q_s = self._Q_s
        h_old,phi_n_old,g_s_old,g_b_old = v_old
        h_new,phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        # Calculate spatial stencil and add to the interior of the rhs vector
        rhs = np.empty(nr)
        rhs[1:-1] = g_s_old[1:-1]-g_s_new[1:-1] 
        rhs[1:-1] += (0.5*dt*D/dr**2)*( r_half[1:]*(g_s_old[2:]-g_s_old[1:-1]) \
                                       +r_half[1:]*(g_s_new[2:]-g_s_new[1:-1]) \
                                       -r_half[:-1]*(g_s_old[1:-1]-g_s_old[:-2]) \
                                       -r_half[:-1]*(g_s_new[1:-1]-g_s_new[:-2]))/r[1:-1]
        # Now the forcing term
        rhs[1:-1] -= 0.5*dt*D*Q_s*( (h_old>h_ast)*(g_s_old-g_b_old)\
                                   +(h_new>h_ast)*(g_s_new-g_b_new))[1:-1]
        # Now set the appropriate boundary values ( (d/dr)g_s=0 at both r=0 and r=r_max )
        rhs[ 0] = -3.0*g_s_new[ 0]+4.0*g_s_new[ 1]-g_s_new[ 2]
        rhs[-1] =  3.0*g_s_new[-1]-4.0*g_s_new[-2]+g_s_new[-3]
        # done
        return rhs
    def _g_s_equation_LHS0(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        h dependence in the g_s equation.
        """
        # There is no dependence (we don't discretise the gradient of the heavyside function)
        return None
    def _g_s_equation_LHS1(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        Phi_n dependence in the g_s equation.
        """
        # There is no dependence 
        return None
    def _g_s_equation_LHS2(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        g_s dependence in the g_s equation.
        """
        r = self._r
        nr = len(r)
        dr = r[1]
        r_half = self._r_half
        h_ast = self._h_ast
        D = self._D
        Q_s = self._Q_s
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        # Pre-allocate diagonals for the sparse matrix construction
        A_diag_m2 = np.zeros(nr) # for upwind component
        A_diag_m1 = np.empty(nr)
        A_diag_p0 = np.ones(nr)  # note the ones
        A_diag_p1 = np.empty(nr) 
        A_diag_p2 = np.zeros(nr) # for boundary components
        # Set the entries which enforce the BC at r=0 and r=r_max (these don't change)
        A_diag_p0[ 0] =  3.0  
        A_diag_p1[ 0] = -4.0
        A_diag_p2[ 0] =  1.0
        A_diag_m2[-1] = -1.0  
        A_diag_m1[-1] =  4.0
        A_diag_p0[-1] = -3.0
        # Set up the diagonals relating to the interior
        A_diag_m1[1:-1]  = -(0.5*dt*D/dr**2)*r_half[:-1]/r[1:-1]
        A_diag_p0[1:-1] +=  (0.5*dt*D/dr**2)*2.0 # 2.0=(r_half[1:]+r_half[:-1])/r[1:-1]
        A_diag_p1[1:-1]  = -(0.5*dt*D/dr**2)*r_half[1:]/r[1:-1]
        # Now add the forcing component
        A_diag_p0[1:-1] += (0.5*dt*D*Q_s)*(h_new>h_ast)[1:-1]
        # Final construction 
        A_22 = diags([A_diag_m2[2:],A_diag_m1[1:],A_diag_p0,A_diag_p1[:-1],A_diag_p2[:-2]],\
                     [-2,-1,0,1,2])#,format="csr")
        return A_22
    def _g_s_equation_LHS3(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        g_b dependence in the g_s equation.
        """
        h_ast = self._h_ast
        D = self._D
        Q_s = self._Q_s
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        A_diag = -0.5*dt*D*Q_s*(h_new>h_ast)
        A_diag[[0,-1]] = 0
        return diags(A_diag)#,format="csr")
    def _g_b_equation_RHS(self,v_old,v_new,dt=None):
        """
        Calculate the RHS vector component corresponding to the g_b equation.
        """
        r = self._r
        nr = len(r)
        dr = r[1]
        r_half = self._r_half
        g_ast = self._gamma_ast
        h_ast = self._h_ast
        Pe = self._Pe
        Q_b = self._Q_b
        Upsilon = self._Upsilon
        lambda_ast = self._lambda_ast
        h_old,Phi_n_old,g_s_old,g_b_old = v_old
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        # Additional terms for convenience
        H_old = (h_old>h_ast)
        H_new = (h_new>h_ast)
        h_old_half = 0.5*(h_old[:-1]+h_old[1:])
        h_new_half = 0.5*(h_new[:-1]+h_new[1:])
        # Initialise the rhs vector with the advective terms...
        prefix = np.empty(nr)
        prefix[[0,-1]] = 0
        prefix[1:-1] = (-0.5*dt*Pe*g_ast/3.0)*r[1:-1]**(-1)
        #adv_new = self._advective_term(r,h_new,3,g_b_new*(1-phi_n_new),True,prefix)
        #adv_old = self._advective_term(r,h_old,3,g_b_old*(1-phi_n_old),True,prefix)
        adv_new = self._advective_term(r,h_new,3,g_b_new,True,prefix*H_new)
        adv_old = self._advective_term(r,h_old,3,g_b_old,True,prefix*H_old)
        rhs = adv_new+adv_old
        if np.isfinite(lambda_ast): # add the slip terms
            prefix[1:-1] *= 3.0/lambda_ast
            adv_new = self._advective_term(r,h_new,2,g_b_new,True,prefix*H_new)
            adv_old = self._advective_term(r,h_old,2,g_b_old,True,prefix*H_old)
            rhs += adv_new+adv_old
        # Now add the easy terms
        rhs[1:-1] += -Pe*0.5*(h_old+h_new)[1:-1]*(g_b_new-g_b_old)[1:-1]
        rhs[1:-1] += 0.5*dt*Q_b*( H_old*(g_s_old-g_b_old) \
                                 +H_new*(g_s_new-g_b_new))[1:-1]
        # The following term no longer has h as it is absorbed into Phi_n
        rhs[1:-1] -= 0.5*dt*Upsilon*( H_old*Phi_n_old*g_b_old \
                                     +H_new*Phi_n_new*g_b_new)[1:-1]
        # Now the second order g_b term
        rhs[1:-1] += (0.5*dt/dr**2)*( H_old[1:-1]*r_half[1:  ]*h_old_half[1:  ]*(g_b_old[2:  ]-g_b_old[1:-1]) \
                                     -H_old[1:-1]*r_half[ :-1]*h_old_half[ :-1]*(g_b_old[1:-1]-g_b_old[ :-2]) \
                                     +H_new[1:-1]*r_half[1:  ]*h_new_half[1:  ]*(g_b_new[2:  ]-g_b_new[1:-1]) \
                                     -H_new[1:-1]*r_half[ :-1]*h_new_half[ :-1]*(g_b_new[1:-1]-g_b_new[ :-2]))/r[1:-1]
        """
        # Perhaps need to treat the whole thing similar to the h equation,
        # e.g. by introducing g_b_old_half,phi_n_new_half,etc. 
        # but I would then also need biased stencils near the ends anyway...
        if False:
            inds = 1-H_new
            #inds = 1-H_old
            rhs[inds] = 0
        """
        # Lastly set the appropriate boundary values ( (d/dr)g_s=0 at both r=0 and r=r_max )
        rhs[ 0] = -3.0*g_b_new[ 0]+4.0*g_b_new[ 1]-g_b_new[ 2]
        rhs[-1] =  3.0*g_b_new[-1]-4.0*g_b_new[-2]+g_b_new[-3]
        if True:
            # Add an artificial BC, for the same reason as Phi_n, to avoid dealing with the wide h stencil there...
            rhs[ 1] = 4*g_b_new[0]-7*g_b_new[1]+4*g_b_new[2]-g_b_new[3]
        # done
        return rhs
    def _g_b_equation_LHS0(self,g_b_old,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        h dependence in the g_b equation.
        """
        r = self._r
        nr = len(r)
        dr = r[1]
        r_half = self._r_half
        g_ast = self._gamma_ast
        h_ast = self._h_ast
        Pe = self._Pe
        lambda_ast = self._lambda_ast
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        # Additional vector for convenience
        H_over_r = (h_new>h_ast)[1:-1]/r[1:-1]
        # Calculate/fetch the diagonals from the advective term
        prefix = np.empty(nr)
        prefix[[0,-1]] = 0
        prefix[1:-1] = (0.5*dt*Pe*g_ast/3.0)*H_over_r
        diagonals,offsets = self._advective_term_h_gradient(r,h_new,3,g_b_new,True,prefix)
        for i in range(len(diagonals)):
            assert offsets[i]==i-4 # sanity check
        if np.isfinite(lambda_ast): # add the slip terms
            prefix[1:-1] *= 3.0/lambda_ast
            diagonals2,offsets2 = self._advective_term_h_gradient(r,h_new,2,g_b_new,True,prefix)
            for i in range(len(diagonals2)):
                assert offsets2[i]==offsets[i]
                diagonals[i][1:-1] += diagonals2[i][1:-1]
        # Add the parts from the second order g_b term
        diagonals[3][1:-1] += -(0.25*dt/dr**2)*(-r_half[:-1]*(g_b_new[1:-1]-g_b_new[ :-2]))*H_over_r
        diagonals[4][1:-1] -=  (0.25*dt/dr**2)*(-r_half[:-1]*(g_b_new[1:-1]-g_b_new[ :-2]) \
                                                +r_half[1: ]*(g_b_new[2:  ]-g_b_new[1:-1]))*H_over_r
        diagonals[5][1:-1] += -(0.25*dt/dr**2)*( r_half[1: ]*(g_b_new[2:  ]-g_b_new[1:-1]))*H_over_r
        # Add the additional parts to the main diagonal
        diagonals[4][1:-1] += Pe*0.5*(g_b_new-g_b_old)[1:-1]
        """
        #
        if False:
            inds = 1-H_new
            #inds = (h_old<=h_ast) # not available currently
            for i in range(len(diagonals)):
                diagonals[i][inds] = 0
        """
        if True:
            # Related to an artificial BC
            for diag in diagonals:
                diag[1] = 0
        # Final construction 
        A_30 = diags([diagonals[0][4:],diagonals[1][3:],diagonals[2][2:],diagonals[3][1:],diagonals[4],\
                      diagonals[5][:-1],diagonals[6][:-2],diagonals[7][:-3]],\
                      offsets)#,format='csr')
        return A_30
    def _g_b_equation_LHS1(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        Phi_n dependence in the g_b equation.
        (Note: this is much simplified with the 
        modification of the g_b equation.)
        """
        h_ast = self._h_ast
        Upsilon = self._Upsilon
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        A_diag = 0.5*dt*Upsilon*(h_new>h_ast)*g_b_new
        A_diag[[0,-1]] = 0
        if True:
            # Related to an artificial BC
            A_diag[1] = 0
        return diags(A_diag)#,format='csr')
    def _g_b_equation_LHS2(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        g_s dependence in the g_b equation.
        """
        h_ast = self._h_ast
        Q_b = self._Q_b
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        A_diag = -0.5*dt*Q_b*(h_new>h_ast)
        A_diag[[0,-1]] = 0
        if True:
            # Related to an artificial BC
            A_diag[1] = 0
        return diags(A_diag)#,format='csr')
    def _g_b_equation_LHS3(self,h_old,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        g_b dependence in the g_b equation.
        """
        r = self._r
        nr = len(r)
        dr = r[1]
        r_half = self._r_half
        g_ast = self._gamma_ast
        h_ast = self._h_ast
        Pe = self._Pe
        Q_b = self._Q_b
        Upsilon = self._Upsilon
        lambda_ast = self._lambda_ast
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        # Additional vector for convenience
        h_half = 0.5*(h_new[1:]+h_new[:-1])
        H_new = h_new>h_ast
        H_over_r = H_new[1:-1]/r[1:-1]
        # Calculate/fetch the diagonals from the advective term
        prefix = np.empty(nr)
        prefix[[0,-1]] = 0
        prefix[1:-1] = (0.5*dt*Pe*g_ast/3.0)*r[1:-1]**(-1)
        diagonals,offsets = self._advective_term_f_gradient(r,h_new,3,g_b_new,True,prefix*H_new)
        if np.isfinite(lambda_ast): # add the slip terms
            prefix[1:-1] *= 3.0/lambda_ast
            diagonals2,offsets2 = self._advective_term_f_gradient(r,h_new,2,g_b_new,True,prefix*H_new)
            for i in range(len(diagonals2)):
                assert offsets2[i]==offsets[i]
                diagonals[i][1:-1] += diagonals2[i][1:-1]
        diagonals.insert(0,np.zeros(nr))
        offsets.insert(0,-2)
        diagonals.append(np.zeros(nr))
        offsets.append(2)
        # Add the appropriate terms to the main diagonal
        diagonals[2][1:-1] += 0.5*Pe*(h_old+h_new)[1:-1]   # Note h_old is needed here with my current discretisation
        diagonals[2][1:-1] += 0.5*dt*Q_b*H_new[1:-1]
        diagonals[2][1:-1] += 0.5*dt*Upsilon*(H_new*Phi_n_new)[1:-1]
        # Now the diagonal components for the second order g_b term
        diagonals[1][1:-1] -=  (0.5*dt/dr**2)*r_half[:-1]*h_half[:-1]*H_over_r
        diagonals[2][1:-1] -= -(0.5*dt/dr**2)*(r_half[1: ]*h_half[1: ]+r_half[:-1]*h_half[:-1])*H_over_r
        diagonals[3][1:-1] -=  (0.5*dt/dr**2)*r_half[1: ]*h_half[1: ]*H_over_r
        """
        #
        if False:
            inds = 1-H_new
            #inds = (h_old<=h_ast) 
            for i in range(len(diagonals)):
                diagonals[i][inds] = (0 if offsets[i]!=2 else 1) 
        """
        # Set the entries which enforce the BC at r=0 and r=r_max (these don't change)
        diagonals[2][ 0] =  3.0  
        diagonals[3][ 0] = -4.0
        diagonals[4][ 0] =  1.0
        diagonals[0][-1] = -1.0  
        diagonals[1][-1] =  4.0
        diagonals[2][-1] = -3.0
        if True:
            # Add an artificial BC, for the same reason as Phi_n
            diagonals[1][ 1] = -4  
            diagonals[2][ 1] =  7  
            diagonals[3][ 1] = -4
            diagonals[4][ 1] =  1
        # Final construction 
        A_33 = diags([diagonals[0][2:],diagonals[1][1:],diagonals[2],diagonals[3][:-1],diagonals[4][:-2]],\
                     offsets)#,format='csr')
        return A_33
    # public methods for solving
    def set_solver(self,solver):
        """
        Set/change the solver used by the class
        """
        if solver in ['DCN','FCN']:
            self._solver = solver
        else:
            print('Warning: Requested solver does not exist, falling back to DCN')
            self._solver = 'DCN' # default
        # done
    def solve(self,T,dt=None):
        """
        Solve the biofilm evolution for a duration T (from the current time)
        
        Optional: dt can be provided to override that stored internally.
        """
        # Run a solver based on self._solver
        if self._solver=='DCN':
            # A de-coupled non-linear Crank-Nicolson solver
            # (Only h and phi_n are non-linear given the decoupling)
            self._decoupled_Crank_Nicolson(T,dt)
        if self._solver=='FCN':
            # The fully coupled non-linear Crank-Nicolson solver
            self._full_Crank_Nicolson(T,dt)
        return self._h,self._Phi_n,self._g_s,self._g_b
    def _decoupled_Crank_Nicolson(self,T,dt=None):
        """
        This solves the non-linear system equations using Newton iterations
        on each field individually. Consequently the four fields of 
        interest are only weakly coupled through the time stepping.
        This appears to be fine for all parameters ranges we have 
        considered, and is much cheaper computationally.
        """
        # Setup...
        v_old = [self._h,self._Phi_n,self._g_s,self._g_b]
        h_new = self._h.copy()
        Phi_n_new = self._Phi_n.copy()
        g_s_new = self._g_s.copy()
        g_b_new = self._g_b.copy()
        v_new = [h_new,Phi_n_new,g_s_new,g_b_new]
        if dt is None:
            dt = self._dt
        # Define the newton iteration
        def decoupled_newton_iterate(v_old,v_new,dt,order=[0,1,2,3],newt_its=20,newt_tol=1.0E-8,newt_verbose=self._verbose):
            """
            Construct the sparse blocks and RHS vectors for a Newton iteration
            (based on a de-coupled Crank-Nicolson discretisation)
            """
            h_old,Phi_n_old,g_s_old,g_b_old = v_old
            h_new,Phi_n_new,g_s_new,g_b_new = v_new
            for block in order:
                for k in range(newt_its):
                    if block==0:
                        # h equation rows block matrices and rhs vector
                        b_0  = self._h_equation_RHS(v_old,v_new,dt)
                        A_00 = self._h_equation_LHS0(v_new,dt)
                        # Solve the linear system and update current guess
                        dh = spsolve(A_00.tocsr(),b_0)
                        h_new += dh
                        eps = np.linalg.norm(dh)/np.linalg.norm(h_new)
                    elif block==1:
                        # phi_n equation row block matrices and rhs vector
                        b_1  = self._Phi_n_equation_RHS(v_old,v_new,dt)
                        A_11 = self._Phi_n_equation_LHS1(v_new,dt)
                        # Solve the linear system and update current guess
                        dPhi_n = spsolve(A_11.tocsr(),b_1)
                        Phi_n_new += dPhi_n
                        eps = np.linalg.norm(dPhi_n)/np.linalg.norm(Phi_n_new)
                    elif block==2:
                        # g_s equation row block matrices and rhs vector
                        b_2  = self._g_s_equation_RHS(v_old,v_new,dt) 
                        A_22 = self._g_s_equation_LHS2(v_new,dt)
                        # Solve the linear system and update current guess
                        dg_s = spsolve(A_22.tocsr(),b_2)
                        g_s_new += dg_s
                        eps = np.linalg.norm(dg_s)/np.linalg.norm(g_s_new)
                    elif block==3:
                        # g_b equation row block matrices and rhs vector
                        b_3  = self._g_b_equation_RHS(v_old,v_new,dt)  
                        A_33 = self._g_b_equation_LHS3(h_old,v_new,dt) 
                        # Solve the linear system and update current guess
                        dg_b = spsolve(A_33.tocsr(),b_3)
                        g_b_new += dg_b
                        eps = np.linalg.norm(dg_b)/np.linalg.norm(g_b_new)
                    # Check epsilon and the current iteration number for termination conditions...
                    if newt_verbose:
                        print("Newton Method: Completed iteration {:d} with eps={:g}".format(k,eps))
                    if eps<newt_tol:# or block in [1,2,3]: # Note: blocks 1,2,3 are linear
                        if newt_verbose:
                            print("Newton Method: Converged within {:g} in {:d} iterations".format(newt_tol,newt_its))
                        break
                    if k==newt_its-1:
                        print("Newton Method: Failed to converge in {:d} iterations (block {:d}, eps={:g})".format(newt_its,block,eps))
            # done
        # Now perform the Newton iterations until the final time is reached
        t = self._t
        S = t
        while t<=S+T-dt:
            decoupled_newton_iterate(v_old,v_new,dt)
            t += dt
            self._h[:]     = h_new[:]
            self._Phi_n[:] = Phi_n_new[:]
            self._g_s[:]   = g_s_new[:]
            self._g_b[:]   = g_b_new[:]
        if t<S+T and (S+T-t)>1.0E-12:
            decoupled_newton_iterate(v_old,v_new,S+T-t)
            t = S+T
            self._h[:]     = h_new[:]
            self._Phi_n[:] = Phi_n_new[:]
            self._g_s[:]   = g_s_new[:]
            self._g_b[:]   = g_b_new[:]
        # done, no return
    def _full_Crank_Nicolson(self,T,dt=None):
        """
        This solves the non-linear system equations using Newton iterations
        on the entire system of equations simultaneously. Consequently the 
        four fields of interest are strongly coupled through each time step.
        This is generally much costlier for negligible gain over the 
        'decoupled' routine. It may however be useful if we find
        parameters in which non-linear effects are more important.
        """
        # Setup...
        v_old = [self._h,self._Phi_n,self._g_s,self._g_b]
        h_new = self._h.copy()
        Phi_n_new = self._Phi_n.copy()
        g_s_new = self._g_s.copy()
        g_b_new = self._g_b.copy()
        v_new = [h_new,Phi_n_new,g_s_new,g_b_new]
        if dt is None:
            dt = self._dt
        # Define the newton iteration
        def fully_coupled_newton_iterate(v_old,v_new,dt,newt_its=20,newt_tol=1.0E-8,newt_verbose=self._verbose):
            """
            Construct the sparse blocks and RHS vectors for a Newton iteration
            (based on a fully coupled Crank-Nicolson discretisation)
            """
            h_old,Phi_n_old,g_s_old,g_b_old = v_old
            h_new,Phi_n_new,g_s_new,g_b_new = v_new
            nr = len(self._r)
            for k in range(newt_its):
                # h equation rows block matrices and rhs vector
                b_0  = self._h_equation_RHS(v_old,v_new,dt)
                A_00 = self._h_equation_LHS0(v_new,dt)
                A_01 = self._h_equation_LHS1(v_new,dt) # simple diag
                A_02 = self._h_equation_LHS2(v_new,dt) # None
                A_03 = self._h_equation_LHS3(v_new,dt) # simple diag
                # phi_n equation row block matrices and rhs vector
                b_1  = self._Phi_n_equation_RHS(v_old,v_new,dt) 
                A_10 = self._Phi_n_equation_LHS0(v_new,dt)
                A_11 = self._Phi_n_equation_LHS1(v_new,dt)      
                A_12 = self._Phi_n_equation_LHS2(v_new,dt) # None
                A_13 = self._Phi_n_equation_LHS3(v_new,dt) # simple diag
                # g_s equation row block matrices and rhs vector
                b_2  = self._g_s_equation_RHS(v_old,v_new,dt) 
                A_20 = self._g_s_equation_LHS0(v_new,dt) # None
                A_21 = self._g_s_equation_LHS1(v_new,dt) # None
                A_22 = self._g_s_equation_LHS2(v_new,dt)     
                A_23 = self._g_s_equation_LHS3(v_new,dt) # simple diag
                # g_b equation row block matrices and rhs vector
                b_3  = self._g_b_equation_RHS(v_old,v_new,dt)
                A_30 = self._g_b_equation_LHS0(g_b_old,v_new,dt) 
                A_31 = self._g_b_equation_LHS1(v_new,dt) 
                A_32 = self._g_b_equation_LHS2(v_new,dt) # simple diag
                A_33 = self._g_b_equation_LHS3(h_old,v_new,dt) 
                # Construct the sparse block matrix
                b_full = np.concatenate([b_0,b_1,b_2,b_3])
                A_full = bmat([[A_00,A_01,A_02,A_03],
                               [A_10,A_11,A_12,A_13],
                               [A_20,A_21,A_22,A_23],
                               [A_30,A_31,A_32,A_33]],format='csr')
                # Solve the linear system
                dv = spsolve(A_full,b_full)
                eps = np.linalg.norm(dv)/np.linalg.norm(np.concatenate(v_new))
                # Update current guess
                h_new     += dv[    :  nr]
                Phi_n_new += dv[  nr:2*nr]
                g_s_new   += dv[2*nr:3*nr]
                g_b_new   += dv[3*nr:    ]
                # Check epsilon and the current iteration number for termination conditions...
                if newt_verbose:
                    print("Newton Method: Completed iteration {:d} with eps={:g}".format(k,eps))
                if eps<newt_tol:
                    if newt_verbose:
                        print("Newton Method: Converged within {:g} in {:d} iterations".format(newt_tol,newt_its))
                    break
                if k==newt_its-1:
                    print("Newton Method: Failed to converge in {:d} iterations".format(newt_its))
            # done
        # Now perform the Newton iterations until the final time is reached 
        t = self._t
        S = t
        while t<=S+T-dt:
            fully_coupled_newton_iterate(v_old,v_new,dt)
            t += dt
            self._h[:]     = h_new[:]
            self._Phi_n[:] = Phi_n_new[:]
            self._g_s[:]   = g_s_new[:]
            self._g_b[:]   = g_b_new[:]
        if t<S+T and (S+T-t)>1.0E-12:
            fully_coupled_newton_iterate(v_old,v_new,S+T-t)
            t = S+T
            self._h[:]     = h_new[:]
            self._Phi_n[:] = Phi_n_new[:]
            self._g_s[:]   = g_s_new[:]
            self._g_b[:]   = g_b_new[:]
        # done, no return
    # End of class    
    
