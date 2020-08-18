"""
A helper class for solving the non-linear time dependent equations 
of biofilm growth which includes models of the cell concentration 
and also nutrient concentrations in both the substrate and biofilm.
All of these are asumed to be radially symmetric and depend on 
r and t, and the article concentration additionally depends on z. 
The specific equations solved by this class are described in the
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
of equations that result from a compact finite difference discretisation
(although iterative solvers for some variables can be toggled through 
private class switches). Both methods can be expected to achieve 2nd order 
convergence in both space and time.

Compatibility notes: 
The code was written in Python3 (3.7.3 specifically) although it 
should also work in 2.7.x releases that are not to old as well.
The scientific computing packages numpy and scipy are required.
Again, any version that is not unreasonably old should be fine.
You will probably also want matplotlib for plotting.

Maintainer: Brendan Harding 
Initial development: June-July 2020
Last updated: August 2020
"""

import numpy as np
from scipy.sparse.linalg import spsolve,spilu,LinearOperator,gmres,bicgstab
from scipy.sparse import diags,bmat,coo_matrix

class gmres_counter(object):
    """
    Convenience class for monitoring gmres iterations (from scipy.sparse.linalg)
    (Useful for debugging purposes)
    """
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('gmres: iteration {:03d} residual = {:s}'.format(self.niter,str(rk)))

class BiofilmTwoDLubricationModel(object):
    """
    Helper class for solving the PDEs describing the development of 
    a radially symmetric and thin yeast biofilm over time.
    The model/system that is solved includes the biofilm height,
    the cell concentration, and the nutrient concentrations in both
    the biofilm and the substrate.
    """
    def __init__(self,R=2.0,dr=0.5**7,nxi=33,dt=None,params=None,solver='DCN',verbose=False):
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
        self._nxi = nxi
        self._xi = np.linspace(0,1,nxi)
        self._R,self._XI = np.meshgrid(self._r,self._xi)
        # Set up the parameters
        if dt is None:
            self._dt = 0.25*dr # this is quite conservative... (and assumes dr<h*dxi)
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
        # The following were used in initial debugging and testing and generally need not be changed
        self._Phi_n_DCN_solver = 3 # Changes the numerical method used for solving Phi_n
        self._FCN_solver_mode = -1 # Change the FCN solver 
        self._add_top_Phi_bc = False
        self._use_artificial_dr_bc = True # untested with False...
        # done
    def _set_default_initial_conditions(self):
        """
        Sets the initial conditions to be those described by 
        equation 6.22 of Alex Tam's thesis.
        """
        self._t = 0
        r = self._r
        R = self._R
        XI = self._XI
        self._h = self._b + (self._H0-self._b)*(r<1)*(1-r**2)**4
        self._Phi_n = (XI**3-0.5*XI**4)*self._h[np.newaxis,:]*(R<1)*(1-3*R**2+2*R**3)
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
    def get_xi(self):
        """
        Returns the array for the radial coordinates.
        """
        return self._xi
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
        Note: This will not alter Phi_n=int_0^{h xi} phi_n dz. If it is desired that this 
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
        Update the cumulative cell volume fraction Phi_n (=int_0^{h xi} phi_n dz). 
        For example, use this to set the initial condition.
        (Note this over-writes the current solution in the class.)
        It is expected that Phi_n be provided in the re-scaled coordinates r,xi.
        Accepts a callable function Phi_n(r,xi), or an array (with correct shape).
        """
        if callable(Phi_n):
            self._Phi_n[:,:] = Phi_n(self._XI,self._R)
        else:
            assert Phi_n.shape==self._R.shape
            self._Phi_n[:,:] = Phi_n
        # done
    def get_Phi_n(self):
        """
        Returns the current cumulative cell volume fraction Phi_n (=int_0^{h xi} phi_n dz).
        (Note this is given with respect to the re-scaled coordinates r,xi.)
        """
        return self._Phi_n
    def get_phi_n_bar(self):
        """
        Returns the vertically averaged cell volume fraction bar{phi_n} =(1/h) int_0^{h} phi_n dz.
        (Note this is given with respect to the re-scaled coordinates r,xi.)
        """
        return self._Phi_n[-1,:]/self._h
    def set_phi_n(self,phi_n):
        """
        Update the cell volume fraction phi_n. 
        For example, use this to set the initial condition.
        (Note this over-writes the current solution in the class.)
        It is expected that phi_n be provided in re-scaled coordinates r,xi.
        Accepts a callable function phi_n(r,xi), or an array (with correct length).
        Note: This internally updates Phi_n=\int_0^{h xi} phi_n dz using the existing h.
        If h is also to be updated, it should be done first!
        """
        XI,R = self._XI,self._R
        if callable(phi_n):
            phi_n_int_dxi = XI[1,0]*np.cumsum(0.5*(phi_n(XI,R)[1:,:]+phi_n(XI,R)[:-1,:]),axis=0)
        else:
            assert phi_n.shape==self._R.shape
            phi_n_int_dxi = XI[1,0]*np.cumsum(0.5*(phi_n[1:,:]+phi_n[:-1,:]),axis=0)
        self._Phi_n[0,:] = 0
        self._Phi_n[1:,:] = phi_n_int_dxi*self._h[np.newaxis,:]
        self._Phi_n[(self._h<self._h_ast)[np.newaxis,:]] = 0 # zero areas where h is small
        # done
    def get_phi_n(self):
        """
        Returns the current cell volume fraction phi_n. 
        (Note this is given with respect to the re-scaled coordinates r,xi.)
        """
        phi_n = np.empty_like(self._Phi_n)
        phi_n[1:-1,:] = 0.5*(self._Phi_n[2:,:]-self._Phi_n[:-2,:])*(self._nxi-1)/self._h[np.newaxis,:]
        phi_n[ 0,:] = 0.5*(-3*self._Phi_n[ 0,:]+4*self._Phi_n[ 1,:]-self._Phi_n[ 2,:])*(self._nxi-1)/self._h[np.newaxis,:]
        phi_n[-1,:] = 0.5*( 3*self._Phi_n[-1,:]-4*self._Phi_n[-2,:]+self._Phi_n[-3,:])*(self._nxi-1)/self._h[np.newaxis,:]
        phi_n[:,self._h<self._h_ast] = 0
        return phi_n
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
            assert len(g_b)==len(self._r)
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
        if f is None: # This is the only place f is actually used...
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
        forcing_old = (h_old>h_ast)*(1.0+Psi_m)*Phi_n_old[-1,:]*g_b_old
        forcing_new = (h_new>h_ast)*(1.0+Psi_m)*Phi_n_new[-1,:]*g_b_new
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
        Phi_n dependence in the height equation (Phi_n = int_0^{h xi} phi_n dz).
        The internal time step dt is used if one is not provided.
        """
        h_ast = self._h_ast
        Psi_m = self._Psi_m
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        # Note: this block has a rectangular shape
        nr,nxi = len(self._r),len(self._xi)
        row = np.arange(2,nr-2)
        col = nxi-1+nxi*row
        dat = -0.5*dt*(1.0+Psi_m)*((h_new>h_ast)*g_b_new)[2:-2]
        return coo_matrix((dat,(row,col)),shape=(nr,nr*nxi))#.tocsr()
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
        A_diag = -0.5*dt*(1.0+Psi_m)*(h_new>h_ast)*Phi_n_new[-1,:]
        A_diag[[0,1,-2,-1]] = 0
        return diags(A_diag)#,format="csr")
    # Add private methods relating to the discretisation of the fourth order 'advective' term
    def _advective_term_alt(self,r,h,f,near_boundary=True):
        """
        Finite difference discretisation of:
        (d/dr)[ f r (d/dr)[ (1/r) (d/dr)[ r (dh/dr) ] ] ]
        This version handles f which is two dimensional.
        Note the h**p factor and the prefix have been dropped in this alt version.
        """
        r_half = 0.5*(r[1:]+r[:-1])
        dr = r[1]-r[0]
        h_half = 0.5*(h[1:]+h[:-1])
        D_half =  (r_half[2:  ]*(h[3:  ]-h[2:-1])-r_half[1:-1]*(h[2:-1]-h[1:-2]))/r[2:-1]\
                 -(r_half[1:-1]*(h[2:-1]-h[1:-2])-r_half[ :-2]*(h[1:-2]-h[ :-3]))/r[1:-2]
        f_half = 0.5*(f[:,1:]+f[:,:-1])
        res = np.empty(f.shape)
        res[:,[0,1,-2,-1]] = 0
        res[:,2:-2] =  r_half[np.newaxis,2:-1]*D_half[np.newaxis,1:  ]*f_half[:,2:-1] \
                      -r_half[np.newaxis,1:-2]*D_half[np.newaxis, :-1]*f_half[:,1:-2]
        if near_boundary:
            # At index one we expoloit that 0 = (d/dr)[ (1/r) (d/dr)[ r (dh/dr) ] ] for r=0
            D_p2 = 0.5*(D_half[ 0]+D_half[ 1])
            res[:,1] = 0.5*r[2]*D_p2*f[:,2]
            # At index -2 we can exploit that 0 = (dh/dr)
            # The width of the stencil is widened to achieve this though...
            D_m5o2 =  0.25*(r[-2]*(h[-1]-h[-3])-r[-4]*(h[-3]-h[-5]))/r[-3]\
                     -0.25*(r[-3]*(h[-2]-h[-4])-r[-5]*(h[-4]-h[-6]))/r[-4]
            D_m3o2 =  0.25*(                   -r[-3]*(h[-2]-h[-4]))/r[-2]\
                     -0.25*(r[-2]*(h[-1]-h[-3])-r[-4]*(h[-3]-h[-5]))/r[-3]
            res[:,-2] =  r_half[-1]*D_m3o2*f_half[:,-1] \
                        -r_half[-2]*D_m5o2*f_half[:,-2]
        return res/dr**4
    def _advective_term_h_gradient_alt(self,r,h,p=3,f=None,near_boundary=True,prefix=None):
        """
        Finite difference discretisation of the gradient of
        prefix * (d/dr)[ r h^p f (d/dr)[ (1/r) (d/dr)[ r (dh/dr) ] ] ]
        with respect to h.
        This version handles f which is two dimensional.
        
        Note: the caller is responsible for enforcing boundary conditions
        """
        r_half = 0.5*(r[1:]+r[:-1])
        dr = r[1]-r[0]
        h_half = 0.5*(h[1:]+h[:-1])
        D_half =  (r_half[2:  ]*(h[3:  ]-h[2:-1])-r_half[1:-1]*(h[2:-1]-h[1:-2]))/r[2:-1]\
                 -(r_half[1:-1]*(h[2:-1]-h[1:-2])-r_half[ :-2]*(h[1:-2]-h[ :-3]))/r[1:-2]
        if f is None:
            f = np.ones((1,len(r)))
            f_half = f[:,:-1]
        else:
            f_half = 0.5*(f[:,1:]+f[:,:-1])
        Dh_diag_p2 = np.empty(f.shape)
        Dh_diag_p1 = np.empty(f.shape)
        Dh_diag_p0 = np.empty(f.shape)
        Dh_diag_m1 = np.empty(f.shape)
        Dh_diag_m2 = np.empty(f.shape)
        Dh_diag_p2[:,[0,1,-2,-1]] = 0
        Dh_diag_p1[:,[0,1,-2,-1]] = 0
        Dh_diag_p0[:,[0,1,-2,-1]] = 0
        Dh_diag_m1[:,[0,1,-2,-1]] = 0
        Dh_diag_m2[:,[0,1,-2,-1]] = 0
        Dh_diag_p1[:,2:-2] =  f_half[:,2:-1]*(r_half[2:-1]*0.5*p*h_half[2:-1]**(p-1)*D_half[1:]/dr**4)[np.newaxis,:]
        Dh_diag_p0[:,2:-2] =  f_half[:,2:-1]*(r_half[2:-1]*0.5*p*h_half[2:-1]**(p-1)*D_half[1:]/dr**4)[np.newaxis,:] \
                             -f_half[:,1:-2]*(r_half[1:-2]*0.5*p*h_half[1:-2]**(p-1)*D_half[:-1]/dr**4)[np.newaxis,:]
        Dh_diag_m1[:,2:-2] = -f_half[:,1:-2]*(r_half[1:-2]*0.5*p*h_half[1:-2]**(p-1)*D_half[:-1]/dr**4)[np.newaxis,:]
        # I think the following 5 are okay...
        Dh_diag_p2[:,2:-2]  =  f_half[:,2:-1]*(r_half[2:-1]*h_half[2:-1]**p*(r_half[3:  ]/r[3:-1])/dr**4)[np.newaxis,:]
        Dh_diag_p1[:,2:-2] += -f_half[:,2:-1]*(r_half[2:-1]*h_half[2:-1]**p*(r_half[2:-1]/r[2:-2]+2)/dr**4)[np.newaxis,:] \
                              -f_half[:,1:-2]*(r_half[1:-2]*h_half[1:-2]**p*(r_half[2:-1]/r[2:-2])/dr**4)[np.newaxis,:]  
        Dh_diag_p0[:,2:-2] +=  f_half[:,2:-1]*(r_half[2:-1]*h_half[2:-1]**p*(r_half[2:-1]/r[3:-1]+2)/dr**4)[np.newaxis,:] \
                              +f_half[:,1:-2]*(r_half[1:-2]*h_half[1:-2]**p*(r_half[1:-2]/r[1:-3]+2)/dr**4)[np.newaxis,:]  
        Dh_diag_m1[:,2:-2] += -f_half[:,2:-1]*(r_half[2:-1]*h_half[2:-1]**p*(r_half[1:-2]/r[2:-2])/dr**4)[np.newaxis,:] \
                              -f_half[:,1:-2]*(r_half[1:-2]*h_half[1:-2]**p*(r_half[1:-2]/r[2:-2]+2)/dr**4)[np.newaxis,:] 
        Dh_diag_m2[:,2:-2]  =  f_half[:,1:-2]*(r_half[1:-2]*h_half[1:-2]**p*(r_half[ :-3]/r[1:-3])/dr**4)[np.newaxis,:]
        if near_boundary:
            # Pre-allocate additional diagonals for the boundary terms
            Dh_diag_p3 = np.zeros(f.shape)
            Dh_diag_m3 = np.zeros(f.shape)
            Dh_diag_m4 = np.zeros(f.shape)
            # At index one we expoloit that 0 = (d/dr)[ (1/r) (d/dr)[ r (dh/dr) ] ]
            D_p2 = 0.5*(D_half[ 0]+D_half[ 1])
            Dh_diag_p1[:,1]  =  0.5*r[2]*p*h[2]**(p-1)*f[:,2]*D_p2/dr**4
            Dh_diag_p3[:,1]  =  0.5*r[2]*h[2]**p*f[:,2]*0.5*(r_half[3]/r[3])/dr**4
            Dh_diag_p2[:,1]  = -0.5*r[2]*h[2]**p*f[:,2]/dr**4
            Dh_diag_p1[:,1] +=  0.5*r[2]*h[2]**p*f[:,2]*0.5*(r_half[2]/r[3]-r_half[1]/r[1])/dr**4  
            Dh_diag_p0[:,1]  = -0.5*r[2]*h[2]**p*f[:,2]/dr**4  
            Dh_diag_m1[:,1]  =  0.5*r[2]*h[2]**p*f[:,2]*0.5*(r_half[0]/r[1])/dr**4
            # At index -2 we can exploit that 0 = (dh/dr)
            # The width of the stencil is widened to achieve this though...
            D_m5o2 =  0.25*(r[-2]*(h[-1]-h[-3])-r[-4]*(h[-3]-h[-5]))/r[-3]\
                     -0.25*(r[-3]*(h[-2]-h[-4])-r[-5]*(h[-4]-h[-6]))/r[-4]
            D_m3o2 =  0.25*(                   -r[-3]*(h[-2]-h[-4]))/r[-2]\
                     -0.25*(r[-2]*(h[-1]-h[-3])-r[-4]*(h[-3]-h[-5]))/r[-3]
            Dh_diag_p1[:,-2] =  r_half[-1]*0.5*p*h_half[-1]**(p-1)*f_half[:,-1]*D_m3o2 
            Dh_diag_p0[:,-2] =  r_half[-1]*0.5*p*h_half[-1]**(p-1)*f_half[:,-1]*D_m3o2 \
                               -r_half[-2]*0.5*p*h_half[-2]**(p-1)*f_half[:,-2]*D_m5o2
            Dh_diag_m1[:,-2] = -r_half[-2]*0.5*p*h_half[-2]**(p-1)*f_half[:,-2]*D_m5o2
            # I think the following are okay...
            Dh_diag_p1[:,-2] +=  r_half[-1]*h_half[-1]**p*f_half[:,-1]*( r[-2]/r[-3])*0.25/dr**4 \
                                -r_half[-2]*h_half[-2]**p*f_half[:,-2]*(-r[-2]/r[-3])*0.25/dr**4
            Dh_diag_p0[:,-2] +=  r_half[-1]*h_half[-1]**p*f_half[:,-1]*(-r[-3]/r[-4])*0.25/dr**4 \
                                -r_half[-2]*h_half[-2]**p*f_half[:,-2]*(-r[-3]/r[-2])*0.25/dr**4
            Dh_diag_m1[:,-2] +=  r_half[-1]*h_half[-1]**p*f_half[:,-1]*(-2)*0.25/dr**4 \
                                -r_half[-2]*h_half[-2]**p*f_half[:,-2]*( 2)*0.25/dr**4
            Dh_diag_m2[:,-2]  =  r_half[-1]*h_half[-1]**p*f_half[:,-1]*( 2)*0.25/dr**4 \
                                -r_half[-2]*h_half[-2]**p*f_half[:,-2]*( r[-3]/r[-2])*0.25/dr**4
            Dh_diag_m3[:,-2]  =  r_half[-1]*h_half[-1]**p*f_half[:,-1]*( r[-4]/r[-3])*0.25/dr**4 \
                                -r_half[-2]*h_half[-2]**p*f_half[:,-2]*( r[-4]/r[-3])*0.25/dr**4
            Dh_diag_m4[:,-2]  =  r_half[-1]*h_half[-1]**p*f_half[:,-1]*(-r[-5]/r[-4])*0.25/dr**4
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
            if len(prefix.shape)==1:
                for diagonal in diagonals:
                    diagonal[:,1:-1] *= prefix[np.newaxis,1:-1]
            elif len(prefix.shape)==2:
                for diagonal in diagonals:
                    diagonal[:,1:-1] *= prefix[:,1:-1]
            # else do nothing...
        return diagonals,offsets
    def _Phi_n_equation_explicit(self,v_old,dt=None):
        """
        Calculate a simple forward Euler step of the Phi_n equations.
        (Here Phi_n = int_0^{h xi} phi_n dz, the input v_old,v_new must contain Phi_n rather than phi_n)
        The internal time step dt is used if one is not provided.
        Note: This is generally going to be unstable, however I have been able to 'get lucky' with some grid choices.
        """
        r = self._r
        nr = len(r)
        dr = r[1]
        R,XI = self._R,self._XI
        dxi = XI[1,1]
        gamma_ast = self._gamma_ast
        h_ast = self._h_ast
        Psi_d = self._Psi_d
        Psi_m = self._Psi_m
        lambda_ast = self._lambda_ast
        h_old,Phi_n_old,g_s_old,g_b_old = v_old
        if dt is None:
            dt = self._dt
        # Setup the vertical velocity factor
        # Note: the second line of each v_z terms do not include r=0 or r=R parts, they are not needed regardless
        #       The v_z terms also exclude the 1/h factor...
        fot_old = self._advective_term(r,h_old)
        v_z_old = (1.0+Psi_m)*g_b_old[np.newaxis,:]*(Phi_n_old-XI*(Phi_n_old[-1,:])[np.newaxis,:])
        v_z_old[:,1:-1] += gamma_ast/6.0*(2.0*XI+XI**3-3*XI**2)[:,1:-1]/R[:,1:-1]*(fot_old)[np.newaxis,1:-1]
        # Setup the horizontal 'advection' stencil
        Phi_int_dxi_old = np.cumsum(0.5*((Phi_n_old*(1-XI))[1:,:]+(Phi_n_old*(1-XI))[:-1,:]),axis=0)*dxi
        integral_old = np.empty(Phi_n_old.shape)
        integral_old[0 ,:] = 0
        integral_old[1:,:] = Phi_int_dxi_old
        f_old = (Phi_n_old*(0.5*XI**2-XI)+integral_old)*h_old[np.newaxis,:]**2
        if np.isfinite(lambda_ast):
            f_old -= Phi_n_old*h_old[np.newaxis,:]/lambda_ast
        adv_old = self._advective_term_alt(r,h_old,f_old)
        # Initialise the update with the forcing term
        delta_Phi_n  = Phi_n_old*(g_b_old-Psi_d)[np.newaxis,:]
        # Add the vertical advection part (note no flux through the top or bottom)
        delta_Phi_n[1:-1,1:-1] -= v_z_old[1:-1,1:-1]/h_old[np.newaxis,1:-1]*(Phi_n_old[2:,1:-1]-Phi_n_old[:-2,1:-1])/(2.0*dxi)
        # Add the horizontal 'advection' part 
        delta_Phi_n[:,1:-1] += gamma_ast/r[np.newaxis,1:-1]*adv_old[:,1:-1]
        # Perform the update
        Phi_n_new = Phi_n_old+dt*delta_Phi_n
        # Enforce the boundary conditions post update
        Phi_n_new[:,-1] = 0
        Phi_n_new[:, 0] = (4*Phi_n_new[:,1]-Phi_n_new[:,0])/3.0
        if self._use_artificial_dr_bc: # if artificial 'BC' is also enforced near r=0
            Phi_n_new[:,0] = 0.2*(9*Phi_n_new[:,2]-4*Phi_n_new[:,3])
            Phi_n_new[:,1] = 0.2*(8*Phi_n_new[:,2]-3*Phi_n_new[:,3])
        if False: # if high order BC enforcement at r=0
            Phi_n_new[:,0] = (18*Phi_n_new[:,1]-9*Phi_n_new[:,2]+2*Phi_n_new[:,3])/11.0
        if False: # if both high order BC enforcement at r=0 and additional artificial BC near r=0
            Phi_n_new[:,0] = 0.2*(9*Phi_n_new[:,2]-4*Phi_n_new[:,3]) # (note: it works out same as above...)
            Phi_n_new[:,1] = 0.2*(8*Phi_n_new[:,2]-3*Phi_n_new[:,3])
        Phi_n_new[ 0,:] = 0 # by definition
        #Phi_n_new[-1,:] = 2*Phi_n_new[-2,:]-Phi_n_new[-3,:] # need to do something here? maybe enforce d^2\Phi_n/d\xi^2=0
        Phi_n_new[-1,:] = 2.5*Phi_n_new[-2,:]-2*Phi_n_new[-3,:]+0.5*Phi_n_new[-4,:] # higher order...
        # Zero parts where h is still too small
        Phi_n_new[:,h_old<=h_ast] = 0
        # done
        return Phi_n_new
    def _Phi_n_equation_semi_implicit(self,v_old,dt=None,explicit_r_advection=False):
        """
        Calculate a simple backward Euler step of the Phi_n equations.
        (Here Phi_n = int_0^{h xi} phi_n dz, the input v_old,v_new must contain Phi_n rather than phi_n)
        The internal time step dt is used if one is not provided.
        Note: This is semi-implicit in the sense we linearise the equations to make it somewhat easier to implement.
        This currently works reasonably well in the current form...
        """
        r = self._r
        nr = len(r)
        dr = r[1]
        R,XI = self._R,self._XI
        nxi = len(self._xi)
        dxi = XI[1,1]
        gamma_ast = self._gamma_ast
        h_ast = self._h_ast
        Psi_d = self._Psi_d
        Psi_m = self._Psi_m
        lambda_ast = self._lambda_ast
        h_old,Phi_n_old,g_s_old,g_b_old = v_old
        if dt is None:
            dt = self._dt
        # Initialise the lhs matrix with ones on the main diagonal
        A_p0_p0 = np.ones(Phi_n_old.shape)
        # Initialise the rhs vector with the 'old' Phi_n
        rhs = Phi_n_old.copy()
        # Note: the xi=0 boundary condition should require no changes to the above (since Phi_n_old should be 0 on the bottom)
        rhs[0,:] = 0  # but we set it explicitly to be absolutely clear
        # Note: the same applies to the r=R boundary condition (where we enforce Phi_n=0 since h=b here)
        rhs[:,-1] = 0 # but we again set it explicitly to be absolutely clear
        # For the xi=1 boundary condition we implicitly make the 2nd derivative zero
        A_p0_m1 = np.zeros(Phi_n_old.shape)
        A_p0_m2 = np.zeros(Phi_n_old.shape)
        A_p0_m3 = np.zeros(Phi_n_old.shape) # required for higher order stencil
        A_p0_p0[-1,2:-1] =  2.0 #  1 low order,  2 higher order
        A_p0_m1[-1,2:-1] = -5.0 # -2 low order, -5 higher order
        A_p0_m2[-1,2:-1] =  4.0 #  1 low order,  4 higher order
        A_p0_m3[-1,2:-1] = -1.0 #               -1 higher order 
        rhs[-1,2:-1] = 0
        # Now the BC at r=0 (and the artificial one I enforce next to it)
        A_p1_p0 = np.zeros(Phi_n_old.shape)
        A_p2_p0 = np.zeros(Phi_n_old.shape)
        A_m1_p0 = np.zeros(Phi_n_old.shape) # required for the artificial BC
        A_p3_p0 = np.zeros(Phi_n_old.shape) # required for higher order stencil at r=0
        A_p0_p0[:,0] =  3.0;A_p1_p0[:,0] = -4.0;A_p2_p0[:,0] =  1.0
        rhs[:,0] = 0
        if self._use_artificial_dr_bc: # if artificial 'BC' is also enforced near r=0
            A_p0_p0[:,0] = 3.0;A_p1_p0[:,0] = -4.0;A_p2_p0[:,0] = 1.0
            A_m1_p0[:,1] = 4.0;A_p0_p0[:,1] = -7.0;A_p1_p0[:,1] = 4.0;A_p2_p0[:,1] = -1.0
            rhs[:,1] = 0
        if False: # if high order BC enforcement at r=0
            A_p0_p0[:,0] =  11.0;A_p1_p0[:,0] = -18.0;A_p2_p0[:,0] =   9.0;A_p3_p0[:,0] = - 2.0
        if False: # if both high order BC enforcement at r=0 and additional artificial BC near r=0
            A_p0_p0[:,0] = 11.0;A_p1_p0[:,0] = -18.0;A_p2_p0[:,0] = 9.0;A_p3_p0[:,0] = -2.0
            A_m1_p0[:,1] =  4.0;A_p0_p0[:,1] = - 7.0;A_p1_p0[:,1] = 4.0;A_p2_p0[:,1] = -1.0
            rhs[:,1] = 0
        # Add the forcing terms on the 'interior' (this need not be implicit really)
        A_p0_p0[1:-1,2:-1] += -dt*(g_b_old-Psi_d)[np.newaxis,2:-1] # implicit forcing...
        #rhs[1:-1,2:-1] += dt*Phi_n_old[1:-1,2:-1]*(g_b_old-Psi_d)[np.newaxis,2:-1] # explicit forcing...
        # Setup the vertical velocity factor
        # Note: the second line of each v_z terms do not include r=0 or r=R parts, they are not needed regardless
        #       The v_z terms also exclude the 1/h factor...
        fot_old = self._advective_term(r,h_old)
        v_z_old = (1.0+Psi_m)*g_b_old[np.newaxis,:]*(Phi_n_old-XI*(Phi_n_old[-1,:])[np.newaxis,:])
        v_z_old[:,1:-1] += gamma_ast/6.0*(2.0*XI+XI**3-3*XI**2)[:,1:-1]/R[:,1:-1]*(fot_old)[np.newaxis,1:-1]
        # Now add this to the appropriate diagonals...
        A_p0_p1 = np.zeros(Phi_n_old.shape)
        A_p0_m1[1:-1,2:-1] = -dt/(2*dxi)*v_z_old[1:-1,2:-1]/h_old[np.newaxis,2:-1] # central...
        A_p0_p1[1:-1,2:-1] = +dt/(2*dxi)*v_z_old[1:-1,2:-1]/h_old[np.newaxis,2:-1]
        #A_p0_m1[1:-1,2:-1] += -dt/dxi*(v_z_old[1:-1,2:-1]>0)*v_z_old[1:-1,2:-1]/h_old[np.newaxis,2:-1] # upwinded...
        #A_p0_p0[1:-1,2:-1] += +dt/dxi*(v_z_old[1:-1,2:-1]>0)*v_z_old[1:-1,2:-1]/h_old[np.newaxis,2:-1]
        #A_p0_p0[1:-1,2:-1] += -dt/dxi*(v_z_old[1:-1,2:-1]<0)*v_z_old[1:-1,2:-1]/h_old[np.newaxis,2:-1]
        #A_p0_p1[1:-1,2:-1] += +dt/dxi*(v_z_old[1:-1,2:-1]<0)*v_z_old[1:-1,2:-1]/h_old[np.newaxis,2:-1]
        
        
        # Setup the horizontal 'advection' stencil
        if explicit_r_advection: # true - explicit, false - implicit
            Phi_int_dxi_old = np.cumsum(0.5*((Phi_n_old*(1-XI))[1:,:]+(Phi_n_old*(1-XI))[:-1,:]),axis=0)*dxi
            integral_old = np.empty(Phi_n_old.shape)
            integral_old[0 ,:] = 0
            integral_old[1:,:] = Phi_int_dxi_old
            f_old = (Phi_n_old*(0.5*XI**2-XI)+integral_old)*h_old[np.newaxis,:]**2
            if np.isfinite(lambda_ast):
                f_old -= Phi_n_old*h_old[np.newaxis,:]/lambda_ast
            adv_old = self._advective_term_alt(r,h_old,f_old)
            # Add the horizontal 'advection' part to the system
            # Note: currently this is treated explicitly, which seems to work okay for the most part...
            rhs[1:-1,2:-1] += dt*gamma_ast*adv_old[1:-1,2:-1]/r[np.newaxis,2:-1]
        else:
            # Note: we can re-use the _advective_term_f_gradient function here
            diagonals_h2,offsets_h2 = self._advective_term_f_gradient(r,h_old,2,Phi_n_old)
            assert offsets_h2[0]==-1
            A_m1_p0[1:-1,2:-1] += -dt*gamma_ast*(0.5*XI**2-XI)[1:-1,2:-1]*diagonals_h2[0][np.newaxis,2:-1]/r[np.newaxis,2:-1]
            A_p0_p0[1:-1,2:-1] += -dt*gamma_ast*(0.5*XI**2-XI)[1:-1,2:-1]*diagonals_h2[1][np.newaxis,2:-1]/r[np.newaxis,2:-1]
            A_p1_p0[1:-1,2:-1] += -dt*gamma_ast*(0.5*XI**2-XI)[1:-1,2:-1]*diagonals_h2[2][np.newaxis,2:-1]/r[np.newaxis,2:-1]
            if np.isfinite(lambda_ast):
                diagonals_h1,offsets_h1 = self._advective_term_f_gradient(r,h_old,1,Phi_n_old)
                assert offsets_h1[0]==-1
                A_m1_p0[1:-1,2:-1] += +dt*gamma_ast/lambda_ast*diagonals_h1[0][np.newaxis,2:-1]/r[np.newaxis,2:-1]
                A_p0_p0[1:-1,2:-1] += +dt*gamma_ast/lambda_ast*diagonals_h1[1][np.newaxis,2:-1]/r[np.newaxis,2:-1]
                A_p1_p0[1:-1,2:-1] += +dt*gamma_ast/lambda_ast*diagonals_h1[2][np.newaxis,2:-1]/r[np.newaxis,2:-1]
            # Now add the integral component (note this is somewhat denser than usual)
            # (Note: it might be easier to build the entire matrix directly in coo format?)
            r_i,xi_i = np.meshgrid(range(nr),range(nxi))
            indices = xi_i*nr+r_i
            H = (h_old>h_ast) # Use this to zero out bits where h is too small...
            row,col,dat = [],[],[]
            for j in range(1,nxi-1): # exclude the first and last index... (the first is 0 regardless)
                for k in range(j+1):
                    c = 0.5*dxi if (k==0 or k==j) else dxi
                    row.append(indices[j,2:-1])
                    col.append(indices[k,1:-2])
                    dat.append(-dt*gamma_ast*c*(1-XI[k,2:-1])*H[2:-1]*diagonals_h2[0][2:-1]/r[2:-1])
                    row.append(indices[j,2:-1])
                    col.append(indices[k,2:-1])
                    dat.append(-dt*gamma_ast*c*(1-XI[k,2:-1])*H[2:-1]*diagonals_h2[1][2:-1]/r[2:-1])
                    row.append(indices[j,2:-1])
                    col.append(indices[k,3:  ])
                    dat.append(-dt*gamma_ast*c*(1-XI[k,2:-1])*H[2:-1]*diagonals_h2[2][2:-1]/r[2:-1])
            M_trap = coo_matrix((np.concatenate(dat),(np.concatenate(row),np.concatenate(col))),shape=(nr*nxi,nr*nxi))
            
        
        # Zero parts where h is still too small
        h_small = (h_old<=h_ast)
        A_p0_p0[:,h_small] = 1
        rhs[:,h_small] = 0
        A_m1_p0[:,h_small] = 0;A_p1_p0[:,h_small] = 0;A_p2_p0[:,h_small] = 0;#A_p3_p0[:,h_small] = 0;
        A_p0_m3[:,h_small] = 0;A_p0_m2[:,h_small] = 0;A_p0_m1[:,h_small] = 0;A_p0_p1[:,h_small] = 0;
        # Now setup the sparse linear system...
        if explicit_r_advection:
            A_11 = diags([A_p0_p0.ravel(),
                          A_m1_p0.ravel()[1:],A_p1_p0.ravel()[:-1],A_p2_p0.ravel()[:-2],#A_p3_p0.ravel()[:-3],
                          A_p0_m3.ravel()[3*nr:],A_p0_m2.ravel()[2*nr:],A_p0_m1.ravel()[nr:],A_p0_p1.ravel()[:-nr]],
                         [0,
                          -1,1,2,#3,
                          -3*nr,-2*nr,-nr,nr],
                         format="csr")
        else:
            A_11_partial = diags([A_p0_p0.ravel(),
                                  A_m1_p0.ravel()[1:],A_p1_p0.ravel()[:-1],A_p2_p0.ravel()[:-2],#A_p3_p0.ravel()[:-3],
                                  A_p0_m3.ravel()[3*nr:],A_p0_m2.ravel()[2*nr:],A_p0_m1.ravel()[nr:],A_p0_p1.ravel()[:-nr]],
                                 [0,
                                  -1,1,2,#3,
                                  -3*nr,-2*nr,-nr,nr],
                                 format="coo")
            A_11 = (A_11_partial+M_trap).tocsr()
        # Now solve the sparse linear system...
        Phi_n_new = spsolve(A_11,rhs.ravel()).reshape(Phi_n_old.shape)
        # done
        return Phi_n_new
    def _Phi_n_equation_RHS(self,v_old,v_new,dt=None):
        """
        Calculate the RHS vector component corresponding to the Phi_n equation.
        (Here Phi_n = int_0^{h xi} phi_n dz, the input v_old,v_new must contain Phi_n rather than phi_n)
        The internal time step dt is used if one is not provided.
        """
        r,xi = self._r,self._xi
        nr,nxi = len(r),len(xi)
        dr,dxi = r[1],xi[1]
        R,XI = self._R,self._XI
        r_half = self._r_half
        h_ast = self._h_ast
        gamma_ast = self._gamma_ast
        Psi_d = self._Psi_d
        Psi_m = self._Psi_m
        lambda_ast = self._lambda_ast
        h_old,Phi_n_old,g_s_old,g_b_old = v_old
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        # Some extra fields for convenience
        H_old,H_new = (h_old>h_ast),(h_new>h_ast) # Use this to zero out bits where h is too small...
        Hor_old,Hor_new = H_old[2:-1]/r[2:-1],H_new[2:-1]/r[2:-1]
        # Setup the rhs field and initialise on the interior with the difference in the fields
        rhs = np.zeros(Phi_n_new.shape)
        rhs[1:-1,2:-1] = -(Phi_n_new[1:-1,2:-1]-Phi_n_old[1:-1,2:-1]*H_old[np.newaxis,2:-1])
        # Note: the H_old in the above line should ensure that delta_Phi will be 0 where-ever h remains small
        #       (although it should be redundant since Phi_n_old should be zero here regardless)
        # Add the forcing term
        rhs[1:-1,2:-1] += 0.5*dt*( Phi_n_old[1:-1,2:-1]*(H_old*(g_b_old-Psi_d))[np.newaxis,2:-1]\
                                  +Phi_n_new[1:-1,2:-1]*(H_new*(g_b_new-Psi_d))[np.newaxis,2:-1]) 
        
        
        
        # Setup the vertical velocity factor
        # Note: the second line of each v_z terms do not include r=0 or r=R parts, they are not needed regardless
        #       The v_z terms also exclude the 1/h factor...
        fot_new = self._advective_term(r,h_new)
        fot_old = self._advective_term(r,h_old)
        #v_z_old = (1.0+Psi_m)*g_b_old[np.newaxis,:]*(Phi_n_old-XI*(Phi_n_old[-1,:])[np.newaxis,:])\
        #          +gamma_ast/6.0*(2.0*XI+XI**3-3*XI**2)/R*(fot_old)[np.newaxis,:]
        v_z_old = (1.0+Psi_m)*g_b_old[np.newaxis,:]*(Phi_n_old-XI*(Phi_n_old[-1,:])[np.newaxis,:])
        v_z_old[:,1:-1] += gamma_ast/6.0*(2.0*XI+XI**3-3*XI**2)[:,1:-1]/R[:,1:-1]*fot_old[np.newaxis,1:-1]
        #v_z_new = +(1.0+Psi_m)*g_b_new[np.newaxis,:]*(Phi_n_new-XI*(Phi_n_new[-1,:])[np.newaxis,:])\
        #          +gamma_ast/6.0*(2.0*XI+XI**3-3*XI**2)/R*fot_new[np.newaxis,:]
        v_z_new = (1.0+Psi_m)*g_b_new[np.newaxis,:]*(Phi_n_new-XI*(Phi_n_new[-1,:])[np.newaxis,:])
        v_z_new[:,1:-1] += gamma_ast/6.0*(2.0*XI+XI**3-3*XI**2)[:,1:-1]/R[:,1:-1]*fot_new[np.newaxis,1:-1]
        # Add the vertical advection part (note no flux through the top or bottom...
        rhs[1:-1,2:-1] -= 0.25*dt/dxi*( v_z_old[1:-1,2:-1]*(Phi_n_old[2:,2:-1]-Phi_n_old[:-2,2:-1])*(H_old/h_old)[np.newaxis,2:-1]\
                                       +v_z_new[1:-1,2:-1]*(Phi_n_new[2:,2:-1]-Phi_n_new[:-2,2:-1])*(H_new/h_new)[np.newaxis,2:-1])
        # Setup the horizontal 'advection' stencil
        Phi_int_dxi_old = np.cumsum(0.5*((Phi_n_old*(1-XI))[1:,:]+(Phi_n_old*(1-XI))[:-1,:]),axis=0)*dxi
        Phi_int_dxi_new = np.cumsum(0.5*((Phi_n_new*(1-XI))[1:,:]+(Phi_n_new*(1-XI))[:-1,:]),axis=0)*dxi
        integral_old = np.empty(Phi_n_old.shape)
        integral_old[0 ,:] = 0
        integral_old[1:,:] = Phi_int_dxi_old
        integral_new = np.empty(Phi_n_new.shape)
        integral_new[0 ,:] = 0
        integral_new[1:,:] = Phi_int_dxi_new
        f_old = (Phi_n_old*(0.5*XI**2-XI)+integral_old)*h_old[np.newaxis,:]**2
        f_new = (Phi_n_new*(0.5*XI**2-XI)+integral_new)*h_new[np.newaxis,:]**2
        if np.isfinite(lambda_ast):
            f_old -= Phi_n_old*h_old[np.newaxis,:]/lambda_ast
            f_new -= Phi_n_new*h_new[np.newaxis,:]/lambda_ast
        adv_new = self._advective_term_alt(r,h_new,f_new)
        adv_old = self._advective_term_alt(r,h_old,f_old)
        # Add the horizontal 'advection' part 
        rhs[1:-1,2:-1] += 0.5*dt*gamma_ast*( adv_new[1:-1,2:-1]*Hor_new[np.newaxis,:]
                                            +adv_old[1:-1,2:-1]*Hor_old[np.newaxis,:])
        
        
        
        # Set all of the entries relating to boundary conditions
        # Set the RHS corresponding to the \xi=0 boundary condition (delta_Phi+Phi_n_new)=0
        rhs[0,2:-1] = -Phi_n_new[0,2:-1]
        # Set the RHS corresponding to the r=R boundary condition (delta_Phi+Phi_n_new)=0 (since h=b~0 is enforced)
        rhs[:,  -1] = -Phi_n_new[:,  -1]
        if self._add_top_Phi_bc:
            # Set the RHS corresponding to the \xi=1 boundary condition d^2/dr^2(delta_Phi+Phi_n_new)=0
            rhs[-1,2:-1] = -2*Phi_n_new[-1,2:-1]+5*Phi_n_new[-2,2:-1]-4*Phi_n_new[-3,2:-1]+Phi_n_new[-4,2:-1]
        else:
            # Implement the discretisation of the horizontal advection
            rhs[-1,2:-1] = -(Phi_n_new[-1,2:-1]-Phi_n_old[-1,2:-1]*H_old[2:-1])\
                           +0.5*dt*gamma_ast*(adv_new[-1,2:-1]*Hor_new+adv_old[-1,2:-1]*Hor_old)\
                           +0.5*dt*( Phi_n_old[-1,2:-1]*(H_old*(g_b_old-Psi_d))[2:-1]\
                                    +Phi_n_new[-1,2:-1]*(H_new*(g_b_new-Psi_d))[2:-1]) 
        # Set the RHS corresponding to the r=0 boundary condition d/dr(delta_Phi+Phi_n_new)=0
        rhs[:, 0] = -3.0*Phi_n_new[:,0]+4.0*Phi_n_new[:,1]-Phi_n_new[:,2] 
        if False: # optional, higher order stencil
            rhs[:, 0] = -11*Phi_n_new[:,0]+18*Phi_n_new[:,1]-9*Phi_n_new[:,2]+2*Phi_n_new[:,3]
        if self._use_artificial_dr_bc:
            # Set the RHS corresponding to the introduced r=dr condition Phi(dr)=Phi(0)+0.5*dr^2*Phi''(0)
            rhs[:, 1] = 4*Phi_n_new[:,0]-7*Phi_n_new[:,1]+4*Phi_n_new[:,2]-Phi_n_new[:,3]
        # done
        return rhs.ravel()
    def _Phi_n_equation_LHS0(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        h dependence in the Phi_n equation.
        (Here Phi_n = int_0^{h xi} phi_n dz, the input v_old,v_new must contain Phi_n rather than phi_n)
        """
        r,xi = self._r,self._xi
        nr,nxi = len(r),len(xi)
        dr,dxi = r[1],xi[1]
        R,XI = self._R,self._XI
        r_half = self._r_half
        h_ast = self._h_ast
        gamma_ast = self._gamma_ast
        Psi_m = self._Psi_m
        lambda_ast = self._lambda_ast
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        # Note this block has rectangular shape
        # Setup some index arrays for constructing the matrix in coo format
        r_i,xi_i = np.meshgrid(range(nr),range(nxi))
        indices = xi_i*nr+r_i
        row,col,dat = [],[],[]
        H = (h_new>h_ast) # Use this to zero out bits where h is too small...
        Hor = H[1:-1]/r[1:-1]
        # Setup the vertical advection components first
        # Do the easier part first
        fot_new = self._advective_term(r,h_new)
        #v_z_new = +(1.0+Psi_m)*g_b_new[np.newaxis,:]*(Phi_n_new-XI*(Phi_n_new[-1,:])[np.newaxis,:])\
        #          +gamma_ast/6.0*(2.0*XI+XI**3-3*XI**2)/R*fot_new[np.newaxis,:]
        v_z_new = (1.0+Psi_m)*g_b_new[np.newaxis,:]*(Phi_n_new-XI*(Phi_n_new[-1,:])[np.newaxis,:])
        v_z_new[:,1:-1] += gamma_ast/6.0*(2.0*XI+XI**3-3*XI**2)[:,1:-1]/R[:,1:-1]*fot_new[np.newaxis,1:-1]
        xi_adv_term1 = -0.25*dt/dxi*v_z_new[1:-1,:]*(Phi_n_new[2:,:]-Phi_n_new[:-2,:])*(H/h_new**2)[np.newaxis,:]
        row.append(indices[1:-1,1:-1].ravel())
        col.append(r_i[1:-1,1:-1].ravel())
        dat.append(xi_adv_term1[:,1:-1].ravel())
        if self._use_artificial_dr_bc:
            #xi_adv_term1[:,1] = 0 # Need to modify this in conjunction with the 'artificial' BC at r=dr
            row[-1] = indices[1:-1,2:-1].ravel()
            col[-1] = r_i[1:-1,2:-1].ravel()
            dat[-1] = xi_adv_term1[:,2:-1].ravel()
        # Now the more difficult/involved part...
        # First get diagonals relating to the fourth order h term
        diagonals_h3,offsets_h3 = self._advective_term_h_gradient(r,h_new,3)
        if self._use_artificial_dr_bc:
            # Need to modify diagonals in conjunction with the 'artificial' BC at r=dr
            for k in range(len(diagonals_h3)):
                diagonals_h3[k][1] = 0
        # now construct the 2D factor and then add the diagonals to the matrix
        twoD_factor = 0.25*dt/dxi*gamma_ast*(XI**3-3*XI**2+2*XI)[1:-1,1:-1]*(Phi_n_new[2:,1:-1]-Phi_n_new[:-2,1:-1])\
                                           *H[np.newaxis,1:-1]/(6.0*r*h_new)[np.newaxis,1:-1]
        diag_h3_m4_dat = diagonals_h3[0][np.newaxis,4:-1]*twoD_factor[:,3:]
        row.append(indices[1:-1,4:-1].ravel())
        col.append(r_i[1:-1,0:-5].ravel())
        dat.append(diag_h3_m4_dat.ravel())
        diag_h3_m3_dat = diagonals_h3[1][np.newaxis,3:-1]*twoD_factor[:,2:]
        row.append(indices[1:-1,3:-1].ravel())
        col.append(r_i[1:-1,0:-4].ravel())
        dat.append(diag_h3_m3_dat.ravel())
        diag_h3_m2_dat = diagonals_h3[2][np.newaxis,2:-1]*twoD_factor[:,1:]
        row.append(indices[1:-1,2:-1].ravel())
        col.append(r_i[1:-1,0:-3].ravel())
        dat.append(diag_h3_m2_dat.ravel())
        diag_h3_m1_dat = diagonals_h3[3][np.newaxis,1:-1]*twoD_factor[:,:]
        row.append(indices[1:-1,1:-1].ravel())
        col.append(r_i[1:-1,0:-2].ravel())
        dat.append(diag_h3_m1_dat.ravel())
        diag_h3_p0_dat = diagonals_h3[4][np.newaxis,1:-1]*twoD_factor[:,:]
        row.append(indices[1:-1,1:-1].ravel())
        col.append(r_i[1:-1,1:-1].ravel())
        dat.append(diag_h3_p0_dat.ravel())
        diag_h3_p1_dat = diagonals_h3[5][np.newaxis,1:-1]*twoD_factor[:,:]
        row.append(indices[1:-1,1:-1].ravel())
        col.append(r_i[1:-1,2:].ravel())
        dat.append(diag_h3_p1_dat.ravel())
        diag_h3_p2_dat = diagonals_h3[6][np.newaxis,1:-2]*twoD_factor[:,:-1]
        row.append(indices[1:-1,1:-2].ravel())
        col.append(r_i[1:-1,3:].ravel())
        dat.append(diag_h3_p2_dat.ravel())
        diag_h3_p3_dat = diagonals_h3[7][np.newaxis,1:-3]*twoD_factor[:,:-2]
        row.append(indices[1:-1,1:-3].ravel())
        col.append(r_i[1:-1,4:].ravel())
        dat.append(diag_h3_p3_dat.ravel())
        # Now we need to do the radial 'advective' term
        # First get diagonals relating to the fourth order h term
        Phi_int_dxi_new = np.cumsum(0.5*((Phi_n_new*(1-XI))[1:,:]+(Phi_n_new*(1-XI))[:-1,:]),axis=0)*dxi
        h2_factor = Phi_n_new*(0.5*XI**2-XI)
        h2_factor[1:,:] += Phi_int_dxi_new
        h2_prefix = np.zeros(nr)
        h2_prefix[1:-1] = -0.5*dt*gamma_ast*Hor
        diagonals_h2,offsets_h2 = self._advective_term_h_gradient_alt(r,h_new,2,h2_factor,True,h2_prefix)
        if np.isfinite(lambda_ast):
            h1_prefix = np.zeros(nr)
            h1_prefix[1:-1] = 0.5*dt*gamma_ast/lambda_ast*Hor
            diagonals_h1,offsets_h1 = self._advective_term_h_gradient_alt(r,h_new,1,Phi_n_new,True,h1_prefix)
            for k in range(len(diagonals_h2)):
                diagonals_h2[k] += diagonals_h1[k]
        if self._use_artificial_dr_bc:
            # Need to modify diagonals in conjunction with the 'artificial' BC at r=dr
            for k in range(len(diagonals_h2)):
                diagonals_h2[k][:,1] = 0
        diag_h2_m4_dat = diagonals_h2[0][1:-1,4:-1]
        row.append(indices[1:-1,4:-1].ravel())
        col.append(r_i[1:-1,0:-5].ravel())
        dat.append(diag_h2_m4_dat.ravel())
        diag_h2_m3_dat = diagonals_h2[1][1:-1,3:-1]
        row.append(indices[1:-1,3:-1].ravel())
        col.append(r_i[1:-1,0:-4].ravel())
        dat.append(diag_h2_m3_dat.ravel())
        diag_h2_m2_dat = diagonals_h2[2][1:-1,2:-1]
        row.append(indices[1:-1,2:-1].ravel())
        col.append(r_i[1:-1,0:-3].ravel())
        dat.append(diag_h2_m2_dat.ravel())
        diag_h2_m1_dat = diagonals_h2[3][1:-1,1:-1]
        row.append(indices[1:-1,1:-1].ravel())
        col.append(r_i[1:-1,0:-2].ravel())
        dat.append(diag_h2_m1_dat.ravel())
        diag_h2_p0_dat = diagonals_h2[4][1:-1,1:-1]
        row.append(indices[1:-1,1:-1].ravel())
        col.append(r_i[1:-1,1:-1].ravel())
        dat.append(diag_h2_p0_dat.ravel())
        diag_h2_p1_dat = diagonals_h2[5][1:-1,1:-1]
        row.append(indices[1:-1,1:-1].ravel())
        col.append(r_i[1:-1,2:].ravel())
        dat.append(diag_h2_p1_dat.ravel())
        diag_h2_p2_dat = diagonals_h2[6][1:-1,1:-2]
        row.append(indices[1:-1,1:-2].ravel())
        col.append(r_i[1:-1,3:].ravel())
        dat.append(diag_h2_p2_dat.ravel())
        diag_h2_p3_dat = diagonals_h2[7][1:-1,1:-3]
        row.append(indices[1:-1,1:-3].ravel())
        col.append(r_i[1:-1,4:].ravel())
        dat.append(diag_h2_p3_dat.ravel())
        if not self._add_top_Phi_bc:
            row.append(indices[-1,4:-1].ravel())
            col.append(r_i[-1,0:-5].ravel())
            dat.append(diagonals_h2[0][-1,4:-1].ravel())
            row.append(indices[-1,3:-1].ravel())
            col.append(r_i[-1,0:-4].ravel())
            dat.append(diagonals_h2[1][-1,3:-1].ravel())
            row.append(indices[-1,2:-1].ravel())
            col.append(r_i[-1,0:-3].ravel())
            dat.append(diagonals_h2[2][-1,2:-1].ravel())
            row.append(indices[-1,1:-1].ravel())
            col.append(r_i[-1,0:-2].ravel())
            dat.append(diagonals_h2[3][-1,1:-1].ravel())
            row.append(indices[-1,1:-1].ravel())
            col.append(r_i[-1,1:-1].ravel())
            dat.append(diagonals_h2[4][-1,1:-1].ravel())
            row.append(indices[-1,1:-1].ravel())
            col.append(r_i[-1,2:].ravel())
            dat.append(diagonals_h2[5][-1,1:-1].ravel())
            row.append(indices[-1,1:-2].ravel())
            col.append(r_i[-1,3:].ravel())
            dat.append(diagonals_h2[6][-1,1:-2].ravel())
            row.append(indices[-1,1:-3].ravel())
            col.append(r_i[-1,4:].ravel())
            dat.append(diagonals_h2[7][-1,1:-3].ravel())
        # done, construct and return
        return coo_matrix((np.concatenate(dat),(np.concatenate(row),np.concatenate(col))),shape=(nr*nxi,nr))#.tocsr()
    def _Phi_n_equation_LHS1(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        phi_n dependence in the Phi_n equation.
        (Here Phi_n = int_0^{h xi} phi_n dz, the input v_old,v_new must contain Phi_n in place of phi_n)
        """
        r,xi = self._r,self._xi
        nr,nxi = len(r),len(xi)
        dr,dxi = r[1],xi[1]
        R,XI = self._R,self._XI
        r_half = self._r_half
        h_ast = self._h_ast
        gamma_ast = self._gamma_ast
        Psi_d = self._Psi_d
        Psi_m = self._Psi_m
        lambda_ast = self._lambda_ast
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        # Setup some index arrays for constructing the matrix in coo format
        r_i,xi_i = np.meshgrid(range(nr),range(nxi))
        indices = xi_i*nr+r_i
        H = (h_new>h_ast) # Use this to zero out bits where h is too small...
        Hor = H[2:-1]/r[2:-1]
        A_p0_p0 = np.ones(Phi_n_new.shape)
        row,col,dat = [indices.ravel()],[indices.ravel()],[A_p0_p0.ravel()] # initialise with a view of A_p0_p0
        # We start by filling out the interior stencils
        # Add the forcing term to the main diagonal
        A_p0_p0[1:-1,2:-1] += -0.5*dt*H[np.newaxis,2:-1]*(g_b_new-Psi_d)[np.newaxis,2:-1] 
        # Add the simple non-linear component of the vertical advection term
        A_p0_p0[1:-1,2:-1] += +0.25*dt/dxi*(1+Psi_m)*(H*g_b_new/h_new)[np.newaxis,2:-1]*(Phi_n_new[2:,2:-1]-Phi_n_new[:-2,2:-1])
        # Add the other non-linear component of the vertical advection term
        #for k in range(1,nxi-1): # exclude the two ends
        #    row.append(indices[k,2:-1])
        #    col.append(indices[-1,2:-1])
        #    dat.append(-0.25*dt/dxi*(1+Psi_m)*(XI[k]*H*g_b_new/h)[2:-1]*(Phi_n_new[k+1,2:-1]-Phi_n_new[k-1,2:-1]))
        row.append(indices[1:-1,2:-1].ravel())
        col.append(np.broadcast_to(indices[-1,2:-1],(nxi-2,nr-3)).ravel())
        dat.append((-0.25*dt/dxi*(1+Psi_m)*(H*g_b_new/h_new)[np.newaxis,2:-1]*XI[1:-1,2:-1]\
                                *(Phi_n_new[2:,2:-1]-Phi_n_new[:-2,2:-1])).ravel())
        # Add the remaining vertical advection term
        fot_new = self._advective_term(r,h_new)
        #v_z_new = +(1.0+Psi_m)*g_b_new[np.newaxis,:]*(Phi_n_new-XI*(Phi_n_new[-1,:])[np.newaxis,:])\
        #          +gamma_ast/6.0*(2.0*XI+XI**3-3*XI**2)/R*fot_new[np.newaxis,:]
        v_z_new = (1.0+Psi_m)*g_b_new[np.newaxis,:]*(Phi_n_new-XI*(Phi_n_new[-1,:])[np.newaxis,:])
        v_z_new[:,1:-1] += gamma_ast/6.0*(2.0*XI+XI**3-3*XI**2)[:,1:-1]/R[:,1:-1]*fot_new[np.newaxis,1:-1]
        A_p0_m1 = np.zeros(Phi_n_new.shape)
        A_p0_p1 = np.zeros(Phi_n_new.shape)
        A_p0_m1[1:-1,2:-1] = -0.5*dt/(2.0*dxi)*H[np.newaxis,2:-1]*v_z_new[1:-1,2:-1]/h_new[np.newaxis,2:-1]
        A_p0_p1[1:-1,2:-1] = +0.5*dt/(2.0*dxi)*H[np.newaxis,2:-1]*v_z_new[1:-1,2:-1]/h_new[np.newaxis,2:-1]
        row.append(indices[1:-1,2:-1].ravel());row.append(indices[1:-1,2:-1].ravel())
        col.append(indices[2:  ,2:-1].ravel());col.append(indices[ :-2,2:-1].ravel())
        dat.append(A_p0_p1[1:-1,2:-1].ravel());dat.append(A_p0_m1[1:-1,2:-1].ravel())
        # Add the radial 'advective' terms
        # Note: we can re-use the self._advective_term_f_gradient function here
        A_m1_p0 = np.zeros(Phi_n_new.shape)
        A_p1_p0 = np.zeros(Phi_n_new.shape)
        diagonals_h2,offsets_h2 = self._advective_term_f_gradient(r,h_new,2,Phi_n_new)
        assert offsets_h2[0]==-1
        A_m1_p0[1:-1,2:-1] += -0.5*dt*gamma_ast*(0.5*XI**2-XI)[1:-1,2:-1]*Hor[np.newaxis,:]*diagonals_h2[0][np.newaxis,2:-1]
        A_p0_p0[1:-1,2:-1] += -0.5*dt*gamma_ast*(0.5*XI**2-XI)[1:-1,2:-1]*Hor[np.newaxis,:]*diagonals_h2[1][np.newaxis,2:-1]
        A_p1_p0[1:-1,2:-1] += -0.5*dt*gamma_ast*(0.5*XI**2-XI)[1:-1,2:-1]*Hor[np.newaxis,:]*diagonals_h2[2][np.newaxis,2:-1]
        if np.isfinite(lambda_ast):
            diagonals_h1,offsets_h1 = self._advective_term_f_gradient(r,h_new,1,Phi_n_new)
            assert offsets_h1[0]==-1
            A_m1_p0[1:-1,2:-1] += +0.5*dt*gamma_ast/lambda_ast*Hor[np.newaxis,:]*diagonals_h1[0][np.newaxis,2:-1]
            A_p0_p0[1:-1,2:-1] += +0.5*dt*gamma_ast/lambda_ast*Hor[np.newaxis,:]*diagonals_h1[1][np.newaxis,2:-1]
            A_p1_p0[1:-1,2:-1] += +0.5*dt*gamma_ast/lambda_ast*Hor[np.newaxis,:]*diagonals_h1[2][np.newaxis,2:-1]
        row.append(indices[1:-1,2:-1].ravel());row.append(indices[1:-1,2:-1].ravel())
        col.append(indices[1:-1,3:  ].ravel());col.append(indices[1:-1,1:-2].ravel())
        dat.append(A_p1_p0[1:-1,2:-1].ravel());dat.append(A_m1_p0[1:-1,2:-1].ravel())
        # Now add the integral component (note this is somewhat denser than other components)
        for j in range(1,nxi-1): # exclude the first and last index... (the first is 0 regardless)
            for k in range(j+1):
                c = 0.5*dxi if (k==0 or k==j) else dxi
                row.append(indices[j,2:-1])
                col.append(indices[k,1:-2])
                dat.append(-0.5*dt*gamma_ast*c*(1-XI[k,2:-1])*Hor*diagonals_h2[0][2:-1])
                row.append(indices[j,2:-1])
                col.append(indices[k,2:-1])
                dat.append(-0.5*dt*gamma_ast*c*(1-XI[k,2:-1])*Hor*diagonals_h2[1][2:-1])
                row.append(indices[j,2:-1])
                col.append(indices[k,3:  ])
                dat.append(-0.5*dt*gamma_ast*c*(1-XI[k,2:-1])*Hor*diagonals_h2[2][2:-1])
        # Now we need to enforce the boundary conditions...
        # When \xi=0 then we want (delta_Phi+Phi_n_new) = 0 ==> delta_Phi = -Phi_n_new 
        #   ==> so ones on the main diagonal is fine, and rhs needs to be set accordingly.
        # When r=R then we want (delta_Phi+Phi_n_new) = 0 ==> delta_Phi = -Phi_n_new 
        #   ==> so ones on the main diagonal is fine, and rhs needs to be set accordingly.
        if self._add_top_Phi_bc:
            # When \xi=1 then we want d^2/d\xi^2(delta_Phi+Phi_n_new) = 0
            #   ==> stencil 1,-2,_1_ for 1st order, -1,4,-5,_2_ for second order
            A_p0_xi1m1 = np.empty(nr)
            A_p0_xi1m2 = np.empty(nr)
            A_p0_xi1m3 = np.empty(nr)
            A_p0_p0[-1,2:-1] =  2.0 #  1 low order,  2 higher order
            A_p0_xi1m1[2:-1] = -5.0 # -2 low order, -5 higher order
            A_p0_xi1m2[2:-1] =  4.0 #  1 low order,  4 higher order
            A_p0_xi1m3[2:-1] = -1.0 #               -1 higher order 
            row.append(indices[-1,2:-1]);row.append(indices[-1,2:-1]);row.append(indices[-1,2:-1])
            col.append(indices[-2,2:-1]);col.append(indices[-3,2:-1]);col.append(indices[-4,2:-1])
            dat.append(A_p0_xi1m1[2:-1]);dat.append(A_p0_xi1m2[2:-1]);dat.append(A_p0_xi1m3[2:-1])
        else:
            # Implement the discretisation of the horizontal advection
            A_p0_p0[-1,2:-1] = 1.0-0.5*dt*H[2:-1]*(g_b_new-Psi_d)[2:-1] 
            # the radial advective part... (could really just sub XI=1 here...)
            A_m1_p0[-1,2:-1] += -0.5*dt*gamma_ast*(0.5*XI**2-XI)[-1,2:-1]*Hor*diagonals_h2[0][2:-1]
            A_p0_p0[-1,2:-1] += -0.5*dt*gamma_ast*(0.5*XI**2-XI)[-1,2:-1]*Hor*diagonals_h2[1][2:-1]
            A_p1_p0[-1,2:-1] += -0.5*dt*gamma_ast*(0.5*XI**2-XI)[-1,2:-1]*Hor*diagonals_h2[2][2:-1]
            if np.isfinite(lambda_ast):
                #diagonals_h1,offsets_h1 = self._advective_term_f_gradient(r,h_new,1,Phi_n_new) # should already exist...
                #assert offsets_h1[0]==-1
                A_m1_p0[-1,2:-1] += +0.5*dt*gamma_ast/lambda_ast*Hor*diagonals_h1[0][2:-1]
                A_p0_p0[-1,2:-1] += +0.5*dt*gamma_ast/lambda_ast*Hor*diagonals_h1[1][2:-1]
                A_p1_p0[-1,2:-1] += +0.5*dt*gamma_ast/lambda_ast*Hor*diagonals_h1[2][2:-1]
            row.append(indices[-1,2:-1].ravel());row.append(indices[-1,2:-1].ravel())
            col.append(indices[-1,3:  ].ravel());col.append(indices[-1,1:-2].ravel())
            dat.append(A_p1_p0[-1,2:-1].ravel());dat.append(A_m1_p0[-1,2:-1].ravel())
            # Now add the integral component (note this is somewhat denser than other components)
            j = nxi-1
            for k in range(j+1):
                c = 0.5*dxi if (k==0 or k==j) else dxi
                row.append(indices[j,2:-1])
                col.append(indices[k,1:-2])
                dat.append(-0.5*dt*gamma_ast*c*(1-XI[k,2:-1])*Hor*diagonals_h2[0][2:-1])
                row.append(indices[j,2:-1])
                col.append(indices[k,2:-1])
                dat.append(-0.5*dt*gamma_ast*c*(1-XI[k,2:-1])*Hor*diagonals_h2[1][2:-1])
                row.append(indices[j,2:-1])
                col.append(indices[k,3:  ])
                dat.append(-0.5*dt*gamma_ast*c*(1-XI[k,2:-1])*Hor*diagonals_h2[2][2:-1])
        # When r=0 we want to enforce d/dr(delta_Phi+Phi_n_new) = 0
        #   ==> stencil _3_,-4,1 for second order
        A_r0p1_p0 = np.empty(nxi)
        A_r0p2_p0 = np.empty(nxi)
        A_p0_p0[:,0] =  3.0
        A_r0p1_p0[:] = -4.0
        A_r0p2_p0[:] =  1.0
        row.append(indices[:,0]);row.append(indices[:,0])
        col.append(indices[:,1]);col.append(indices[:,2])
        dat.append(A_r0p1_p0);   dat.append(A_r0p2_p0)
        if False:
            # Coud implement optional higher order stencil for r=0 here, but not really needed...
            pass
        if self._use_artificial_dr_bc:
            # When r=dr we also enforce Phi(dr)=Phi(0)+0.5*dr^2*Phi''(0) (to smooth things out a bit here)
            #   ==> stencil -4,_7_,-4,1 (derived from a forward 2nd order stencil for Phi''(0))
            A_r1m1_p0 = np.empty(nxi)
            A_r1p1_p0 = np.empty(nxi)
            A_r1p2_p0 = np.empty(nxi)
            A_r1m1_p0[:] = -4.0
            A_p0_p0[:,1] =  7.0
            A_r1p1_p0[:] = -4.0
            A_r1p2_p0[:] =  1.0
            row.append(indices[:,1]);row.append(indices[:,1]);row.append(indices[:,1])
            col.append(indices[:,0]);col.append(indices[:,2]);col.append(indices[:,3])
            dat.append(A_r1m1_p0);   dat.append(A_r1p1_p0);   dat.append(A_r1p2_p0)
        # Final constructions
        A_11 = coo_matrix((np.concatenate(dat),(np.concatenate(row),np.concatenate(col))),shape=(nr*nxi,nr*nxi))
        # done
        return A_11
    def _Phi_n_equation_LHS2(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        g_s dependence in the Phi_n equation.
        (Here Phi_n = int_0^{h xi} phi_n dz, the input v_new must contain Phi_n rather than phi_n)
        """
        # Note: there is no dependence on g_s
        return None 
    def _Phi_n_equation_LHS3(self,v_new,dt=None):
        """
        Calculate the LHS matrix block corresponding to the 
        g_b dependence in the Phi_n equation.
        (Here Phi_n = int_0^{h xi} phi_n dz, the input v_new must contain Phi_n rather than phi_n)
        """
        nr,nxi = len(self._r),len(self._xi)
        XI = self._XI
        dxi = XI[1,1]
        Psi_m = self._Psi_m
        h_new,Phi_n_new,g_s_new,g_b_new = v_new
        if dt is None:
            dt = self._dt
        H = (h_new>self._h_ast) # Use this to zero out bits where h is too small...
        # Note: This block has a rectangular shape
        r_i,xi_i = np.meshgrid(np.arange(nr),np.arange(nxi))
        indices = r_i+nr*xi_i
        row,col,dat = [],[],[]
        # First add the component coming from the forcing term
        forcing_term = -0.5*dt*Phi_n_new*H
        row.append(indices[1:-1,1:-1].ravel())
        col.append(r_i[1:-1,1:-1].ravel())
        dat.append(forcing_term[1:-1,1:-1].ravel())
        if self._use_artificial_dr_bc:
            #forcing_term[:,1] = 0 # Need to modify this in conjunction with the 'artificial' BC at r=dr
            row[-1] = indices[1:-1,2:-1].ravel()
            col[-1] = r_i[1:-1,2:-1].ravel()
            dat[-1] = forcing_term[1:-1,2:-1].ravel()
        # Now add the component coming from the vertical advection term
        # Note: the following term (from the vertical advection) excludes xi=0 and xi=1 entries
        xi_adv_term = 0.25*dt/dxi*(1+Psi_m)*(Phi_n_new[1:-1,:]-XI[1:-1,:]*(Phi_n_new[-1,:])[np.newaxis,:])\
                                           *(Phi_n_new[2:,:]-Phi_n_new[:-2,:])*(H/h_new)[np.newaxis,:]
        row.append(indices[1:-1,1:-1].ravel())
        col.append(r_i[1:-1,1:-1].ravel())
        dat.append(xi_adv_term[:,1:-1].ravel())
        if self._use_artificial_dr_bc:
            #xi_adv_term[:,1] = 0 # Need to modify this in conjunction with the 'artificial' BC at r=dr
            row[-1] = indices[1:-1,2:-1].ravel()
            col[-1] = r_i[1:-1,2:-1].ravel()
            dat[-1] = xi_adv_term[:,2:-1].ravel()
        if not self._add_top_Phi_bc:
            # Add forcing on top row...
            row.append(indices[-1,1:-1].ravel())
            col.append(r_i[-1,1:-1].ravel())
            dat.append(forcing_term[-1,1:-1].ravel())
        # done, construct and return
        return coo_matrix((np.concatenate(dat),(np.concatenate(row),np.concatenate(col))),shape=(nr*nxi,nr))#.tocsr()
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
        rhs[1:-1] -= 0.5*dt*Upsilon*( H_old*Phi_n_old[-1,:]*g_b_old \
                                     +H_new*Phi_n_new[-1,:]*g_b_new)[1:-1]
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
        if self._use_artificial_dr_bc:
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
        if self._use_artificial_dr_bc:
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
        # Note: this block has a rectangular shape
        nr,nxi = len(self._r),len(self._xi)
        row = np.arange(1,nr-1)
        col = nxi-1+nxi*row
        dat = 0.5*dt*Upsilon*((h_new>h_ast)*g_b_new)[1:-1]
        if self._use_artificial_dr_bc:
            # Related to the enforcement of an artificial BC at r=dr
            #dat[1] = 0
            row = row[1:]
            col = col[1:]
            dat = dat[1:]
        return coo_matrix((dat,(row,col)),shape=(nr,nr*nxi))#.tocsr()
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
        if self._use_artificial_dr_bc:
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
        diagonals[2][1:-1] += 0.5*dt*Upsilon*H_new[1:-1]*Phi_n_new[-1,1:-1]
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
        if self._use_artificial_dr_bc:
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
        The private switch _Phi_n_DCN_solver may be used to toggle 
        through a few different modifications
        of the Phi_n solver.
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
                        if self._Phi_n_DCN_solver==0:
                            # Explicit Phi_n iterate
                            Phi_n_new[:,:] = self._Phi_n_equation_explicit(v_old,dt)
                            eps = -1
                        elif self._Phi_n_DCN_solver==1:
                            # Semi-implicit Phi_n iterate (but explicit in the d/dr term)
                            #Phi_n_new[:,:] = self._Phi_n_equation_semi_implicit(v_old,dt,explicit_r_advection=True)
                            Phi_n_new[:,:] = self._Phi_n_equation_semi_implicit([h_new,Phi_n_old,g_s_old,g_b_old],
                                                                                dt,explicit_r_advection=True)
                            eps = -1
                        elif self._Phi_n_DCN_solver==2:
                            # Implicit Phi_n iterate (linearised in the non-linear d/d\xi term)
                            Phi_n_new[:,:] = self._Phi_n_equation_semi_implicit([h_new,Phi_n_old,g_s_old,g_b_old],dt)
                            eps = -1
                        elif self._Phi_n_DCN_solver==3: 
                            # A decoupled Crank-Nicolson iteration
                            # Phi_n equation row block matrices and rhs vector
                            b_1  = self._Phi_n_equation_RHS(v_old,v_new,dt)
                            A_11 = self._Phi_n_equation_LHS1(v_new,dt)
                            # Solve the linear system and update current guess
                            dPhi_n = spsolve(A_11.tocsr(),b_1)
                            Phi_n_new += dPhi_n.reshape(Phi_n_new.shape)
                            eps = np.linalg.norm(dPhi_n)/np.linalg.norm(Phi_n_new.ravel())
                        else: #self._Phi_n_DCN_solver==4
                            # A decoupled Crank-Nicolson iteration
                            # Phi_n equation row block matrices and rhs vector
                            b_1  = self._Phi_n_equation_RHS(v_old,v_new,dt)
                            A_11 = self._Phi_n_equation_LHS1(v_new,dt).tocsc()
                            # Solve the linear system and update current guess
                            A_11_ILU = spilu(A_11)
                            P_op = LinearOperator(A_11.shape,A_11_ILU.solve)
                            dPhi_n,info = gmres(A_11,b_1,M=P_op,tol=1.0E-8,atol=1.0E-15)
                            if info!=0:
                                print("gmres iteration failed with code ",info)
                            Phi_n_new += dPhi_n.reshape(Phi_n_new.shape)
                            eps = np.linalg.norm(dPhi_n)/np.linalg.norm(Phi_n_new.ravel())
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
            self._h[:]       = h_new[:]
            self._Phi_n[:,:] = Phi_n_new[:,:]
            self._g_s[:]     = g_s_new[:]
            self._g_b[:]     = g_b_new[:]
        if t<S+T and (S+T-t)>1.0E-12:
            decoupled_newton_iterate(v_old,v_new,S+T-t)
            t = S+T
            self._h[:]       = h_new[:]
            self._Phi_n[:,:] = Phi_n_new[:,:]
            self._g_s[:]     = g_s_new[:]
            self._g_b[:]     = g_b_new[:]
        # done, no return
    def _full_Crank_Nicolson(self,T,dt=None):
        """
        This solves the non-linear system equations using Newton iterations
        on the entire system of equations simultaneously. Consequently the 
        four fields of interest are strongly coupled through each time step.
        This is generally much costlier for negligible gain over the 
        'decoupled' routine. It may however be useful if we find
        parameters in which non-linear effects are more important.
        The private switch _FCN_solver_mode can be used to toggle through
        a variety of different modifications to the solver.
        """
        # Setup...
        v_old = [self._h,self._Phi_n,self._g_s,self._g_b]
        h_new = self._h.copy()
        Phi_n_new = self._Phi_n.copy() # may want to flatten this?
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
            nxi = len(self._xi)
            for k in range(newt_its):
                # h equation rows block matrices and rhs vector
                b_0  = self._h_equation_RHS(v_old,v_new,dt)
                A_00 = self._h_equation_LHS0(v_new,dt)
                if self._FCN_solver_mode>=3:
                    A_01 = self._h_equation_LHS1(v_new,dt) # simple diag
                if self._FCN_solver_mode>=1:
                    A_02 = self._h_equation_LHS2(v_new,dt) # None
                    A_03 = self._h_equation_LHS3(v_new,dt) # simple diag
                # phi_n equation row block matrices and rhs vector
                b_1  = self._Phi_n_equation_RHS(v_old,v_new,dt)   
                if self._FCN_solver_mode>=3:
                    A_10 = self._Phi_n_equation_LHS0(v_new,dt) 
                A_11 = self._Phi_n_equation_LHS1(v_new,dt)      
                if self._FCN_solver_mode>=2:
                    A_12 = self._Phi_n_equation_LHS2(v_new,dt) # None
                    A_13 = self._Phi_n_equation_LHS3(v_new,dt) # simple diag
                # g_s equation row block matrices and rhs vector
                b_2  = self._g_s_equation_RHS(v_old,v_new,dt) 
                if self._FCN_solver_mode>=1:
                    A_20 = self._g_s_equation_LHS0(v_new,dt) # None
                if self._FCN_solver_mode>=2:
                    A_21 = self._g_s_equation_LHS1(v_new,dt) # None
                A_22 = self._g_s_equation_LHS2(v_new,dt)     
                if self._FCN_solver_mode>=1:
                    A_23 = self._g_s_equation_LHS3(v_new,dt) # simple diag
                # g_b equation row block matrices and rhs vector
                b_3  = self._g_b_equation_RHS(v_old,v_new,dt)
                if self._FCN_solver_mode>=1:
                    A_30 = self._g_b_equation_LHS0(g_b_old,v_new,dt) 
                if self._FCN_solver_mode>=2:
                    A_31 = self._g_b_equation_LHS1(v_new,dt) 
                if self._FCN_solver_mode>=1:
                    A_32 = self._g_b_equation_LHS2(v_new,dt) # simple diag
                A_33 = self._g_b_equation_LHS3(h_old,v_new,dt) 
                # Construct the sparse block matrix
                b_full = np.concatenate([b_0,b_1,b_2,b_3])
                if self._FCN_solver_mode<=0: # roughly equivalent to DCN
                    A_full = bmat([[A_00,None,None,None],
                                   [None,A_11,None,None],
                                   [None,None,A_22,None],
                                   [None,None,None,A_33]],format='csr') 
                    # Initial debugging test --> success (up to t=10/32)
                elif self._FCN_solver_mode==1: # add coupling between h,g_s and g_b
                    A_full = bmat([[A_00,None,A_02,A_03],
                                   [None,A_11,None,None],
                                   [A_20,None,A_22,A_23],
                                   [A_30,None,A_32,A_33]],format='csr') 
                    # Second debugging test --> success (up to t=5)
                elif self._FCN_solver_mode==2: # add coupling between g_b and Phi_n (noting A_12 and A_21 are None)
                    A_full = bmat([[A_00,None,A_02,A_03],
                                   [None,A_11,A_12,A_13],
                                   [A_20,A_21,A_22,A_23],
                                   [A_30,A_31,A_32,A_33]],format='csr') 
                    # Third debugging test --> success (up to t=10/32)
                    # Going from the 2nd to 3rd debug test significantly increased the time to solve, but result looks okay.
                elif self._FCN_solver_mode==3: # Use the complete FCN matrix
                    A_full = bmat([[A_00,A_01,A_02,A_03],
                                   [A_10,A_11,A_12,A_13],
                                   [A_20,A_21,A_22,A_23],
                                   [A_30,A_31,A_32,A_33]],format='csr') # Full solve
                elif self._FCN_solver_mode>=4: # Use the complete FCN matrix
                    A_full = bmat([[A_00,A_01,A_02,A_03],
                                   [A_10,A_11,A_12,A_13],
                                   [A_20,A_21,A_22,A_23],
                                   [A_30,A_31,A_32,A_33]],format='csc') # Full solve
                    # Note: another significant slow down going from the 3rd debug test to the full solver...
                # Solve the linear system
                if self._FCN_solver_mode<=3:
                    # Direct method
                    dv = spsolve(A_full,b_full)
                elif self._FCN_solver_mode==4:
                    # Iterative method
                    counter = gmres_counter(newt_verbose)
                    #A_iLU = spilu(A_full) # defaults: fill_factor=10,drop_tol=1.0E-4
                    #P_op = LinearOperator((nr*(3+nxi),nr*(3+nxi)),A_iLU.solve)
                    #dv,info = gmres(A_full,b_full,M=P_op,tol=1.0E-8,atol=1.0E-15) # defaults: tol=1.0E-5,restart=20
                    # This seems to work quite well really and is much quicker... (with only 1 thread!)
                    # But... the gmres solver seems to hang around t=4...
                    #        It may be an error in the matrix as h starts to move, 
                    #        but a preliminary check suggests this is not the case...
                    #        (although s couple of extra newton iteration do seem to be required...)
                    #        Rather, I may need to play around with gmres and spilu parameters...
                    #        Yes, some different parameters seems to get past that point...
                    # Potential improvements: 
                    # a) use only diagonal blocks for the preconditioner
                    # b) replace A_full with a linear operator in the gmres call
                    #    (using a matrix free function to construct the linear operator...)
                    # The following seems a bit more robust, but slows things down a little
                    #A_iLU = spilu(A_full,fill_factor=20,drop_tol=1.0E-5)
                    #P_op = LinearOperator((nr*(3+nxi),nr*(3+nxi)),A_iLU.solve)
                    #dv,info = gmres(A_full,b_full,M=P_op,tol=1.0E-8,atol=1.0E-15,restart=50)
                    # Maybe this is sufficient? No, it just gets stuck a little later...
                    A_iLU = spilu(A_full,fill_factor=10,drop_tol=0.5**15)
                    P_op = LinearOperator((nr*(3+nxi),nr*(3+nxi)),A_iLU.solve)
                    dv,info = gmres(A_full,b_full,M=P_op,tol=1.0E-8,atol=1.0E-15,restart=20,callback=counter)
                    if info!=0:
                        print("gmres iteration failed with code ",info)
                elif self._FCN_solver_mode==5:
                    # Iterative method bicgstab
                    A_iLU = spilu(A_full) # defaults: fill_factor=10,drop_tol=1.0E-4
                    P_op = LinearOperator((nr*(3+nxi),nr*(3+nxi)),A_iLU.solve)
                    dv,info = bicgstab(A_full,b_full,M=P_op,tol=1.0E-8,atol=1.0E-15) # defaults: tol=1.0E-5,atol=None
                    # Note: bicgstab fails at the same place as gmres, it is no faster/slower up to that point
                    if info!=0:
                        print("bicgstab iteration failed with code ",info)
                elif self._FCN_solver_mode==-1:
                    # Try a 'matrix free' gmres method (apart from the pre-conditioner construction)
                    A_iLU = spilu(A_full) # note this has only the diagonal blocks currently...
                    P_op = LinearOperator((nr*(3+nxi),nr*(3+nxi)),A_iLU.solve)
                    def A_fun(x):
                        h_temp     = h_new     + x[          :        nr]
                        Phi_n_temp = Phi_n_new + x[        nr:(1+nxi)*nr].reshape((nxi,nr))
                        g_s_temp   = g_s_new   + x[(1+nxi)*nr:(2+nxi)*nr]
                        g_b_temp   = g_b_new   + x[(2+nxi)*nr:          ]
                        v_temp = [h_temp,Phi_n_temp,g_s_temp,g_b_temp]
                        Ax_0  = self._h_equation_RHS(v_old,v_temp,dt)
                        Ax_1  = self._Phi_n_equation_RHS(v_old,v_temp,dt)  
                        Ax_2  = self._g_s_equation_RHS(v_old,v_temp,dt) 
                        Ax_3  = self._g_b_equation_RHS(v_old,v_temp,dt)
                        Ax = np.concatenate([Ax_0,Ax_1,Ax_2,Ax_3])
                        Ax -= b_full
                        return Ax
                    A_op = LinearOperator((nr*(3+nxi),nr*(3+nxi)),A_fun)
                    dv,info = gmres(A_op,b_full,M=P_op,tol=1.0E-8,atol=1.0E-15)
                    # This works really well actually, and doesn't get hung up where the others did...
                    if info!=0:
                        print("gmres (matrix free) iteration failed with code ",info)
                else:
                    print("Unrecognized FCN_solver_mode parameter, exiting...")
                    return
                    
                # Calculate epsilon
                eps = np.linalg.norm(dv)/np.linalg.norm(np.concatenate([array.ravel() for array in v_new]))
                # Update current guess
                h_new     += dv[          :        nr]
                Phi_n_new += dv[        nr:(1+nxi)*nr].reshape((nxi,nr))
                g_s_new   += dv[(1+nxi)*nr:(2+nxi)*nr]
                g_b_new   += dv[(2+nxi)*nr:          ]
                # Check epsilon and the current iteration number for termination conditions...
                if newt_verbose:
                    print("Newton Method: Completed iteration {:d} with eps={:g}".format(k+1,eps))
                if eps<newt_tol:
                    if newt_verbose:
                        print("Newton Method: Converged within {:g} in {:d} iterations".format(newt_tol,k+1))
                    break
                if k==newt_its-1:
                    print("Newton Method: Failed to converge in {:d} iterations (eps={:g})".format(newt_its,eps))
            # done
        # Now perform the Newton iterations until the final time is reached 
        t = self._t
        S = t
        while t<=S+T-dt:
            fully_coupled_newton_iterate(v_old,v_new,dt)
            t += dt
            self._h[:]       = h_new[:]
            self._Phi_n[:,:] = Phi_n_new[:,:]
            self._g_s[:]     = g_s_new[:]
            self._g_b[:]     = g_b_new[:]
        if t<S+T and (S+T-t)>1.0E-12:
            fully_coupled_newton_iterate(v_old,v_new,S+T-t)
            t = S+T
            self._h[:]       = h_new[:]
            self._Phi_n[:,:] = Phi_n_new[:,:]
            self._g_s[:]     = g_s_new[:]
            self._g_b[:]     = g_b_new[:]
        # done, no return
    # End of class    
    
