# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 21:48:29 2020

@author: HBK
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
from scipy import integrate
import time
from scipy.sparse import diags,kron
import scipy.sparse.linalg as las

def destroy(N):
    """
    Parameters
    ----------
    N : int, dimension of the destroy operator
    Returns
    -------
    b : N*N array
    """
    upper_diag = np.sqrt(np.arange(1,N))
    b = np.diag(upper_diag, 1)
    return b
    

def create(N):
    """
    Parameters
    ----------
    N : int, dimension of the creation operator
    Returns
    -------
    b_dagger : N*N array
    """
    b_dagger = destroy(N).T.conjugate()
    return b_dagger


def tensor(matrices):
    """
    Parameters
    ----------
    matrices: list of matrix to perform Kronecker product
    Returns
    -------
    m1: composite matrix
    """
    m1 = matrices[0]
    for m in matrices[1:]:
        m1 = kron(m1, m)
    return m1


class QTransport(object):
    T = np.pi / (2 * 0.125)  # Time scale 
    def __init__(self, s, d, sites, Gamma=None,E=None):
        """
        Parameters
        ----------
        s : int
            Total numbers of site
        d : int
            dimension of sites
        site : array
            position array of the site
        Gamma: float
            sink rate
        n_p:
            int, number of phonon states
        """
        self.s = s
        self.d = d
        self.sites = sites
        self.Gamma = Gamma
        self.E = E
    def get_s(self):
        return self.s
    
    
    def get_d(self):
        return self.d
    
    
    def get_sites(self):
        return self.sites
    
    
    def get_Gamma(self):
        return self.Gamma
    
    
    def set_s(self, new_s):
        self.s = new_s
        
        
    def set_d(self, new_d):
        self.d = new_d
        
        
    def set_sites(self, new_sites):
        self.sites = new_sites
        
        
    def set_Gamma(self, new_Gamma):
        self.Gamma = new_Gamma
        
        
    def __str__(self):
        return 'Transport with ' + str(self.s) + ' sites in ' \
                + str(self.d) + ' dimension'
    
    
    def state(self,i):
        """
        Generate a state with probability 1 at (i+1)th site
        """
        s = self.s
        state = np.zeros(s)
        state[i] = 1
        return state.reshape((len(state),1))
    
        
    def H(self, Gamma=None, E=None):
        """Calculate the Hamiltonian
        
        Parameters
        ----------
        sites, 1D Array: The positions(Cartesian) of a intemediate site for multi-sites configrations
        dim (s,d), Tuple: Number of sites and dimension of the problem
        Gamma, Float: The draining rate of the given sink
        
        Returns
        -------
        2D Array: Hamiltonian Matrix of the configuration 
        """
        s = self.s
        d = self.d
        Gamma = self.Gamma
        
        if E is not None:
            if len(E) != s:
                raise ValueError('E should have the same value as total of sites')
                
        if d==1:
            in_site = np.array([[1]])
            out_site = np.array([[-1]])
        elif d==2:
            in_site = np.array([[1,0]])
            out_site = np.array([[-1,0]])
        elif d==3:
            in_site = np.array([[1,0,0]])
            out_site = np.array([[-1,0,0]])
            
        sites = self.sites.reshape((s-2,d))
        sites = np.append(sites, in_site, axis=0) # *100 Anstrogm
        r = np.append(out_site, sites,axis=0)
        
        # Construct the symmetric matri
        H = np.zeros((s, s))
        c_d = 1 # The dipole constant 
    
        for i in range(s):
            for j in range(s):
                if i != j:
                    r_v = 0
                    for k in range(d):
                        r_v += (r[i,k] - r[j,k])**2
                    if r_v**0.5 >= 0.01:
                        # avoid clustering in the configuration: 0.01 is the r_min
                        H[i, j] = c_d / (r_v)**1.5
                    else:
                        H[i, j] = c_d / (0.01)**3
                        
        if Gamma is not None:
            H = H - 1j * (Gamma/2) * (self.state(s-1) @ self.state(s-1).T)
            
        
        if E is not None:
            H = H + np.diag(E)
        
        return H
    
    
    def site_plotter_3d(self, w=None, filename='3d_sites_plot.pdf', figsize=(25,25)):
        """
        Parameters
        ----------
        sites : 2d array
            Must be in the form of [[x1,y1,z1],[x2,y2,z2]...]
        filename : str, optional
            name of the file. The default is '3d_sites_plot'.
        figsize : TYPE, optional
            size of figure. The default is (25,25).
        Returns
        -------
        None
    
        """
        fig = plt.figure(num=1, clear=True, figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        (x, y) = np.meshgrid(np.arange(-1, 1, 0.01), np.arange(-1, 1, 0.01))
        z = np.zeros(np.shape(x))
        
        # X, Y, Z are sites:
        sites = self.sites
        s = self.s
        d = self.d
        new_sites = np.zeros((s-2,3))
        if d == 1:
            new_sites[:,0] = sites
            sites = new_sites
        elif d==2:
            new_sites[:,0] = sites[0::2]
            new_sites[:,1] = sites[1::2]
            sites = new_sites
        else:
            sites = sites.reshape(-1,3)
        
        sites_cor = sites.T
        X = sites_cor[0]
        Y = sites_cor[1]
        Z = sites_cor[2]
        
        # plot base surface
        ax.plot_surface(x, y, z, alpha=0.1)
        
        # plot axis line:
        ax.plot([-1,1], [0,0], [0,0],color='k', linewidth=3)
        
        # plot the guide line:
        for site in sites:
            s1 = site[0]
            s2 = site[1]
            s3 = site[2]
            ax.plot([s1,s1,s1],[0,s2,s2],[0,0,s3], '--',
                  color='k', linewidth=3, alpha=0.3)
        
        # Plot sites (in/out and intermediate)
        ax.scatter([-1,1],[0,0],[0,0],color='r',s=1000*np.ones(2), alpha=1)
        ax.scatter(X,Y,Z,color='k',s=1000*np.ones(len(X)), alpha=1)
        ax.text(-1, 0, 0.01, "in", size=50, alpha=1)
        ax.text(1, 0, 0.01, "out", size=50, alpha=1)
        # Plot the frequency:
            
        if w is not None:
            for w_i, site in zip(w[1:-1], sites):
                ax.text(site[0], site[1], site[2]-0.15, str(np.round(w_i, decimals=1)), size=40, alpha=1)
        ax.text(-1, 0, -0.15, str(np.round(w[0], decimals=1)), size=40, alpha=1)
        ax.text(1, 0, -0.15, str(np.round(w[-1], decimals=1)), size=40, alpha=1)
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        ax.set_axis_off()
        #ax.set(xlabel='x', ylabel='y', zlabel='z')
        fig.tight_layout()
        plt.savefig(filename.format(bbox_inches='tight'))
        return None

    



class CohQT(QTransport):
    
    def __init__(self, s, d, sites, T_m=np.pi/(2 * 0.125)/10):
        # super().__init__(s,d,sites)
        QTransport.__init__(self,s,d,sites)
        self.T_m = T_m
    
    
    def get_T_m(self):
        return self.T_m
    
    
    def get_p_a(self,t_a):
        """
        Parameters
        ----------
        t_a : float
            Time range for averaging
        
        Returns
        -------
        p_a : float
            Time averaged probability

        """
        p = lambda t: (abs(self.state(self.s-1).T @ la.expm(-1j * self.H() * t) @ self.state(0))**2).item()
        p_a = integrate.quad(p, 0, t_a, epsrel=0.0001, limit=1000)[0] / t_a
        
        return p_a
    
    
    def get_p_m(self):
        """
        -------
        p_m : function object
            population at site s_n at time t
        """
        p_m = lambda s_n,t: (abs(self.state(s_n-1).T @ la.expm(-1j * self.H() * t) @ self.state(0))**2).item()
        return p_m
    
    
    def plot_p(self, site_num, time):
        """
        Parameters
        ----------
        site_num : int, from 1...N
        time : time array for plotting
        Returns
        -------
        None.
        """
        P = np.zeros(len(time))
        for i, t_s in enumerate(time):
            P[i] = abs(self.state(site_num-1).T @ la.expm(-1j * self.H() * t_s) @ self.state(0))**2
        plt.title('Population Dynamics (Closed) for N={0}'.format(site_num))
        plt.plot(time/self.T, P)
        plt.xlabel('Time (T)')
        plt.ylabel('Population')
        plt.show()
    
    
    def P_m(self):
        """
        Returns
        -------
        P_m : float
            Max Population within T_m time range
        t_p : TYPE
            Time at which the maxP value is obtained
        """
        t = np.linspace(0, self.T_m, 1000)
        P = np.zeros(len(t))
        
        for i, t_s in enumerate(t):
            P[i] = abs(self.state(self.s-1).T @ la.expm(-1j * self.H() * t_s) @ self.state(0))**2
        
        t_p_i = np.argmax(P)
        P_m = P[t_p_i]
        t_p =  t_p_i * (t[1]-t[0])
        
        return P_m, t_p
    
    def __str__(self):
        return 'Coherent transport with ' + str(self.s) + ' sites in ' +\
            str(self.d) + ' dimensions. Monitoring time T_m is ' + str(self.get_T_m()/self.T) +\
            'T.'    


class NCohQT(QTransport):
    
    def __init__(self,s,d,sites, Gamma,E=None):
        QTransport.__init__(self,s,d,sites, Gamma, E)

        
    def T_r(self,epabs=0.001,limit=250):
        """
        Returns
        -------
        T_r : float
            Transfer time for the give transport
        p_gamma : float
            Population at the output site at the time 1/Gamma

        """
        Gamma = self.Gamma
        T = self.T
        
        
        psi = lambda t: abs(self.state(0).T @ la.expm(1j * np.conjugate(self.H(Gamma=Gamma,E=self.E)).T * t) \
                            @ la.expm(-1j * self.H(Gamma=Gamma,E=self.E) * t) @ self.state(0)).item()
        
        
        if psi(1000*T) >= 0.0001:
            t_f = 1000*T
            res = integrate.quad(psi, 0, t_f, limit=limit, epsabs=epabs)
            T_r = res[0]
        
        else:
            t_f = np.inf
            res = integrate.quad(psi, 0, t_f, limit=limit, epsabs=epabs)
            T_r = res[0]
        
        # T_r = integrate.quad(psi, 0, t_f, points=np.linspace(0,150*T,150), limit=limit, epsabs=epabs,full_output=1)[0]
        p_gamma = 1 - psi(1/Gamma)
        
        return T_r, p_gamma
    
    
    def get_p_t(self):
        """
        Returns
        -------
        p_t : function object
            Return sink population p_t as a FUNCTION of time
        """
        p_t = lambda t: 1 - abs(self.state(0).T @ la.expm(1j * np.conjugate(self.H(Gamma=self.Gamma,E=self.E)).T * t) \
                                @ la.expm(-1j * self.H(Gamma=self.Gamma,E=self.E) * t) @ self.state(0)).item()
        return p_t
    
    
    def get_p_r(self):
        """
        Returns
        -------
        p_r : function
            Return the FUNCTION that calculates the population at each site

        """
        p_r = lambda s_n,t: (abs(self.state(s_n-1).T @ la.expm(-1j * self.H(Gamma=self.Gamma,E=self.E) * t) @ self.state(0))**2).item()
        return p_r
    
    
    def plot_p_r(self, site_num, time, label=None):
        """
        Parameters
        ----------
        site_num : int, from 1...N
        time : time array for plotting
        Returns
        -------
        None.
        """
        P = np.zeros(len(time))
        for i, t_s in enumerate(time):
            P[i] = abs(self.state(site_num-1).T @ la.expm(-1j * self.H(Gamma=self.Gamma,E=self.E) * t_s) @ self.state(0))**2
        plt.title('Population Dynamics (sys Sink) for N={0}'.format(site_num))
        plt.plot(time/self.T, P,label=label)
        plt.xlabel('Time (T)')
        plt.ylabel('Population')
        return P
    
    def get_func(self,func):
        """
        Input:
            str: 'Psi', 'Rho', 'psi', 'p_t'
        Returns
        -------
        psi(t): Integrand of the transfer time
        p_t(t): Sink population
        p_r(s_n,t): Population at each site s_n
        """
        Gamma = self.Gamma
        psi = lambda t: abs(self.state(0).T @ la.expm(1j * np.conjugate(self.H(Gamma=Gamma,E=self.E)).T * t) \
                            @ la.expm(-1j * self.H(Gamma=Gamma,E=self.E) * t) @ self.state(0)).item()
        p_r = lambda s_n,t: (abs(self.state(s_n-1).T @ la.expm(-1j * self.H(Gamma=self.Gamma,E=self.E) * t) @ self.state(0))**2).item()
        p_t = lambda t: 1 - abs(self.state(0).T @ la.expm(1j * np.conjugate(self.H(Gamma=self.Gamma,E=self.E)).T * t) \
                                @ la.expm(-1j * self.H(Gamma=self.Gamma,E=self.E) * t) @ self.state(0)).item()
        
        if func=='psi':
            return psi
        elif func=='p_t':
            return p_t
        elif func=='p_r':
            return p_r
        else:
            raise ValueError('No such function')
            

    def plot_long_dyc(self, time, filename="NCoh_dyc.pdf", log_scale=True):
        """
        Parameters
        ----------
        dim : Tuple
            (s, d); dimension of the sites. 
        site : 1d Array
            The positions array of all sites. 
        time : Float
            The final time scale on the plot
        Gamma : Float
            The Sinking rate
    
        Returns
        -------
        None.
        """
        site = self.sites
        Gamma = self.Gamma
        T = np.pi / (2 * 0.125)
        s = self.s
        d = self.d
        n_site = np.arange(2, s) # intermediate site number
        if log_scale is True:
            t = np.logspace(np.log10(1e-3*T), np.log10(time), 2500)
        else:
            t = np.linspace(0, time, 2500)
        p_r = self.get_func('p_r')
        p_t = self.get_func('p_t')
        T_r = self.T_r()[0]
       # p_t = qt.T_r(dim, site, Gamma=Gamma, get_p_t=True)  # Sink Population
        # plot figures
        plt.figure(figsize=(20,10))
        fig, ax = plt.subplots(1,1)
        
        for n in n_site:
            P_r = [p_r(n, ti) for ti in t]
            plt.plot(t/T, P_r, color='silver', alpha=0.7)
    
        P_t =[p_t(ti) for ti in t]
        P_r_out = [p_r(s, ti) for ti in t]
        P_r_in = [p_r(1, ti) for ti in t]
        plt.plot(t/T, P_t, color='k', label=r'$\mathfrak{p}$', alpha=0.8)
        plt.plot(t/T, P_r_in, color='b', label='in', alpha=0.8)
        plt.plot(t/T, P_r_out, color='r', label='out', alpha=0.8)
        plt.ylabel('Population')
        plt.xlabel('Time (T)')
        plt.xscale('log')
        plt.vlines(T_r/T, ymin=0, ymax=1, colors='k', ls=':')
        plt.text(T_r/T - 0.005, 0.95, r'$\mathfrak{T}$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename.format(bbox_inches='tight'))
    
    
    def __str__(self):
        return 'Non-coherent transport with ' + str(self.s) + ' sites in ' +\
            str(self.d) + ' dimensions. Sinking rate Gamma is ' + str(self.get_Gamma()*self.T) +\
            '/T.'    



class OpenQT(QTransport):
    
    def __init__(self, s, d, sites, w, Gamma=None, n_p=3,E=None,g=1,g_t=0):
        # super().__init__(s,d,sites)
        QTransport.__init__(self,s,d,sites,Gamma=None)
        self.n_p = n_p
        self.Gamma=Gamma
        self.w=w
        self.n_p = n_p
        self.E=E
        self.g=g
        self.g_t=g_t
        # Total wavefunction Psi


    def H_Total(self, Gamma=None, g=0, g_t=0):
        '''
        Parameters
        ----------
        w : s-2 array, phonon frequecy at each intermediate site
        
        Returns
        -------
        None.
        '''

        s = self.s
        n_p = self.n_p # number of phonon states
        w=self.w
        if s != len(w):
            raise ValueError('w should have the same size as sites')
        
        dimS = s
        dimB = n_p**s
        b_dag = create(n_p)
        b = destroy(n_p)
        b_dag_b = b_dag @ b
        I_S = np.eye(dimS)
        H_S = self.H(Gamma=Gamma,E=self.E)

        I_B = np.eye(dimB)
        H_B = np.zeros((n_p**s, n_p**s))
        H_SB = np.zeros((s*n_p**s, s*n_p**s))
        g = g
        g_t= g_t
        Snm = np.zeros((dimS, dimS))
        Snn = np.zeros((dimS, dimS))
        qn = np.zeros((dimB, dimB))

        for m_i in range(s):

            qn = tensor([np.eye(int(n_p**(m_i))), (b + b_dag), np.eye(int(n_p**(s-m_i-1)))])
            Snn = I_S[:,m_i].reshape(-1,1) @ I_S[:,m_i].reshape(-1,1).T
            H_B = H_B + tensor([np.eye(int(n_p**(m_i))), w[m_i] * b_dag_b, np.eye(int(n_p**(s-m_i-1)))])

            for m_j in range(m_i+1, s):
                Snm = I_S[:,m_i].reshape(-1,1) @ I_S[:,m_j].reshape(-1,1).T \
                    + I_S[:,m_j].reshape(-1,1) @ I_S[:,m_i].reshape(-1,1).T

                qnm = tensor([np.eye(int(n_p**(m_i))), (b + b_dag), np.eye(int(n_p**(s-m_i-1)))]) \
                -tensor([np.eye(int(n_p**(m_j))), (b + b_dag), np.eye(int(n_p**(s-m_j-1)))])
                H_SB = H_SB + kron(Snm, g_t*qnm*H_S[m_i,m_j])
                #print("**************************")

            H_SB = H_SB + g*kron(Snn, qn)
            
        
        H_Total = kron(H_S,I_B) + kron(I_S,H_B) + H_SB
        return H_Total
    
    

    
    def T_r(self,epabs=0.1,limit=50):
        """
        Returns
        -------
        T_r : float
            Transfer time for the give transport
        p_gamma : float
            Population at the output site at the time 1/Gamma
        """

        Gamma = self.Gamma
        T = self.T
        psi = self.get_func('psi')

        if psi(1000*T) >= 0.0001:
            t_f = 1000*T
            res = integrate.quad(psi, 0, t_f, limit=limit, epsabs=epabs)
            T_r = res[0]
            #T_r = 12.5
        
        else:
            t_f = np.inf
            res = integrate.quad(psi, 0, t_f, limit=limit, epsabs=epabs)
            T_r = res[0]

        p_gamma = 1 - psi(1/Gamma)

        return T_r, p_gamma
    
    
    def T_r_s(self,filt=0.05*4*np.pi):
        """
        Returns
        -------
        T_r : float
            Transfer time for the give transport
        p_gamma : float
            Population at the output site at the time 1/Gamma
        """
        Gamma = self.Gamma
        T = self.T
        psi = self.get_func('psi')

        return psi(filt)


    def get_func(self,func):
        """
        Input:
            str: 'Psi', 'Rho', 'psi', 'p_t','p_r'
        Returns
        -------
        Psi(t): Total wave function in the composite system 
        Rho(t): Total density matrix
        psi(t): Integrand of the transfer time
        p_t(t): Sink population
        p_r(s_n,t): Population at each site s_n
        """
        # t0 = time.time()
        Gamma = self.Gamma
        T = self.T
        s = self.s
        m = s
        g = self.g
        g_t = self.g_t
        n_p = self.n_p # number of phonon states
        dimS = s
        dimB = n_p**m
        #dim = dimS * dimB
        I_S = np.eye(dimS)
        I_B = np.eye(dimB)
        Eval, U = la.eig(self.H_Total(Gamma=Gamma,g=g,g_t=g_t))
        psi_in = kron(I_S[:,0].reshape(-1,1),I_B)
        psi_out = kron(I_S[:,-1].reshape(-1,1), I_B)
        Psi = lambda t: U @ diags(np.exp(-1j*Eval*t)) @ la.inv(U) @ psi_in
        # t1 = time.time()
        # print("time",t1-t0)

        def Rho(t):
            Psi_t = Psi(t)
            return Psi_t.T.conjugate() @ Psi_t 
        
        # def Rho2(t):
        #     res = psi_in.T @ U @ np.diag(np.exp(2*np.imag(Eval)*t)) @ np.conjugate(la.inv(U)) @ psi_in
        #     return res
        
        def Rho2(t):
            Psi_t = Psi(t)
            return Psi_t @ Psi_t.T.conjugate()
        

        # print(np.round(np.abs(U@U.T),2))
        
        psi = lambda t: np.trace(np.real(Rho(t)))/dimB
        p_t = lambda t: 1 - psi(t)
        p_r = lambda s_n,t: np.sum(np.real(np.diagonal(Rho2(t))[dimB*s_n-dimB:dimB*s_n]/dimB)).item()
        
        if func=='Psi':
            return Psi
        elif func=='Rho':
            return Rho
        elif func=='psi':
            return psi
        elif func=='p_t':
            return p_t
        elif func=='p_r':
            return p_r
        else:
            raise ValueError('No such function')
            

    def plot_p_r(self, site_num, time, label=None):
        """
        Parameters
        ----------
        site_num : int, from 1...N
        time : time array for plotting
        Returns
        -------
        Plot of site_num population as a function of time
        """
        p_r = self.get_func('p_r')
        P = np.zeros(len(time))
        for i, t_s in enumerate(time):
            P[i] = p_r(site_num, t_s)
        plt.title('Population Dynamics (Open_sys Sink) for N={0}'.format(site_num))
        plt.plot(time/self.T, P,label=label)
        plt.xlabel('Time (T)')
        plt.ylabel('Population')

        return P

    def plot_long_dyc(self, time, filename="open_dyc.pdf",log_scale=True):
        """
        Parameters
        ----------
        dim : Tuple
            (s, d); dimension of the sites. 
        site : 1d Array
            The positions array of all sites. 
        time : Float
            The final time scale on the plot
        Gamma : Float
            The Sinking rate
    
        Returns
        -------
        None.
        """
        site = self.sites
        Gamma = self.Gamma
        T = np.pi / (2 * 0.125)
        s = self.s
        d = self.d
        n_site = np.arange(2, s) # intermediate site number
        if log_scale is True:
            t = np.logspace(np.log10(1e-4*T), np.log10(time), 250)
        else:
            t = np.linspace(0, time, 150)
        p_r = self.get_func('p_r')
        p_t = self.get_func('p_t')
        T_r = self.T_r(epabs=0.01)[0]
       # p_t = qt.T_r(dim, site, Gamma=Gamma, get_p_t=True)  # Sink Population
        # plot figures
        plt.figure(figsize=(20,10))
        fig, ax = plt.subplots(1,1)
        
        for n in n_site:
            P_r = [p_r(n, ti) for ti in t]
            plt.plot(t/T, P_r, color='silver', alpha=0.7)
    
        P_t =[p_t(ti) for ti in t]
        P_r_out = [p_r(s, ti) for ti in t]
        P_r_in = [p_r(1, ti) for ti in t]
        plt.plot(t/T, P_t, color='k', label=r'$\mathfrak{p}$', alpha=0.8)
        plt.plot(t/T, P_r_in, color='b', label='in', alpha=0.8)
        plt.plot(t/T, P_r_out, color='r', label='out', alpha=0.8)
        plt.ylabel('Population')
        plt.xlabel('Time (T)')
        plt.xscale('log')
        plt.vlines(T_r/T, ymin=0, ymax=1, colors='k', ls=':')
        plt.text(T_r/T - min(t)*0.01, 0.95, r'$\mathfrak{T}$')
        plt.legend()
        #plt.tight_layout()
        plt.savefig(filename.format(bbox_inches='tight'))
    