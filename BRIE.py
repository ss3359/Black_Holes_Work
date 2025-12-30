
import numpy as np
import pandas as pd 
import sympy as sp
import scipy as si
from sklearn.model_selection import train_test_split 
from sympy import Matrix, sin,cos,tan, exp, diff
import matplotlib.pyplot as plt 
import matplotlib.animation as an



# G = 6.6743e-11 # Gravitational Constant m^3 kg^-1s^-2
G=1
M=1
c=1


t,r,theta,phi=sp.symbols('t r theta phi')

u=[t,r,theta,phi]
class Schwartzchild: 
    dt=0.001
    def __init__(self,t,r,theta,phi):
        self.t=t
        self.r=r
        self.theta=theta
        self.phi=phi
        self.rs=(2*G*M)/(c*c)
        
    def d_du(self,dt,dr,dtheta,dphi):  
        return np.array([dt,dr,dtheta,dphi]) 
    
    def dg(self,f,symbol): 
        return diff(f,symbol)
    
    def g_μν(self): 
        return Matrix([[-(1-(self.rs/r)),0,0,0],
                       [0,(1-(self.rs/r))**-1,0,0], 
                       [0,0,r**2,0], 
                       [0,0,0,(r*sin(theta))**2]])
    def inv_g_μν(self): 
        g_μν=self.g_μν()
        return g_μν.inv()
    
    def Christoffell_Symbols(self): 
        g_μν=self.g_μν()
        inv_gμν=self.inv_g_μν()

        Symbols=[[[0 for _ in range(len(u))] for _ in range(len(u))]for _ in range(len(u))]
        for μ in range(len(u)):
            for ν in range(len(u)): 
                for λ in range(len(u)):
                    Γ_μν_λ=0
                    for σ in range(len(u)): 
                        Γ_μν_λ+=sp.Rational(1,2)*inv_gμν[λ,σ]*(self.dg(g_μν[σ,ν],u[μ])+self.dg(g_μν[σ,μ],u[ν])-self.dg(g_μν[μ,ν],u[σ]))    
                    Γ_μν_λ=sp.lambdify(u,sp.sympify(Γ_μν_λ),'numpy')
                    Symbols[μ][ν][λ]=Γ_μν_λ 
        return Symbols

    def GeodesicEquationAcceletation(self,mu,U,DU,Symbols): 

        result=0
        for alpha in range(len(u)): 
            for beta in range(len(u)):
                result-=(Symbols[alpha][beta][mu](U[0],U[1],U[2],U[3]))*DU[alpha]*DU[beta]
        return result
    
    def GeodesicEquationVelcity(self,mu,DU): 
        return DU[mu]
    
    def RK4_Uodate(self): 
        dt=self.dt
        n=100
        pos=np.array([self.t,self.r,self.theta,self.phi])
        vel=self.d_du(1/(1-(2/self.r)),0,0,0)
        CS_Symbols=self.Christoffell_Symbols()

        print(f"Initial Position: {pos}")
        print(f"Initial Velcoity: {vel}")
        print("\n")

        pos_vectors=[]
        pos_vectors.append(pos)
        rs=2*G*M
        for i in range(n): 
            for k in range(len(u)): 
                pos_copy=pos.copy()
                vel_copy=vel.copy()

                K1=self.GeodesicEquationAcceletation(k,pos_copy,vel_copy,CS_Symbols)
                J1=self.GeodesicEquationVelcity(k,vel)
                pos_copy[k]+=(dt/2.0)*K1
                vel_copy[k]+=(dt/2.0)*J1
                
                K2=self.GeodesicEquationAcceletation(k,pos_copy,vel_copy,CS_Symbols)
                J2=self.GeodesicEquationVelcity(k,vel)
                pos_copy[k]+=(dt/2.0)*K2
                vel_copy[k]+=(dt/2.0)*J2

                K3=self.GeodesicEquationAcceletation(k,pos_copy,vel_copy,CS_Symbols)
                J3=self.GeodesicEquationVelcity(k,vel)
                pos_copy[k]+=dt*K3
                vel_copy[k]+=dt*J3

                K4=self.GeodesicEquationAcceletation(k,pos_copy,vel_copy,CS_Symbols)
                J4=self.GeodesicEquationVelcity(k,vel)

                vel[k]+=(1.0/6.0)*(K1+2*K2+2*K3+K4)
                pos[k]+=(1.0/6.0)*(J1+2*J2+2*J3+J4)

            pos[1] = max(pos[1], rs + 1e-3)
            pos_vectors.append(pos.copy())
            print(f"New Position: {pos}")
            print(f"New Velocity: {vel}")
            print()

        return pos_vectors

    def PlotPoints(self,pos_polar): 
        vectors_carts=[]
        for vector in pos_polar: 
            print(vector)
            x=vector[1]*np.cos(vector[3])
            y=vector[1]*np.sin(vector[3])
            z=0  # or  z=vector[1]*np.cos(vector[2]) 
            vector_cart=np.array([x,y,z])
            vectors_carts.append(vector_cart)
        vectors_carts=np.array(vectors_carts)

        fig,ax=plt.subplots()
        ax.set_aspect('equal')
        ax.set_xlim(-50,50)
        ax.set_ylim(-50,50)

        bh=plt.Circle((0,0),2,color='black')
        ax.add_patch(bh)
        particle,=ax.plot([],[],'ro')

        def update(frame): 
            particle.set_data(vectors_carts[frame][0],vectors_carts[frame][1])
            return particle,
        ani=an.FuncAnimation(fig,update,frames=len(vectors_carts),interval=50,blit=True)
      
        plt.show()


def main(): 
    # initial_r_points=np.linspace(0.001,10,endpoint=True)
    # initial_phi_points=np.linspace(0.01,2*np.pi,endpoint=True)
    
    SC=Schwartzchild(0,10,np.pi/2,0)
    
    point_car=SC.RK4_Uodate()
    SC.PlotPoints(point_car)

main()










