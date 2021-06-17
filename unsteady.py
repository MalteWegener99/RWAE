import numpy as np
import matplotlib.pyplot as plt

# CCW is positive

def solve_steady(Vpos, Cpos, Cnorm, Wpos, Ws, Vinf):
    """
    Vpos: 2xN array
    Cpos: 2xN array
    Cnorm: 2xN array
    Wpos: 2xM array
    Ws: 2xM array
    """

    mtmp = np.array([[0,1.0],[-1,0]])
    N = Vpos.shape[1]
    mat = np.zeros((N,N))
    rhs = np.zeros(N)
    for i in range(N):
        # Influences at controlpoint i
        # u,v = Gamma/2/pi/r * [[0,1][-1,0]] (cp-Vpos)
        r = np.reshape(Cpos[:,i],(2,1))-Vpos
        r2 = np.sqrt(r[0,:]**2+r[1,:]**2)
        r2 = r2**(1/2)
        ru = mtmp@r
        ru = ru/np.sqrt(ru[0,:]**2+ru[1,:]**2)
        u = -(1/2/np.pi/r2*ru)
        mat[i,:] = (np.reshape(Cnorm[:,i],(1,2)))@u
        

        rhs[i] = -Vinf*Cnorm[0,i]
        r = np.reshape(Cpos[:,i],(2,1))-Wpos
        r2 = np.sqrt(r[0,:]**2+r[1,:]**2)
        r2 = r2**(1/2)
        ru = mtmp@r
        ru = ru/np.sqrt(ru[0,:]**2+ru[1,:]**2)
        u = -(1/2/np.pi/r2*ru)*Ws
        rhs[i] -= np.sum((np.reshape(Cnorm[:,i],(1,2)))@u)
    # plt.imshow(mat)
    # plt.show()
    sol = np.linalg.solve(mat,rhs)
    
    return sol

def gen_arf(aoa, N):
    aoa = -np.radians(aoa)
    Vpos = np.linspace(0,1,N)
    Cpos = Vpos+2/(N-1)/4
    # Cpos[-1] = 1
    Cnorm = np.ones((2,N))
    Cnorm[0,:] = 0
    z = np.zeros_like(Vpos)
    Vpos = np.vstack((Vpos,z))
    Cpos = np.vstack((Cpos,z))

    mat = np.array([[np.cos(aoa),-np.sin(aoa)],[np.sin(aoa),np.cos(aoa)]])
    dp = np.array([-0.25,0]).reshape((2,1))
    Vpos = (mat@(Vpos+dp))-dp
    Cpos = (mat@(Cpos+dp))-dp
    Cnorm = mat@Cnorm

    return Vpos, Cpos, Cnorm

def advect_wake(Vpos, Vs, Wpos, Ws, Vinf):
    """
    Vpos: 2xN array
    Wpos: 2xM array
    Ws: 2xM array
    """
    M = Wpos.shape[1]
    vel = np.zeros_like(Wpos)
    mtmp = np.array([[0,1.0],[-1,0]])

    for i in range(M):
        r = np.reshape(Wpos[:,i],(2,1))-Vpos
        r2 = np.sqrt(r[0,:]**2+r[1,:]**2)
        r2 = r2**(1/2)
        ru = mtmp@r
        ru = ru/np.sqrt(ru[0,:]**2+ru[1,:]**2)
        u = -(1/2/np.pi/r2*ru)*Vs
        u = np.nan_to_num(u)
        vel[:,i] = np.sum(u, axis=1)
        
        r = np.reshape(Wpos[:,i],(2,1))-Wpos
        r2 = np.sqrt(r[0,:]**2+r[1,:]**2)
        r2 = r2**(1/2)
        ru = mtmp@r
        ru = ru/np.sqrt(ru[0,:]**2+ru[1,:]**2)
        u = -(1/2/np.pi/r2*ru)*Ws
        u = np.nan_to_num(u)
        vel[:,i] += np.sum(u, axis=1)
        vel[0,i] += Vinf

    return vel

def streamfunction(N, Vpos, Vs, Wpos, Ws):
    
    px = np.hstack((Vpos[0,:],Wpos[0,:]))
    py = np.hstack((Vpos[1,:],Wpos[1,:]))
    x = np.linspace(1,np.max(px),N)
    y = np.linspace(-1,1,N)
    xx,yy = np.meshgrid(x,y)
    s = np.hstack((Vs,Ws))
    # s = np.clip(s,-1,1)
    sf = yy.copy()
    

    for i in range(px.shape[0]):
        r = np.sqrt((xx-px[i])**2+(yy-py[i])**2)
        sf -= s[i]/2/np.pi*np.log(r)

    plt.contourf(xx,yy,sf,50)
    s = plt.scatter(px,py,c=s)
    plt.colorbar(s)
    plt.show()

def solve(Vinf, k, N, dt, T, aoamax):
    Nt = int(T/dt)

    Wpos = np.zeros((2,1))
    Ws = np.zeros(1)
    Cl = [0]
    a = []
    omega = k*2*Vinf
    for i in range(Nt):
        print(dt*i, Ws.shape)
        aoa = np.sin(dt*i*omega*np.pi)*aoamax
        a.append(aoa)
        Vpos, Cpos, Cnorm = gen_arf(aoa, N)
        circ = solve_steady(Vpos, Cpos, Cnorm, Wpos, Ws, Vinf)
        Cl.append(-np.sum(circ))
        vel = advect_wake(Vpos,circ,Wpos,Ws,Vinf)
        Wpos += vel*dt
        Wnew = (Vpos[:,-1]).reshape((2,1))
        Wpos = np.hstack((Wnew,Wpos))
        Ws = np.hstack((-1*(Cl[-1]-Cl[-2])*dt*np.sqrt(vel[0,0]**2+vel[1,0]**2),Ws))
        d = np.sum(Wpos**2,axis=0)
        Wpos = Wpos[:,d<(20**2)]
        Ws = Ws[d<(20**2)]
        # Nm = int(2*np.pi/omega/dt)
        # if Ws.shape[0] >= Nm:
        #     Ws = Ws[:Nm]
        #     Wpos = Wpos[:,:Nm]
    Cl=Cl[1:]
    plt.plot(Cl)
    a = np.array(a)
    plt.plot(np.radians(a)*2*np.pi,"--")
    plt.show()

    return Vpos, circ, Wpos, Ws

plt.show()
streamfunction(1000,*solve(1,1,50,0.05,20,5 ))
