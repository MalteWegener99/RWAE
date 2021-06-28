import numpy as np
import matplotlib.pyplot as plt

# CCW is positive

def solve_steady(Vpos, Cpos, Cnorm, Wpos, Ws, Vinf, Vnorm):
    """
    Vpos: 2xN array
    Cpos: 2xN array
    Cnorm: 2xN array
    Wpos: 2xM array
    Ws: 2xM array
    Vnorm: 1xN array

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
        ru = mtmp@r
        ru = ru/np.sqrt(ru[0,:]**2+ru[1,:]**2)
        u = -(1/2/np.pi/r2*ru)
        mat[i,:] = (np.reshape(Cnorm[:,i],(1,2)))@u
        

        rhs[i] = -Vinf*Cnorm[0,i]-Vnorm[i]
        r = np.reshape(Cpos[:,i],(2,1))-Wpos
        r2 = np.sqrt(r[0,:]**2+r[1,:]**2)
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
    Vpos = Vpos[:-1]
    Cpos = Vpos+3/(N-1)/4
    Vpos = Vpos+1/(N-1)/4
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
        ru = mtmp@r
        ru = ru/np.sqrt(ru[0,:]**2+ru[1,:]**2)
        u = -(1/2/np.pi/r2*ru)*Vs
        u = np.nan_to_num(u)
        vel[:,i] = np.sum(u, axis=1)
        
        r = np.reshape(Wpos[:,i],(2,1))-Wpos
        r2 = np.sqrt(r[0,:]**2+r[1,:]**2)
        ru = mtmp@r
        ru = ru/np.sqrt(ru[0,:]**2+ru[1,:]**2)
        u = -(1/2/np.pi/r2*ru)*Ws
        u = np.nan_to_num(u)
        vel[:,i] += np.sum(u, axis=1)
        vel[0,i] += Vinf

    return vel

def streamfunction(N, Vpos, Vs, Wpos, Ws,_,__):
    
    px = np.hstack((Vpos[0,:],Wpos[0,:]))
    py = np.hstack((Vpos[1,:],Wpos[1,:]))
    x = np.linspace(-1,max(2,np.max(px)),N)
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
    # plt.axis("equal")
    plt.show()

def velocityfield(N, Vpos, Vs, Wpos, Ws,name):
    
    px = np.hstack((Vpos[0,:],Wpos[0,:]))
    py = np.hstack((Vpos[1,:],Wpos[1,:]))
    xo = np.linspace(-1,max(2,np.max(px)),N)
    yo = np.linspace(-1,1,N)
    xx,yy = np.meshgrid(xo,yo)
    s = np.hstack((Vs,Ws))
    # s = np.clip(s,-1,1)
    sf = yy.copy()
    # sf = 0
    

    for i in range(px.shape[0]):
        r = np.sqrt((xx-px[i])**2+(yy-py[i])**2)
        r = np.clip(r,0.02,np.max(r))
        sf -= s[i]/2/np.pi*np.log(r)
    uy = -np.gradient(sf,xo[1]-xo[0],axis=1)
    ux = np.gradient(sf,yo[1]-yo[0],axis=0)
    um = np.sqrt(ux**2+uy**2)
    # um = np.clip(um,0,3)
    s=plt.contourf(xx,yy,um,50)
    cp = 1-um**2
    plt.plot(Vpos[0,:],Vpos[1,:],"k")

    plt.colorbar(s)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    # plt.axis("equal")
    plt.savefig(name)

def pressurefield(N, Vpos, Vs, Wpos, Ws,name):
    
    px = np.hstack((Vpos[0,:],Wpos[0,:]))
    py = np.hstack((Vpos[1,:],Wpos[1,:]))
    xo = np.linspace(-1,max(2,np.max(px)),N)
    yo = np.linspace(-1,1,N)
    xx,yy = np.meshgrid(xo,yo)
    s = np.hstack((Vs,Ws))
    # s = np.clip(s,-1,1)
    sf = yy.copy()
    # sf = 0
    

    for i in range(px.shape[0]):
        r = np.sqrt((xx-px[i])**2+(yy-py[i])**2)
        r = np.clip(r,0.02,np.max(r))
        sf -= s[i]/2/np.pi*np.log(r)
    uy = -np.gradient(sf,xo[1]-xo[0],axis=1)
    ux = np.gradient(sf,yo[1]-yo[0],axis=0)
    um = np.sqrt(ux**2+uy**2)
    cp = 1-um**2
    # um = np.clip(um,0,3)
    s=plt.contourf(xx,yy,cp,50)
    plt.plot(Vpos[0,:],Vpos[1,:],"k")
    plt.colorbar(s)
    plt.xlabel("x/c")
    plt.ylabel("y/c")
    # plt.axis("equal")
    plt.savefig(name)

def solve( k, N, dt, T, f_aoa, include_acceleration=False):
    Vinf = 1
    Wpos = np.zeros((2,1))
    Ws = np.zeros(1)
    Cl = [0]
    a = [0]
    omega = k*2*Vinf
    T = 2/omega*T
    Nt = int(T/dt)

    for i in range(Nt+1):
        print(i/(Nt+1), Ws.shape,end="                                \r")
        aoa = f_aoa(dt*i*omega*np.pi)
        a.append(aoa)
        Vpos, Cpos, Cnorm = gen_arf(aoa, N)
        _, Cpos2, _ = gen_arf(0, N)
        Cpos2 = Cpos2[0,:]
        Vnorm = (Cpos2-0.25)*(a[-1]-a[-2])/dt*(1 if include_acceleration else 0)
        circ = solve_steady(Vpos, Cpos, Cnorm, Wpos, Ws, Vinf, Vnorm)
        Cl.append(-np.sum(circ)*2)
        vel = advect_wake(Vpos,circ,Wpos,Ws,Vinf)
        Wpos += vel*dt
        Wnew = (Vpos[:,-1]+vel[:,0]*dt).reshape((2,1))
        Wpos = np.hstack((Wnew,Wpos))
        Ws = np.hstack((1/2*(Cl[-1]-Cl[-2])*dt*np.sqrt(vel[0,0]**2+vel[1,0]**2),Ws))
        d = np.sum(Wpos**2,axis=0)
        Wpos = Wpos[:,d<(20**2)]
        Ws = Ws[d<(20**2)]
        # Nm = int(2*np.pi/omega/dt)
        # if Ws.shape[0] >= Nm:
        #     Ws = Ws[:Nm]
        #     Wpos = Wpos[:,:Nm]
    Cl=Cl[1:]
    a=a[1:]
    plt.plot(Cl)
    a = np.array(a)
    plt.plot(np.radians(a)*2*np.pi,"--")
    plt.show()

    return Vpos, circ, Wpos, Ws, a, Cl

def steady_aoa(N, aoa):
    Wpos = np.zeros((2,1))
    Ws = np.zeros(1)
    Vpos, Cpos, Cnorm = gen_arf(aoa, N)
    _, Cpos2, _ = gen_arf(0, N)
    Cpos2 = Cpos2[0,:]
    Vnorm = (Cpos2-0.25)*0
    circ = solve_steady(Vpos, Cpos, Cnorm, Wpos, Ws, 1, Vnorm)
    Cl = (-np.sum(circ)*2)
    return Vpos, circ, Wpos, Ws, aoa, Cl

plt.show()
f = lambda x: np.sin(x)*90
f = lambda x: 0 if x < np.pi else 90
s = steady_aoa

def make(a):
    s = steady_aoa(100,a)
    plt.clf()
    pressurefield(500,*s[:4],"pressure_{}deg.png".format(a))
    plt.clf()
    velocityfield(500,*s[:4],"velocity_{}deg.png".format(a))

for a in [-10,0,10]:
    make(a)