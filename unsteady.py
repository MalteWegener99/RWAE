import numpy as np
import matplotlib.pyplot as plt

# CCW is positive

def solve_steady(Vpos, Cpos, Cnorm, Wpos, Ws, Vinf, Vnorm, lastcirc):
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
    for i in range(N-1):
        # Influences at controlpoint i
        # u,v = Gamma/2/pi/r * [[0,1][-1,0]] (cp-Vpos)
        r = np.reshape(Cpos[:,i],(2,1))-Vpos
        r2 = np.sqrt(r[0,:]**2+r[1,:]**2)
        ru = mtmp@r
        ru = ru/np.sqrt(ru[0,:]**2+ru[1,:]**2)
        u = -(1/2/np.pi/r2*ru)
        mat[i,:] = (np.reshape(Cnorm[:,i],(1,2)))@u
        

        rhs[i] = -Vinf*Cnorm[0,i]+Vnorm[i]
        r = np.reshape(Cpos[:,i],(2,1))-Wpos
        r2 = np.sqrt(r[0,:]**2+r[1,:]**2)
        ru = mtmp@r
        ru = ru/np.sqrt(ru[0,:]**2+ru[1,:]**2)
        u = -(1/2/np.pi/r2*ru)*Ws
        rhs[i] -= np.sum((np.reshape(Cnorm[:,i],(1,2)))@u)

    mat[-1,:] = 1
    rhs[-1] = lastcirc
    # plt.imshow(mat)
    # plt.show()
    sol = np.linalg.solve(mat,rhs)
    
    return sol[:-1], sol[-1]

def gen_arf(aoa, N,dt):
    aoa = -np.radians(aoa)
    Vpos = np.linspace(0,1,N)
    Cpos = Vpos+3/(N-1)/4
    Vpos = Vpos+1/(N-1)/4
    Vpos[-1] = 1+dt*0.25
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

def velocityfield(N, Vpos, Vs, Wpos, Ws,_,__):
    
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
    plt.contour(xx,yy,cp,50,colors="k")
    plt.scatter(Vpos[0,:],Vpos[1,:])
    plt.colorbar(s)
    # plt.axis("equal")
    plt.show()

def pressurefield(N, Vpos, Vs, Wpos, Ws,_,__):
    
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
    plt.scatter(Vpos[0,:],Vpos[1,:])
    plt.colorbar(s)
    # plt.axis("equal")
    plt.show()

def solve( k, N, dt, T, aoamax, include_acceleration=True):
    Vinf = 1
    Wpos = np.zeros((2,1))
    Ws = np.zeros(1)
    Cl = [0]
    a = [0]
    t = [0]
    o = [0]
    omega = k*2*Vinf
    print(omega,"\n\n")
    T = 2*np.pi/omega*T
    dt = 2*np.pi/omega*dt
    Nt = int(T/dt)

    for i in range(1,Nt+1):
        print(i/(Nt+1), Ws.shape,end="                                \r")
        aoa = np.sin(dt*i*omega)*aoamax
        a.append(aoa)
        t.append(dt*i)
        Vpos, Cpos, Cnorm = gen_arf(aoa, N,dt)
        _, Cpos2, _ = gen_arf(0, N,dt)
        Cpos2 = Cpos2[0,:]
        om = omega*np.radians(aoamax)*np.cos(dt*i*omega)
        o.append(om)
        Vnorm = -(Cpos2-0.25)*om*(1 if include_acceleration else 0)
        circ, diff = solve_steady(Vpos, Cpos, Cnorm, Wpos, Ws, Vinf, Vnorm, -Cl[-1]/2)
        # diff *= dt
        Cl.append(-np.sum(circ)*2)
        Wnew = (Vpos[:,-1]).reshape((2,1))
        Wpos = np.hstack((Wnew,Wpos))
        Ws = np.hstack((diff,Ws))#w
        vel = advect_wake(Vpos[:,:-1],circ,Wpos,Ws,Vinf)
        Wpos += vel*dt

        d = np.sum(Wpos**2,axis=0)
        Wpos = Wpos[:,d<(20**2)]
        Ws = Ws[d<(20**2)]
        # Nm = int(2*np.pi/omega/dt)
        # if Ws.shape[0] >= Nm:
        #     Ws = Ws[:Nm]
        #     Wpos = Wpos[:,:Nm]
    print(max(Cl))
    plt.plot(a,Cl)
    a = np.array(a)
    plt.plot(a,np.radians(a)*2*np.pi,"--")
    plt.grid(which="both")
    plt.show()
    exit()

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
velocityfield(1000,*solve(0.1,100,0.005,2, 10))