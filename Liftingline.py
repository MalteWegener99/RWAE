from typing import Final
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def trailingvort(R, xs, rev, shift, N):
    #       ^ r
    #       |
    #  t    |
    # <-----
    # Is right handed so STFU

    """
    Returns:
    The wake elements emitted from a point
    
    Arguments:
    R:      The radial location
    xs:     Streamwise distance covered by the wake in 1 revolution
    rev:    Number of revolutions tot take into account
    shift:  angular shift from the upward location. Used for multiblade rotors
    N:      Number of wake elements
    """

    angle = np.linspace(0,-np.pi*rev*2, (N))
    x = np.linspace(0,xs*rev,N)
    t = np.sin(angle+shift)*R
    r = np.cos(angle+shift)*R

    return x,t,r

def calc_induction(Ct):
    """
    Returns:
    Average axial induction factor, taking into account Glauert Correction
    
    Arguments:
    Ct:     Thrust Coefficient
    """
    CT1=1.816;
    CT2=2*np.sqrt(CT1)-CT1
    if Ct >= CT2:
        return (1 + (Ct-CT1)/(4*(np.sqrt(CT1)-1)))
    else:
        return (1/2-np.sqrt(1-Ct)/2)

def plot_wakesystem(Rp, xs, rev, Nw, Nr, Nb):
    """
    Returns:
    Plots the wake system
    
    Arguments:
    Rp:     Radial positions
    xs:     Streamwise distance covered during 1 revolution
    rev:    Number of revolutions
    Nw:     Number of wake elements per filament
    Nr:     Number of radial elements
    Nb:     Number of blades
    """
    ax = plt.gca(projection="3d")
    for i in range(Rp.shape[0]):
        for shift in np.linspace(0, 2*np.pi, Nb, endpoint=False):
            ax.plot3D(*trailingvort(Rp[i],xs,rev,shift,Nw), "r")

    plt.show()

def calc_midpoint_dir(x, t, r, invert=False):
    """
    Returns:
    Midpoint and direction vector for a 3d line

    Arguments:
    x:      Frist Coordinate
    t:      Second Coordinate
    r:      Third Coordinate

    Optional Keyword Arguments:
    invert: specifies if direction vector should be inverted
    """
    xm = (x[1:]-x[0:-1])/2+x[0:-1]
    tm = (t[1:]-t[0:-1])/2+t[0:-1]
    rm = (r[1:]-r[0:-1])/2+r[0:-1]

    factor = -1 if invert else 1

    dx = factor*(x[1:]-x[0:-1])
    dt = factor*(t[1:]-t[0:-1])
    dr = factor*(r[1:]-r[0:-1])

    return np.array([xm,tm,rm]).T, np.array([dx,dt,dr]).T

def cross(a,b):
    """
    Returns:
    Cross product of a and b where both are (N,3) shape

    Arguments:
    a:      First Vector
    b:      Second Vector
    """
    e1 = a[:,1]*b[:,2] - a[:,2]*b[:,1]
    e2 = a[:,2]*b[:,0] - a[:,0]*b[:,2]
    e3 = a[:,0]*b[:,1] - a[:,1]*b[:,0]

    return np.array([e1,e2,e3]).T

def norm(a, limit=False, CORE=0.00001):
    """
    Returns:
    norm of vectors a given in (N,3) shape

    Arguments:
    a:      Vector

    Optional Keyword Arguments:
    limit: Limit to CORE min
    CORE: limitdistance
    """

    n = np.sqrt(a[:,0]**2+a[:,1]**2+a[:,2]**2)
    if limit:
        n = np.maximum(n,CORE*np.ones_like(n))
    return np.array([n]).T


def make_vel_mat(Rp, rev, Nw, Nb, xs, offset=np.array([0,0,0]), phase_shift=0):
    """
    Returns:
    The matrix used to calculate induced velocities from the Circulation Gama
    Is of shape (N-1) x (N-1) where N is the size of Rp

    Arguments:
    Rp:     points on the rotorblade, separating the chord elements
    rev:    Number of revolutions tot take into account
    Nw:     Number of wake elements for each filament
    Nb:     Number of blades, equally spaced obv
    xs:     Streamwise distance covered by the wake in 1 revolution

    Optional Keyword Arguments:
    offset:         specifies an additional offset for thr rotor from the wake, used for multiple rotors
    phase_shift:    Phase difference between wake and rotorm used for multiple rotors

    Notes:
    Computational cost mainly scales with size of Rp and Nb, but not with Nw
    """
    Nr = Rp.shape[0]-1
    Rc = (Rp[1:]-Rp[0:-1])/2+Rp[0:-1]
    matu = np.zeros((Nr,Nr)) # x
    matv = np.zeros((Nr,Nr)) # t
    matw = np.zeros((Nr,Nr)) # r

    if np.abs(phase_shift) > np.pi:
        print("Warning offset is probably given in deg not radians")


    for i in range(Rc.shape[0]):
        for shift in np.linspace(0, 2*np.pi, Nb, endpoint=False):
            # Calculate wake geom of the lower end
            posl, dirl = calc_midpoint_dir(*trailingvort(Rp[i], xs, rev, shift+phase_shift, int(Nw[i])), invert=True)
            # Calculate wake geom of the upper end
            posu, diru = calc_midpoint_dir(*trailingvort(Rp[i+1], xs, rev, shift+phase_shift, int(Nw[i+1])), invert=False)
            posc, dirc = np.array([0,Rc[i]*np.sin(shift+phase_shift),Rc[i]*np.cos(shift+phase_shift)]), np.array([0,(Rp[i+1]-Rp[i])*np.sin(shift+phase_shift),(Rp[i+1]-Rp[i])*np.cos(shift+phase_shift)])
            posc = np.reshape(posc,(1,3))
            dirc = np.reshape(dirc,(1,3))
            # u = Gamma/4/pi/r^3*dir x r

            #Influence of i on j so we set mat[j,i]
            for j in range(Rc.shape[0]):
                pos = np.array([0,0,Rc[j]])
                rl = (pos-posl)+offset
                ru = (pos-posu)+offset
                rc = (pos-posc)+offset


                cl = cross(dirl,rl)/norm(rl,True)**3
                cu = cross(diru,ru)/norm(ru,True)**3
                # Deal with 0 distance
                if norm(rc) > 0:
                    cc = cross(dirc,rc)/norm(rc,True)**3
                else:
                    cc = np.zeros_like(rc)

                matu[j,i] += (np.sum(cl[:,0])+np.sum(cu[:,0])+np.sum(cc[:,0]))/4/np.pi
                matv[j,i] += (np.sum(cl[:,1])+np.sum(cu[:,1])+np.sum(cc[:,1]))/4/np.pi
                matw[j,i] += (np.sum(cl[:,2])+np.sum(cu[:,2])+np.sum(cc[:,2]))/4/np.pi

    # fig, ax = plt.subplots(1,3)
    # cb = ax[0].imshow(matu)
    # fig.colorbar(cb, ax=ax[0])
    # cb = ax[1].imshow(matv)
    # fig.colorbar(cb, ax=ax[1])
    # cb = ax[2].imshow(matw)
    # fig.colorbar(cb, ax=ax[2])
    # plt.show()

    return matu, matv, matw

class Airfoil:
    """
    Wrapper class for Xfoil airfoil
    """
    def __init__(self, file) -> None:
        self.polar = np.loadtxt(file)

    def Cl(self, alpha):
        return np.interp((alpha), self.polar[:,0], self.polar[:,1])

    def Cd(self, alpha):
        return np.interp((alpha), self.polar[:,0], self.polar[:,2])


def cospacing(x1, x2, N):
    '''
    Returns:
    Cosine spaced array of N values between x1 and x2

    Arguments:
    x1:     Start point
    x2:     End point
    N:      Number of points
    '''

    x = np.linspace(0,1,N)
    x = 1/2*(1-np.cos(x*np.pi))
    x = x*(x2-x1)
    x = x + x1
    return x


def calc_Forces(u_a,u_t, u_r, chord, twist, arf, rho):
    '''
    Returns:
    Normal and tangential force as well as circulation

    Arguments:
    u_a:    axial velocity
    u_t:    tangential velocity
    u_r:    radial velocity
    chord:  chord of the airfoil
    twist:  twist of the airfoil
    arf:    airfoil to query
    rho:    Freestream density
    '''
    vmag2 = u_a**2 + u_t**2 
    phi = np.arctan2(u_a,(u_t))
    alpha = np.degrees(phi)+twist
    cl = arf.Cl(alpha)
    cd = arf.Cd(alpha)
    lift = 0.5*vmag2*cl*chord*rho
    drag = 0.5*vmag2*cd*chord*rho
    fnorm = lift*np.cos(phi)+drag*np.sin(phi)
    ftan =  lift*np.sin(phi)-drag*np.cos(phi)
    # Sign conventions are weirtd in wind energy
    circ = -lift/rho/np.sqrt(vmag2)
    return fnorm , ftan, circ

def Performance_BEM_style(Rh, Rt, chord, twist, pitch, Nb, TSR, Uinf, rho, spacing="cosine", rev=60, Nr=50, lw=0.1, Airfoil=Airfoil("DU95.csv"), iter_max=5000, epsilon=1e-5, a=0.2675, multiple=False, offset=0, phaseshift=0, iteratea=True):
    """
    Returns:
    Performance of the Rotor specified

    Arguments:
    Rh:     Beginning radius of the blade
    Rt:     Tipradius of the blade
    chord:  callable function that calculates the chord at a Radius
    twist:  callable function that calculates the twist at a Radius
    pitch:  blade pitch
    Nb:     Number of blades
    TSR:    Tipspeed ratio
    Uinf:   Freestream velocitys
    rho:    Freestream density

    Optional Keyword Arguments:
    spacing:    How the radial positions are distributed
    rev:        Number of wake revolutions
    Nc:         Radial elements
    lw:         length of a segment in the wake in meters
    Airfoil:    Airfoil polar
    iter_max:   Maximum iterations
    epsilon:    Convergence criteria
    a:          Induction Factor starting point for iteration
    multiple:   If 2 rotors are to be computed next to each other
    offset:     Distance between 2 rotors in Diameters of rotor
    phaseshift: phase offset between the 2 rotors
    iteratea:   Iterate a to convergence or not

    Notes:
    Computational cost mainly scales with size of Rp and Nb, but not with Nw
    """
    #Rotor generation
    Rp = np.zeros(Nr)
    if spacing == "constant":
        Rp = np.linspace(Rh,Rt,Nr)
    elif spacing == "cosine":
        Rp = cospacing(Rh,Rt,Nr)

    else:
        raise ValueError

    dr = (Rp[1:]-Rp[0:-1])
    Rc = (Rp[1:]-Rp[0:-1])/2+Rp[0:-1]

    chord = chord(Rc/Rt)
    twist = twist(Rc/Rt)+pitch

    # Outer loop to converge induction factor

    for j in range(12):
        # calc xs 
        Omega = Uinf*TSR/Rt
        xs = Uinf*(1-a)*2*np.pi/Omega
        Nw = np.round(np.sqrt((Rp*2*np.pi)**2+xs**2)/lw, decimals=1)

        assert xs > 0

        # plot_wakesystem(Rp,xs,rev,Nw,Nr,Nb)

        # Setting up influence matrices
        matx, matt, matr = make_vel_mat(Rp, rev, (Nw*rev), Nb, xs)

        # This approach may seem very sophisticated, but it gives the same results as the naive one, and that is twice as fast
        if multiple:
            # Constructing a block matrix with the the influence of the main turbine in the left and secondary in the right
            matxtemp, matttemp, matrtemp = make_vel_mat(Rp, rev, (Nw*rev), Nb, xs, offset=np.array([0,offset*Rt*2,0]), phase_shift=phaseshift)
            matx = np.block([matx,matxtemp])
            matt = np.block([matt,matttemp])
            matr = np.block([matr,matrtemp])

            # Same for the secondary matrix
            # THe other turbine is of course in teh relative other direction and phaseshift
            matxtemp, matttemp, matrtemp = make_vel_mat(Rp, rev, (Nw*rev), Nb, xs)
            matx2, matt2, matr2 = make_vel_mat(Rp, rev, (Nw*rev), Nb, xs, offset=np.array([0,-offset*Rt*2,0]), phase_shift=-phaseshift)
            matx2 = np.block([matx2,matxtemp])
            matt2 = np.block([matt2,matttemp])
            matr2 = np.block([matr2,matrtemp])
            
            
        Circ = np.zeros_like(Rc)
        if multiple:
            Circ2 = np.zeros_like(Rc)
        # Inner iteration loop loop
        for i in range(iter_max):
            Circt = Circ

            if multiple:
                Circt = np.hstack([Circ,Circ2])

            ux = matx@Circt
            ut = matt@Circt
            ur = matr@Circt

            if multiple:
                ux2 = matx2@Circt
                ut2 = matt2@Circt
                ur2 = matr2@Circt

            Fn, Ft, Circn = calc_Forces(Uinf+ux, Omega*Rc+ut, ur, chord, twist, Airfoil, rho)
            # plt.plot(Rc,np.arctan2(Uinf,(Omega*Rc)))
            # plt.show()
            if multiple:
                Fn2, Ft2, Circn2 = calc_Forces(ux2+Uinf, Omega*Rc+ut2, ur2, chord, twist, Airfoil, rho)

            # I assume that if Circ converges so does Circ2, sue me
            if (np.all(np.abs(Circn-Circ) < epsilon)):
                print("Inner converged in {} iterations".format(i))
                break

            Circ = (Circn+Circ)/2
            d = Circn-Circ
            if multiple:
                Circ2 = (Circn2+Circ2)/2


        else:
            print(np.sum(np.abs(d)))
            print("May not have converged")

        n = 1/(2*np.pi/Omega)
        J = Uinf/n/Rt/2
        A = np.pi*Rt**2

        Ct = np.sum(Fn*dr)*Nb/(0.5*rho*A*Uinf**2)
        Cp = np.sum(Ft*Rc*dr)*Nb*Omega/(0.5*rho*A*Uinf**3)
        eta = Ct/Cp*J

        if iteratea:
            an = np.sum((-(ux)/Uinf)*dr)/np.sum(dr)
        else:
            an = a
            
        if np.abs(an-a) < epsilon:
            a = an
            print("Outer converged in {} iterations".format(j))

            break
        a = an

    else:
        print("May not have converged")
        

    print("CT: ", Ct)
    print("CP: ", Cp)
    print("eta: ", eta)
    print("a: ", np.sum((-(ux)/Uinf)*dr)/np.sum(dr))


    phi = np.arctan2(ux+Uinf,Omega*Rc+ut)
    alpha = np.degrees(phi)+twist

    retval = {}
    retval["R"] = Rc
    retval["Rel"] = Rc/Rt
    retval["CT"] = Ct
    retval["CP"] = Cp
    retval["Gamma"] = Circ/(Uinf**2*np.pi)*(Nb*Omega)
    retval["Fn"] = Fn/(0.5*rho*Uinf**2*Rt)
    retval["Ft"] = Ft/(0.5*rho*Uinf**2*Rt)
    retval["aoa"] = alpha
    retval["phi"] = np.degrees(phi)
    retval["eta"] = eta
    retval["J"] = J
    #Chimmy changa for the next iteration if thats wanted
    retval["a"] = np.sum((-(ux)/Uinf)*dr)/np.sum(dr)
    retval["as"] = -ux/Uinf
    retval["at"] = -ut/(Omega*Rc)




    return retval



Rotor = (   50*0.2, #Root radius
            50, #Tip radius
            lambda r_R: 3*(1-r_R)+1, # meters
            lambda r_R: -14*(1-r_R), # degrees
            2, # degrees
            3) #Numebr of blades

Flow = (10,1)

def cos_vsconst():
    sol = Performance_BEM_style(*Rotor, 10, *Flow, spacing="constant", Nr=30, rev=200, lw=1, multiple=False, offset=1, a=0.5)
    plt.plot(sol["Rel"],-sol["Gamma"],"tab:green",label="Constant spacing TSR=10")
    sol = Performance_BEM_style(*Rotor, 8, *Flow, spacing="constant", Nr=30, rev=200, lw=1, multiple=False, offset=1, a=0.2469)
    plt.plot(sol["Rel"],-sol["Gamma"],"tab:orange",label="Constant spacing TSR=8")
    sol = Performance_BEM_style(*Rotor, 6, *Flow, spacing="constant", Nr=30, rev=200, lw=1, multiple=False, offset=1, a=0.2469)
    plt.plot(sol["Rel"],-sol["Gamma"],"tab:blue",label="Constant spacing TSR=6")

    sol = Performance_BEM_style(*Rotor, 10, *Flow, spacing="cosine", Nr=30, rev=200, lw=1, multiple=False, offset=1, a=0.5)
    plt.plot(sol["Rel"],-sol["Gamma"],"--", c="tab:green",label="Cosine spacing TSR=10")
    sol = Performance_BEM_style(*Rotor, 8, *Flow, spacing="cosine", Nr=30, rev=200, lw=1, multiple=False, offset=1, a=0.2469)
    plt.plot(sol["Rel"],-sol["Gamma"],"--", c="tab:orange",label="Cosine spacing TSR=8")
    sol = Performance_BEM_style(*Rotor, 6, *Flow, spacing="cosine", Nr=30, rev=200, lw=1, multiple=False, offset=1, a=0.2469)
    plt.plot(sol["Rel"],-sol["Gamma"],"--", c="tab:blue",label="Cosine spacing TSR=6")
    plt.legend()
    plt.xlabel("r/R")
    plt.ylabel("$\Gamma$")


    plt.savefig("consvcos")

def wakedisc():
    ls = np.logspace(-0.3,0.3,10)
    w = []
    w2 = []
    for l in ls:
        sol = Performance_BEM_style(*Rotor, 8, *Flow, spacing="constant", Nr=30, rev=200, lw=l, multiple=False, offset=1, a=0.246)
        w.append(sol["CT"])
        w2.append(sol["CP"])

    fig, ax1 = plt.subplots()
    w = np.array(w)
    w2 = np.array(w2)

    color = 'tab:red'
    plt.xlabel("$L_f$")
    ax1.set_ylabel('$\Delta C_T$', color=color)
    ax1.loglog(ls[1:], np.abs(w[1:]-w[:-1]), "-o", color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('$\Delta C_P$', color=color)  # we already handled the x-label with ax1
    ax2.loglog(ls[1:], np.abs(w2[1:]-w2[:-1]), "--x", color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("convwakedisc")

def wakelen():
    ls = np.logspace(0,2.7,20)
    w = []
    w2 = []
    for l in ls:
        sol = Performance_BEM_style(*Rotor, 8, *Flow, spacing="constant", Nr=30, rev=l, lw=0.5, multiple=False, offset=1, a=0.246)
        w.append(sol["CT"])
        w2.append(sol["CP"])

    w = np.array(w)
    w2 = np.array(w2)

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    plt.xlabel("$N_r$")
    ax1.set_ylabel('$\Delta C_T$', color=color)
    ax1.loglog(ls[1:], np.abs(w[1:]-w[:-1]), "-o", color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('$\Delta C_P$', color=color)  # we already handled the x-label with ax1
    ax2.loglog(ls[1:], np.abs(w2[1:]-w2[:-1]), "--x", color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig("convwakelength")

# wakelen()
wakedisc()
# sol = Performance_BEM_style(*Rotor, 8, *Flow, spacing="constant", Nr=30, rev=200, lw=1, multiple=False, offset=1, a=0.2469)
# plt.plot(sol["Rel"],sol["aoa"])
# plt.plot(sol["Rel"],sol["phi"], "--")
# plt.show()

# plt.plot(sol["Rel"],sol["as"])
# plt.plot(sol["Rel"],sol["at"])
# plt.plot(sol["Rel"],-sol["Gamma"])

# plt.ylim([0,1.4])
# plt.show()
