import numpy as np
import matplotlib.pyplot as plt
import cmocean.cm as cms
import numba as nb
import scipy.optimize as opt
color = cms.haline

def calc_a(CT,yaw,xi):
    a = np.zeros(np.shape(CT))
    CT1=1.816;
    CT2=2*np.sqrt(CT1)-CT1
    a[CT>=CT2] = (1 + (CT-CT1)/(4*(np.sqrt(CT1)-1)))[CT>=CT2]
    a[CT<CT2] = (1/2-np.sqrt(1-CT)/2)[CT<CT2]
    # a = (1/2-np.sqrt(1-CT)/2)
    return  np.minimum(a,0.95)


def Prandtlf(r, TSR, NBlades, a):
    d = NBlades/2*np.sqrt(1+TSR**2*r**2/(1-a)**2)
    ft = 2/np.pi*np.arccos(np.exp(-d*(1-r)/r))
    fr = 2/np.pi*np.arccos(np.exp(-d*(r-0.2)/r))
    f = np.maximum(ft*fr,np.full_like(ft,1e-4))
    return f

class Airfoil:
    def __init__(self, file) -> None:
        self.polar = np.loadtxt(file)

    def Cl(self, alpha):
        return np.interp((alpha), self.polar[:,0], self.polar[:,1])

    def Cd(self, alpha):
        return np.interp((alpha), self.polar[:,0], self.polar[:,2])


def forces(u_a,u_t,chord, twist, arf, uinf, nb, dr, dpsi):
    '''
    Calculates the forces at the given position.
    '''
    vmag2 = u_a**2 + u_t**2
    phi = np.arctan2(u_a,u_t)
    alpha = np.degrees(phi)+twist
    cl = arf.Cl(alpha)
    cd = arf.Cd(alpha)
    lift = 0.5*vmag2*cl*chord
    drag = 0.5*vmag2*cd*chord
    fnorm = lift*np.cos(phi)+drag*np.sin(phi)
    ftan = lift*np.sin(phi)-drag*np.cos(phi)
    return fnorm , ftan


def solve(R, Uinf, tsr, chord_d, twist_d, NB,yaw, N):
    rho = 1#.225
    yaw = np.radians(yaw)
    # initiatlize variables
    N2 = 360
    dr = (0.8)/(N)*R
    dpsi = 360/N2/180*np.pi
    # ASk me for Ligma
    r = (np.linspace(0.2*R+dr/2,R-dr/2,N))
    psi = np.linspace(-180,180,N2)/180*np.pi
    r,psi = np.meshgrid(r,psi)
    Area = r*dr*dpsi
    arf = Airfoil('DU95.csv')
    chord = chord_d(r/R)
    twist = twist_d(r/R)
    a= np.full_like(r, 0) # axial induction
    al = np.full_like(r, 0) # tangential induction factor
    Omega = Uinf*tsr/R
    
    Niterations = 10000
    Erroriterations = 0.0000001 # error limit for iteration rpocess, in absolute value of induction
    # u_a = 0
    fnorm = 0
    for i in range(Niterations):
        Prandtl = Prandtlf(r/R,tsr,NB,a)
        chi = (0.6*a+1)*yaw
        k = 2*np.tan(0.5*chi)
        u_a = (np.cos(yaw)-a*(1+k*r/R*np.sin(psi)))*Uinf
        u_t = (1+al)*Omega*r-Uinf*np.sin(yaw)*np.cos(psi)
        fn,ft = forces(u_a, u_t, chord,twist,arf,Uinf,NB,dr,dpsi)
        fnorm =fn*dr*NB
        # CT = fnorm/(0.5*r*dr*Uinf**2*2*np.pi)
        CT = fnorm/(0.5*rho*r*dr*Uinf**2*2*np.pi)
        an = calc_a(CT,yaw,chi)
        ap = ft*dr*NB/(2*(2*np.pi*r)*Uinf**2*(1-a)*tsr*r/R)
        #Ligma Balls
        an = an/Prandtl
        ap = ap/Prandtl

        phi = np.arctan2(u_a,u_t)
        alpha = np.degrees(phi)+twist
        #// test convergence of solution, by checking convergence of axial induction
        if (np.all(np.abs(a-an) < Erroriterations)): 
            print(i,"iterations")
            break
        
        a = an/4+3*a/4
        al = ap/2+al/2
        
    else:
        print("Not converged")
    p0  = 101325
    rho = 1.225

    h_neginf    = np.full_like(a,p0 + (rho*Uinf**2)/2)
    h_bef       = 0.5*rho*(Uinf**2 -1*(Uinf*(1-a))**2) + p0 + 0.5*rho*Uinf*(1-a)
    h_aft       = 0.5*rho*( (Uinf*(1-2*a))**2 -1*(Uinf*(1-a))**2) + p0 + 0.5*rho*Uinf*(1-a)
    h_posinf    = p0 + (rho*(Uinf*(1-2*a))**2)/2

    h = [h_neginf,h_bef,h_aft,h_posinf]
    circ = (fn/np.sqrt(u_a**2+u_t**2)/rho)/(np.pi*Uinf**2/NB/(Uinf*tsr/R))
        
    return [a,al,r,psi,Prandtl,h,circ,alpha,np.degrees(phi),fnorm*rho,ft*dr*NB*rho,np.sum(4*a*(np.cos(yaw)+np.sin(yaw)*np.tan(chi/2)-a/np.cos(chi/2)**2)*Area)/(R**2*np.pi-(0.2*R)**2*np.pi),
    np.sum(4*a*(np.cos(yaw)+np.sin(yaw)*np.tan(chi/2)-a/np.cos(chi/2)**2)*(np.cos(yaw)-a)*Area)/(R**2*np.pi-(0.2*R)**2*np.pi)]


def solve_wrapper(TSR, yaw, N=50):
    pitch = 2 # degrees
    chord_distribution = lambda r_R: 3*(1-r_R)+1 # meters
    twist_distribution = lambda r_R: -14*(1-r_R)+pitch # degrees

    Uinf = 15 # unperturbed wind speed in m/s
    # TSR = 8 # tip speed ratio
    Radius = 50
    Omega = Uinf*TSR/Radius
    NBlades = 3
    # yaw = 0

    TipLocation_R =  1
    RootLocation_R =  0.2

    result = solve(Radius, Uinf,TSR,chord_distribution,twist_distribution,NBlades,yaw,N)
    return result

def optimization_objective(pa):
    p, a, b, m, c = pa
    #2,3,-14
    pitch = p*8#2 # degrees
    chord_distribution = lambda r_R: a*10*(1-r_R)+m*5+c*10*(1-r_R)**2# meters
    twist_distribution = lambda r_R: (12.5*b-12.5)*(1-r_R)+pitch # degrees

    Uinf = 15 # unperturbed wind speed in m/s
    TSR = 8 # tip speed ratio
    Radius = 50
    Omega = Uinf*TSR/Radius
    NBlades = 3
    # yaw = 0

    TipLocation_R =  1
    RootLocation_R =  0.2
    targetct = 0.75
    Ct,Cp = solve(Radius, Uinf,TSR,chord_distribution,twist_distribution,NBlades,0,50)[-2:]
    # maximize Cp while ct is constant
    return -Cp/Ct# +1*(np.abs(Ct-targetct))

def optimization_objective2(pa):
    p, a, b, m, c = pa
    #2,3,-14
    pitch = p*8#2 # degrees
    chord_distribution = lambda r_R: a*10*(1-r_R)+m*5+c*10*(1-r_R)**2# meters
    twist_distribution = lambda r_R: (12.5*b-12.5)*(1-r_R)+pitch # degrees

    Uinf = 15 # unperturbed wind speed in m/s
    TSR = 8 # tip speed ratio
    Radius = 50
    Omega = Uinf*TSR/Radius
    NBlades = 3
    # yaw = 0

    TipLocation_R =  1
    RootLocation_R =  0.2
    targetct = 0.75
    Ct,Cp = solve(Radius, Uinf,TSR,chord_distribution,twist_distribution,NBlades,0,50)[-2:]
    # maximize Cp while ct is constant
    return Cp,Ct,Cp/Ct

def optimize():
    res = opt.minimize(optimization_objective,[1,0,0,0.5,0],method="Powell",options={"maxfev":10000}, bounds=[(-1,1),(-1,1),(-1,1),(0,1),(-1,1)])
    print(res)
    print(optimization_objective2(res.x))


def polar_plot_ax(TSR, yaw):
    plt.clf()
    a = solve_wrapper(TSR,yaw)
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location("N")
    im = ax.contourf(a[3], a[2], a[0],25,cmap=color,label="Induction factor")
    ax.set_rmin(0)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("Images/axialinduction"+str(TSR)+str(yaw))

def polar_plot_ta(TSR, yaw):
    plt.clf()
    a = solve_wrapper(TSR,yaw)
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location("N")
    im = ax.contourf(a[3], a[2], a[1],25,cmap=color,label="Induction factor")
    ax.set_rmin(0)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("Images/tangetialinduction"+str(TSR)+str(yaw))

def polar_plot_aoa(TSR, yaw):
    plt.clf()
    a = solve_wrapper(TSR,yaw)
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location("N")
    im = ax.contourf(a[3], a[2], a[7],25,cmap=color,label="angle of attack [deg]")
    ax.set_rmin(0)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("Images/aoa"+str(TSR)+str(yaw))

def polar_plot_phi(TSR, yaw):
    plt.clf()
    a = solve_wrapper(TSR,yaw)
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location("N")
    im = ax.contourf(a[3], a[2], a[8],25,cmap=color,label="inflow angle [deg]")
    ax.set_rmin(0)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("Images/phi"+str(TSR)+str(yaw))

def polar_plot_thrust(TSR, yaw):
    plt.clf()
    a = solve_wrapper(TSR,yaw)
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location("N")
    im = ax.contourf(a[3], a[2], a[9],25,cmap=color,label="Thrust [N]")
    ax.set_rmin(0)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("Images/thrust"+str(TSR)+str(yaw))

def polar_plot_azimt(TSR, yaw):
    plt.clf()
    a = solve_wrapper(TSR,yaw)
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location("N")
    im = ax.contourf(a[3], a[2], a[10],25,cmap=color,label="Azimuthal force [N]")
    ax.set_rmin(0)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("Images/azimth"+str(TSR)+str(yaw))

def convergence():
    plt.clf()
    ns = np.linspace(2,500,25)
    res = []
    for n in ns:
        res.append(solve_wrapper(8,0,int(n))[-2])
    res = np.array(res)
    plt.loglog(ns,res)
    plt.tight_layout()
    plt.xlabel("Number of annuli")
    plt.ylabel("C_T")
    plt.savefig("Images/convergence")

def polar_plot_circ(TSR, yaw):
    plt.clf()
    a = solve_wrapper(TSR,yaw)
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location("N")
    im = ax.contourf(a[3], a[2], a[6],25,cmap=color)
    ax.set_rmin(0)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("Images/circulation"+str(TSR)+str(yaw))

def Malte():
    res = []
    tsrs = [6,8,10]
    col = ["tab:blue","tab:orange","tab:green"]
    for tsr in tsrs:
        res.append(solve_wrapper(tsr,0))

    #plot alpha and phi
    plt.clf()
    for i in range(len(tsrs)):
        plt.plot(res[i][2][0,:],res[i][7][0,:],"-",c=col[i],label=r"$\alpha$ TSR="+str(tsrs[i]))
        plt.plot(res[i][2][0,:],res[i][8][0,:],"--",c=col[i],label=r"$\phi$ TSR="+str(tsrs[i]))
    plt.legend()
    plt.xlabel("Spanwise position [m]")
    plt.ylabel("degrees")
    plt.grid()
    plt.tight_layout()
    plt.savefig("Images/aoaphi")

    plt.clf()
    for i in range(len(tsrs)):
        plt.plot(res[i][2][0,:],res[i][0][0,:],"-",c=col[i],label=r"$a_n$ TSR="+str(tsrs[i]))
        plt.plot(res[i][2][0,:],res[i][1][0,:],"--",c=col[i],label=r"$a_t$ TSR="+str(tsrs[i]))
    plt.legend()
    plt.xlabel("Spanwise position [m]")
    plt.ylabel("Induction factor")
    plt.grid()
    plt.tight_layout()
    plt.savefig("Images/spanwiseinduction")

    plt.clf()
    for i in range(len(tsrs)):
        plt.plot(res[i][2][0,:],res[i][9][0,:],"-",c=col[i],label=r"axial loading TSR="+str(tsrs[i]))
        plt.plot(res[i][2][0,:],res[i][10][0,:],"--",c=col[i],label=r"azimuthal loading TSR="+str(tsrs[i]))
    plt.legend()
    plt.xlabel("Spanwise position [m]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.tight_layout()
    plt.savefig("Images/loading")

    for i in range(len(tsrs)):
        print("CT TSR={}: {}".format(tsrs[i], res[i][-2]))
        print("CP TSR={}: {}".format(tsrs[i], res[i][-2]))

    plt.clf()
    for i in range(len(tsrs)):
        plt.plot(res[i][2][0,:],res[i][9][0,:],"-",c=col[i],label=r"axial loading TSR="+str(tsrs[i]))
        plt.plot(res[i][2][0,:],res[i][10][0,:],"--",c=col[i],label=r"azimuthal loading TSR="+str(tsrs[i]))
    plt.legend()
    plt.xlabel("Spanwise position [m]")
    plt.ylabel("Force [N]")
    plt.grid()
    plt.tight_layout()
    plt.savefig("Images/loading")

    for yaw in [0,15,30]:
        polar_plot_aoa(8,yaw)
        polar_plot_ax(8,yaw)
        polar_plot_azimt(8,yaw)
        polar_plot_phi(8,yaw)
        polar_plot_thrust(8,yaw)
        polar_plot_ta(8,yaw)

Malte()
convergence()
optimize()
res = solve_wrapper(8,0)
print(res[-1]/res[-2])
# polar_plot(8,15)
# yaw = np.linspace(0,30)
# cp = np.zeros_like(yaw)
# for i in range(yaw.shape[0]):
#     cp[i] = solve_wrapper(8,yaw[i])[-1]

# plt.plot(yaw,cp/np.max(cp))
# plt.show()

