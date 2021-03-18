import numpy as np
import matplotlib.pyplot as plt

def calc_a(CT, xi, yaw):
    B = np.cos(yaw)+np.sin(yaw)*np.tan(xi/2)
    K = 1/np.cos(xi/2)**2
    a = np.zeros(np.shape(CT))
    CT1=1.816;
    CT2=np.full_like(CT,2*np.sqrt(CT1)-CT1)
    a[CT>=CT2] = (1 + (CT-CT1)/(4*(np.sqrt(CT1)-1)))[CT>=CT2]
    # a[CT<CT2] = (B/2/K-np.sqrt((B/2/K)**2-1/4/K*CT))[CT<CT2]
    a = (B/2/K-np.sqrt((B/2/K)**2-1/4/K*CT))
    return a

def Prandtlf(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    temp1 = -NBlades/2*(tipradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1-axial_induction)**2))
    Ftip = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    Ftip[np.isnan(Ftip)] = 0
    temp1 = NBlades/2*(rootradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1-axial_induction)**2))
    Froot = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    Froot[np.isnan(Froot)] = 0
    return Froot*Ftip, Ftip, Froot

class Airfoil:
    def __init__(self, file) -> None:
        self.polar = np.loadtxt(file)

    def Cl(self, alpha):
        return np.interp((alpha), self.polar[:,0], self.polar[:,1])

    def Cd(self, alpha):
        return np.interp((alpha), self.polar[:,0], self.polar[:,2])


def forces(vnorm, vtan, chord, twist, arfoil):

    vmag2 = vnorm**2 + vtan**2
    inflowangle = np.arctan2(vnorm,vtan)
    alpha = twist + inflowangle*180/np.pi
    cl = arfoil.Cl(alpha)
    cd = arfoil.Cd(alpha)
    lift = 0.5*vmag2*cl*chord
    drag = 0.5*vmag2*cd*chord
    fnorm = lift*np.cos(inflowangle)+drag*np.sin(inflowangle)
    ftan = lift*np.sin(inflowangle)-drag*np.cos(inflowangle)
    gamma = 0.5*np.sqrt(vmag2)*cl*chord
    return fnorm , ftan


def solve(Uinf, rootradius_R, tipradius_R , Omega, Radius, NBlades, chord_d, twist_d, yaw, N):
    """
    solve balance of momentum between blade element load and loading in the streamtube
    input variables:
    Uinf - wind speed at infinity
    r1_R,r2_R - edges of blade element, in fraction of Radius ;
    rootradius_R, tipradius_R - location of blade root and tip, in fraction of Radius ;
    Radius is the rotor radius
    Omega -rotational velocity
    NBlades - number of blades in rotor
    """

    yaw = np.radians(yaw)
    # initiatlize variables
    r = np.linspace(rootradius_R,tipradius_R,N)
    dr = (tipradius_R-rootradius_R)/N*Radius
    Area = r*dr*2*np.pi*Radius
    arf = Airfoil('DU95.csv')
    chord = chord_d(r)
    twist = twist_d(r)
    a= np.full([N], 0)# axial induction
    aline = np.full([N], 0) # tangential induction factor
    
    Niterations = 10000
    Erroriterations =0.00001 # error limit for iteration rpocess, in absolute value of induction
    
    for i in range(Niterations):
        # ///////////////////////////////////////////////////////////////////////
        # // this is the block "Calculate velocity and loads at blade element"
        # ///////////////////////////////////////////////////////////////////////
        Urotor = Uinf*(1*np.cos(yaw)-a)# axial velocity at rotor
        Utan = (1+aline)*Omega*r*Radius # tangential velocity at rotor
        # calculate loads in blade segment in 2D (N/m)
        fnorm, ftan = forces(Urotor, Utan, chord, twist, arf)
        load3Daxial = fnorm*dr*NBlades # 3D force in axial direction
        # load3Dtan =loads[1]*Radius*(r2_R-r1_R)*NBlades # 3D force in azimuthal/tangential direction (not used here)
      
        # ///////////////////////////////////////////////////////////////////////
        # //the block "Calculate velocity and loads at blade element" is done
        # ///////////////////////////////////////////////////////////////////////

        # ///////////////////////////////////////////////////////////////////////
        # // this is the block "Calculate new estimate of axial and azimuthal induction"
        # ///////////////////////////////////////////////////////////////////////
        # // calculate thrust coefficient at the streamtube 
        # correct new axial induction with Prandtl's correction
        xi = (0.6*a+1)*yaw
        CT = load3Daxial/(0.5*Area*Uinf**2)
        CP = ftan*NBlades*dr*r*Radius*Omega/(0.5*Uinf**3*np.pi*Radius**2)
        
        # calculate new axial induction, accounting for Glauert's correction
        anew =  calc_a(CT, xi, yaw)
        Prandtl, Prandtltip, Prandtlroot = Prandtlf(r, rootradius_R, tipradius_R, Omega*Radius/Uinf, NBlades, anew); 
        Prandtl[Prandtl<0.0001] = 0.0001 # avoid divide by zero
        
        anew = anew *Prandtl# correct estimate of axial induction
        a = 0.75*a+0.25*anew # for improving convergence, weigh current and previous iteration of axial induction

        # calculate aximuthal induction
        aline = ftan*NBlades/(2*np.pi*Uinf*(1-a)*Omega*2*(r*Radius)**2)
        aline = aline * Prandtl# correct estimate of azimuthal induction with Prandtl's correction
        # ///////////////////////////////////////////////////////////////////////////
        # // end of the block "Calculate new estimate of axial and azimuthal induction"
        # ///////////////////////////////////////////////////////////////////////
        
        #// test convergence of solution, by checking convergence of axial induction
        if (np.all(np.abs(a-anew) < Erroriterations)): 
            print("iterations")
            print(i)
            break
        
    else:
        print("Not converged")
    return [a,aline,r,CT,np.trapz(CT,dx=dr),np.trapz(CP,dx=dr)]


def solve_wrapper(TSR, yaw):
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

    result = solve(Uinf, RootLocation_R, TipLocation_R , Omega, Radius, NBlades, chord_distribution, twist_distribution, yaw, 50)
    print(result[-1])
    plt.plot(result[3])
    plt.show()

solve_wrapper(8,0)