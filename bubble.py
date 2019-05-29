import numpy as np
import time
import csv
import os
import sys
from scipy.optimize import minimize
from scipy import integrate
from numpy.polynomial.chebyshev import Chebyshev

# maximal number of optimizer steps
MAX_ITER_IN_OPTIMIZATOR = 200

# [YOUNGS_MODULUS] = [eV / A**2]
YOUNGS_MODULUS = 22.0
# [RIGIDITY] = [eV]
RIGIDITY = 1.2

# Number of coefficients used in Chebyshev polynomials
N_COEF_U = 20
N_COEF_H = 20

# Desired density or mass precision
# [DENSITY_PRECISION] = g/ml
DENSITY_PRECISION = 1e-5
# [MASS_PRECISION] = 1
MASS_PRECISION = 1e-7

# Density interval with data avaliable for fluid energy
LOW_DENSITY_EDGE = 0.00010965
HIGH_DENSITY_EDGE = 0.47417


def get_density_from_pressure_wrapper(dens_gml, pressure_mpa):
    """
    Density of bulk argon from pressure for 300 K from NIST.
    Density should be in MPa.
    Return density in g/ml.
    """
    def density_from_pressure(pressure):
        return np.interp(pressure, pressure_mpa, dens_gml)
    return density_from_pressure


def get_pressure_from_density_wrapper(dens_gml, pressure_mpa):
    """
    Pressure of bulk argon from density for 300 K from NIST.
    rho should be in g/ml.
    Return pressure in MPa.
    """
    def pressure_from_density(density):
        return np.interp(density, dens_gml, pressure_mpa)
    return pressure_from_density


def get_gibbs_energy_from_rho_argon_wrapper(dens_gml, gibbs_evgmla3):
    """
    Gibbs free energy of bulk argon at 300 K from NIST.
    rho should be in g/ml.
    Return energy in eV/(g/ml * A^3).
    """
    def gibbs_energy_from_rho_argon(rho):
        return np.interp(rho, dens_gml, gibbs_evgmla3)
    return gibbs_energy_from_rho_argon


def gamma_gb(rho):
    """
    Return gamma equal to gamma_(graphite/argon).
    Calculated from MD.
    rho should be in g/ml.
    Return gamma in eV/A^2.
    """
    return (1.47270851e-02 / (1.0 + np.exp(-4.22865620e+01 * rho) ) - 1.47270851e-02/2)


def get_surface_from_h_coefs(h, radius):
    """
    Return surface of upper graphene sheet in A^2.
    h is Chebyshev polynomial coefficients for shape.
    radius is radius in A.
    """
    new_pol = 1 + h.deriv(m=1) ** 2
    rr, dh_dr_num = new_pol.linspace(n=1000, domain=np.array([0.0, radius]))
    
    to_int = rr * dh_dr_num ** 0.5
    
    return 2 * np.pi * integrate.trapz(to_int, rr)


def get_fluid_energy(density, volume):
    """
    Energy of fluid in eV with given density and volume. From 0.01 MPa.
    """
    stepsize = 1e-4
    num_of_points = int((density - LOW_DENSITY_EDGE) / stepsize)
    density_xx = np.linspace(LOW_DENSITY_EDGE, density, num_of_points)
    energy_density = integrate.trapz(gibbs_energy_from_rho_argon(density_xx),
                                     density_xx) - 5.206201277038113e-07
    return volume * energy_density


def float_not_in_array(x, array):
    """
    Check if x not in array with given precision.
    """
    return np.logical_not(np.any(abs(array - x) < 1e-10))


def get_volume_from_h_coefs(h, radius):
    """
    Returns volum of the bubble.
    h is Chebyshev polynomial coefficients for shape.
    radius is radius in A.
    """                    
    r = Chebyshev([radius/2, radius/2], domain=[0, radius])   
    pol_to_integrate = h * r
    return 2.0 * np.pi * (pol_to_integrate.integ()(radius) \
                          - pol_to_integrate.integ()(0.0))


def h_constraint_wrapper(radius):
    def h_constraint(grid_points):
        """
        Boundary condition h(r) = 0.
        """
        h_coef = grid_points[N_COEF_U:]
        h = Chebyshev(h_coef, domain=[0, radius])    
        return h(radius)
    return h_constraint


def h_constraint_der_zero_wrapper(radius):
    def h_constraint_der_zero(grid_points):
        """
        Boundary condition h'(0) = 0.
        """
        h_coef = grid_points[N_COEF_U:]
        h = Chebyshev(h_coef, domain=[0, radius])
        return h.deriv(m=1)(0.0)
    return h_constraint_der_zero


def u_constraint_zero_wrapper(radius):
    def u_constraint_zero(grid_points):
        """
        Boundary condition u(0) = 0.
        """
        u_coef = grid_points[:N_COEF_U]
        u = Chebyshev(u_coef, domain=[0, radius])
        return u(0.0)
    return u_constraint_zero


def get_free_energy_wrapper(radius, pressure):

    def get_free_energy_for_minimize(grid_points):
        """
        Calculates total elastic energy for given radius and pressure.
        """
        global free_energy_elastic_stretching, \
        free_energy_elastic_bending, \
        free_energy_elastic_tail, \
        free_energy_external, \
        current_volume
        
        u_coef = grid_points[:N_COEF_U]
        h_coef = grid_points[N_COEF_U:]
        
        u = Chebyshev(u_coef, domain=[0, radius])
        h = Chebyshev(h_coef, domain=[0, radius])
        r = Chebyshev([radius/2, radius/2], domain=[0, radius])
                
        dh_dr = h.deriv(m=1)
        d2h_dr2 = h.deriv(m=2)
        du_dr = u.deriv(m=1)
        
        current_volume = get_volume_from_h_coefs(h, radius)
        
        psi_str_to_integrate = \
        (du_dr ** 2 + du_dr * dh_dr ** 2 \
         + 0.25 * dh_dr ** 4 \
         + (u // r) ** 2) * r
        
        free_energy_elastic_stretching = np.pi * YOUNGS_MODULUS \
        * (psi_str_to_integrate.integ()(radius) \
           - psi_str_to_integrate.integ()(0.0))
        
        free_energy_elastic_tail = np.pi * YOUNGS_MODULUS * u(radius) ** 2
        
        psi_bend_to_integrate = \
        (d2h_dr2 ** 2 + (dh_dr // r) ** 2) * r
        
        free_energy_elastic_bending = RIGIDITY * np.pi \
        * (psi_bend_to_integrate.integ()(radius) \
           - psi_bend_to_integrate.integ()(0.0))
        
#        free_energy_elastic_bending = 0.0
        
        free_energy_external = -current_volume * pressure        

        return free_energy_elastic_stretching \
               + free_energy_elastic_bending \
               + free_energy_elastic_tail \
               + free_energy_external
        
    return get_free_energy_for_minimize


def minimize_for_pressure(radius, density, mass):
    """
    For given radius and pressure calculates solution of equation
    of elastisity of membranes. Then extract Chebyshev's polynomial 
    coefficients. Pressure is evaluated from EOS.
    
    Returns volume, u and h coefficients. pressure deveoped by membrane, 
    density of trapped ssubstance, and mass.
    """
    global count_iterations
    
    pressure = pressure_from_density(density)
    
    pressure_ev = pressure / 160217.66208
    get_free_energy_for_minimize = get_free_energy_wrapper(radius, 
                                                           pressure_ev)
    
#    print('{:>6} {:<20} {:<20} {:<20} {:<20} {:<20}'.format('Step',
#                                                            'Total',
#                                                            'Elastic str inside',
#                                                            'Elastic tail',
#                                                            'Elastic bend',
#                                                            '-PV'
#                                                            ))

    count_iterations = 1
    
    u_initial = np.zeros(N_COEF_U)
    h_initial = np.zeros(N_COEF_H)
    
    h_initial[0] = 5 / 8 * 0.15 * radius
    h_initial[1] = -0.15 * radius / 2
    h_initial[2] = -1 / 8 * 0.15 * radius
    
    h_constraint = h_constraint_wrapper(radius)
    h_constraint_der_zero = h_constraint_der_zero_wrapper(radius)
    u_constraint_zero = u_constraint_zero_wrapper(radius)
    
    cons = [{'type':'eq', 'fun': h_constraint},
            {'type':'eq', 'fun': h_constraint_der_zero},
            {'type':'eq', 'fun': u_constraint_zero}
            ]

    res_minimize = minimize(get_free_energy_for_minimize, 
                            np.append(u_initial, h_initial),
                            constraints=cons,
                            method='SLSQP',
                            options={'disp' : True,
                                    'maxiter' : MAX_ITER_IN_OPTIMIZATOR,
                                    },
                            #callback=callback_minimize_elastic_energy
                            )
    
    u_coef= res_minimize.x[:N_COEF_U]
    h_coef = res_minimize.x[N_COEF_U:]
    h = Chebyshev(h_coef, domain=[0, radius])
    
    print('\n')
    print('H: {}, R: {}, H/R: {}'.format(h(0.0), radius, h(0.0) / radius))
    print('Volume: {}, pressure: {}, density: {}'.format(current_volume, 
                                                         pressure, 
                                                         density))
    print('{:<14}: {}'.format('Mass', mass))
    print('{:<14}: {}'.format('Current mass', current_volume * density))
    print('Running time: {} seconds'.format(time.time() - start_time))
    print('\n')
    
    return current_volume, u_coef, h_coef, pressure, density, \
           current_volume * density

def find_shapes_for_radius(mass, radius):
    """
    For given radius finds pressure developed by graphene membrane and
    density of trapped substance to be in mechanical equilibrium.
    """
    first_density_attempt = 0.21
    delta_density = 0.05
    
    search_mass = []
    
    cur_density = first_density_attempt
    search_mass.append(minimize_for_pressure(radius, cur_density, mass))
    current_mass = search_mass[-1][-1]

    if search_mass[-1][-1] > mass:
        iteration_direction = -1
    
    if search_mass[-1][-1] < mass:
        iteration_direction = +1

    while (delta_density > DENSITY_PRECISION) and (abs(current_mass - mass) / mass > MASS_PRECISION):
        if search_mass[-1][-1] > mass:
            if iteration_direction == +1:
                delta_density = delta_density / 2
            iteration_direction = -1
        
        if search_mass[-1][-1] < mass:
            if iteration_direction == -1:
                delta_density = delta_density / 2
            iteration_direction = +1
        
        cur_density += iteration_direction * delta_density
        
        assert cur_density > LOW_DENSITY_EDGE, 'ERROR: Too small density.'
        assert cur_density < HIGH_DENSITY_EDGE, 'ERROR: Too big density.'
        
        search_mass.append(minimize_for_pressure(radius, cur_density, mass))
        current_mass = search_mass[-1][-1]
    
    return search_mass[-1][1], search_mass[-1][2], search_mass[-1][3], search_mass[-1][4]


def energy_for_radius_wrapper(mass):
    def energy_for_radius(radius):
        """
        Calculates total energy as sum of elastic energy, energy of 
        trapped substance and vdW energy.
        """
        start_time = time.time()
        
        return_shapes = find_shapes_for_radius(mass, radius)
        if return_shapes == -1:
            return -1
        if return_shapes == -2:
            return -2
        
        u_coef, h_coef, pressure, density = return_shapes
        
        h = Chebyshev(h_coef, domain=[0, radius])
        volume = get_volume_from_h_coefs(h, radius)

        free_energy_fluid = get_fluid_energy(density, volume)
        
        free_energy_vdw = np.pi * radius ** 2 * 0.0175 \
                          - np.pi * radius ** 2 * gamma_gb(density) \
                          - get_surface_from_h_coefs(h, radius) * gamma_gb(density)
    
        total_elastic = free_energy_elastic_stretching \
                        + free_energy_elastic_bending \
                        + free_energy_elastic_tail
    
        total_energy = total_elastic + free_energy_fluid + free_energy_vdw
        
        print('\n')
        print('{:<16}: {:<24}'.format('Elastic energy',
                                      total_elastic))
        print('{:<16}: {:<24}'.format('Fluid energy',
                                      free_energy_fluid))
        print('{:<16}: {:<24}'.format('vdW energy',
                                      free_energy_vdw))
        print('{:<16}: {:<24}'.format('Total energy',
                                      total_energy))
        print('\n')
        print('H: {}, R: {}, H/R: {}'.format(h(0.0), radius, h(0.0) / radius))
        print('Pressure: {}'.format(pressure))
        print('Density: {}'.format(density))
        print('Mass: {}'.format(mass))
        print('volume * density: {}'.format(density * current_volume))
        print('Running time: {} seconds'.format(time.time() - start_time))
        print('\n')
        
        return total_energy, \
               free_energy_vdw, \
               free_energy_elastic_stretching, \
               free_energy_elastic_bending, \
               free_energy_elastic_tail, \
               free_energy_fluid, \
               u_coef, \
               h_coef, \
               current_volume, \
               density, \
               pressure, \
               radius, \
               h(0.0), \
               h(0.0)/radius
    return energy_for_radius


def minimize_over_radius(mass):
    """
    Iterates over radius of the bubble with fixed mass of trapped substance
    to find the equilibrium bubble.
    """
    global search_grid
    
    search_grid = []
    
    first_radius_attempt = round((mass / 0.01822624) ** (1 / 3), 0)
    delta_radius = round(first_radius_attempt / 100, 0)

    for cur_radius in np.linspace(first_radius_attempt - delta_radius,
                                  first_radius_attempt + delta_radius, 3):
        search_grid.append(energy_for_radius(cur_radius))
    
    if search_grid[0][0] < search_grid[1][0] < search_grid[2][0]:
        iteration_direction = -1
        tmp = search_grid[2]
        search_grid[2] = search_grid[0]
        search_grid[0] = tmp
        cur_radius = first_radius_attempt - delta_radius
        while search_grid[-2][0] > search_grid[-1][0]:
            cur_radius += iteration_direction * delta_radius
            search_grid.append(energy_for_radius(cur_radius))
    elif search_grid[0][0] > search_grid[1][0] > search_grid[2][0]:
        iteration_direction = +1
        cur_radius = first_radius_attempt + delta_radius
        while search_grid[-2][0] > search_grid[-1][0]:
            cur_radius += iteration_direction * delta_radius
            search_grid.append(energy_for_radius(cur_radius))
    
    
    for i in range(3):
        position_of_minimum = np.argmin(np.array([el[0] for el in search_grid]))
        
        r_already_measured = np.array([el[11] for el in search_grid])
        current_r_minimum = r_already_measured[position_of_minimum]
        
        r_wanted = np.linspace(current_r_minimum - delta_radius * 9,
                               current_r_minimum + delta_radius * 9, 19)
        
        r_to_measure = np.array([x for x in r_wanted if float_not_in_array(x, r_already_measured)])
        
        for cur_radius in r_to_measure:
            search_grid.append(energy_for_radius(cur_radius))

    r_measured = np.array([el[11] for el in search_grid])
    en_measured = np.array([el[0] for el in search_grid])
    arg_to_sort = np.argsort(r_measured)
    r_measured = r_measured[arg_to_sort]
    en_measured = en_measured[arg_to_sort]
    cur_min = np.argmin(en_measured)
    coefs = np.polyfit(r_measured[cur_min-5:cur_min+6], 
                       en_measured[cur_min-5:cur_min+6], 2)
    predicted_min = -coefs[1] / (2 * coefs[0])    
    search_grid.append(energy_for_radius(predicted_min))
    
    write_path_csv(search_grid, mass, str(int(mass)) + '.csv')
    
    position_of_minimum = np.argmin(np.array([el[0] for el in search_grid]))
    write_solution_csv(search_grid[position_of_minimum], mass, 'data.csv')
    
#    for cur_radius in np.linspace(100, 2500, 100):
#        return_energy = energy_for_radius(cur_radius)
#        if return_energy == -1:
#            break
#        if return_energy == -2:
#            continue
#        search_grid.append(return_energy)
#        
#    write_path_csv(search_grid, mass, str(int(mass)) + '.csv')
#    position_of_minimum = np.argmin(np.array([el[0] for el in search_grid]))
#    write_solution_csv(search_grid[position_of_minimum], mass, 'data.csv')

    return search_grid


def write_path_csv(search_grid, mass, filename):
    """
    Writes energy curve E_m(R) for fixed mass of trapped substance.
    """
    headers = ['Total energy', 'vdW', 'Stretching', 'Bending', 'Tail',
               'Fluid', 'u coefs', 'h coefs', 'Volume', 'Density',
               'Pressure', 'Radius', 'Height', 'Ratio', 'Mass']
    
    with open (filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        
        for line in search_grid:
            u_str = ''
            for u_coef_cur in line[6]:
                u_str += ' ' + str(u_coef_cur)
            h_str = ''
            for h_coef_cur in line[7]:
                h_str += ' ' + str(h_coef_cur)
            row = {'Total energy': line[0],
                   'vdW': line[1],
                   'Stretching': line[2], 
                   'Bending': line[3],
                   'Tail': line[4],
                   'Fluid': line[5],
                   'u coefs': u_str,
                   'h coefs': h_str,
                   'Volume': line[8],
                   'Density': line[9],
                   'Pressure': line[10],
                   'Radius': line[11],
                   'Height': line[12],
                   'Ratio': line[13],
                   'Mass': mass
                   }
            writer.writerow(row)


def write_solution_csv(line, mass, filename):
    """
    Writes to a file parameters of equilibrium bubble with given mass.
    """
    headers = ['Total energy', 'vdW', 'Stretching', 'Bending', 'Tail',
               'Fluid', 'u coefs', 'h coefs', 'Volume', 'Density',
               'Pressure', 'Radius', 'Height', 'Ratio', 'Mass']
    
    file_exists = os.path.isfile(filename)
    
    with open (filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
    
        if not file_exists:
            writer.writeheader()  # file doesn't exist yet, write a header

        u_str = ''
        for u_coef_cur in line[6]:
            u_str += ' ' + str(u_coef_cur)
        h_str = ''
        for h_coef_cur in line[7]:
            h_str += ' ' + str(h_coef_cur)
        row = {'Total energy': line[0],
               'vdW': line[1],
               'Stretching': line[2], 
               'Bending': line[3],
               'Tail': line[4],
               'Fluid': line[5],
               'u coefs': u_str,
               'h coefs': h_str,
               'Volume': line[8],
               'Density': line[9],
               'Pressure': line[10],
               'Radius': line[11],
               'Height': line[12],
               'Ratio': line[13],
               'Mass': mass
               }
        writer.writerow(row)


def callback_minimize_elastic_energy(xs):
    global count_iterations
    current_free_energy = free_energy_elastic_stretching \
                          + free_energy_elastic_bending \
                          + free_energy_elastic_tail \
                          + free_energy_external
    if count_iterations % 10 == 0 or count_iterations == 1:
        print('{:>6} {:<20} {:<20} {:<20} {:<20} {:<20}'.format(count_iterations,
                                                                current_free_energy,
                                                                free_energy_elastic_stretching,
                                                                free_energy_elastic_tail,
                                                                free_energy_elastic_bending,
                                                                free_energy_external))
    count_iterations += 1

    
if __name__ == '__main__':
    global search_grid
    start_time = time.time()
    # [MASS] = g/ml * A^3
    MASS = float(sys.argv[1])
#    MASS = 500000
    
    (dens_gml, pressure_mpa, gibbs_evgmla3, helm_evgmla3) = \
    np.loadtxt('../ethane_330K_from0_01to70mpa.txt')
    density_from_pressure = \
    get_density_from_pressure_wrapper(dens_gml, pressure_mpa)
    gibbs_energy_from_rho_argon = \
    get_gibbs_energy_from_rho_argon_wrapper(dens_gml, gibbs_evgmla3)
    pressure_from_density = \
    get_pressure_from_density_wrapper(dens_gml, pressure_mpa)
    
    energy_for_radius = \
    energy_for_radius_wrapper(MASS)
    
    search_grid = minimize_over_radius(MASS)
    
    print('Total time: {} seconds'.format(time.time() - start_time))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
