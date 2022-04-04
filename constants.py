
# The purpose of this file is to 1) define the universal constants
# but also 2) reformulate new constants (K_1 and K_2) for our differential
# equations such that our mass and time and position and velocity components
# are small (eg mass numbers are like 1.0, position numbers like 1.3 etc)

# Define universal gravitation constant
G = 6.67408e-11  # N-m2/kg2#Reference quantities
m_nd = 1.989e+30  # kg #mass of the sun
r_nd = 5.326e+12  # m #distance between stars in Alpha Centauri
v_nd = 30000  # m/s #relative velocity of earth around the sun
t_nd = 79.91*365*24*3600*0.51  # s orbital period of Alpha Centauri
# Net constants
K_1 = G*t_nd*m_nd/(r_nd**2*v_nd)
K_2 = v_nd*t_nd/r_nd