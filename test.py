import numpy as np
import matplotlib.pyplot as plt

def method_1(rho, u, E, h, tau, max_iter, psi_p=0.5, psi_u=0.5, beta_limiter=0.2, const_beta=0.1, eps_rel=1e-4, eps_abs=1e-6):
    n = len(rho)
    rho_s = rho.copy()
    u_s = u.copy()
    E_s = E.copy()
    rho_s1 = rho.copy()
    u_s1 = u.copy()
    E_s1 = E.copy()

    s = 0
    convergence = False

    # Init for rho
    beta_rho = np.zeros(n)
    nu_rho = np.zeros(n)

    # Init for u
    beta_u = np.zeros(n)
    nu_u = np.zeros(n)

    # Init for E
    # beta_E = np.zeros(n)
    # nu_E = np.zeros(n)

    while not convergence and s < max_iter:
        
        rho_s[:] = rho_s1[:]
        u_s[:] = u_s1[:]
        E_s[:] = E_s1[:]

        # Init beta
        beta_rho = np.zeros(n)
        beta_u = np.zeros(n)

        monotonicity = False
        
        while not monotonicity:
            print(f"Iteration s = {s}")


            rho_star = psi_p * rho_s1 + (1 - psi_p) * rho_s
            u_star = psi_u * u_s1 + (1 - psi_u) * u_s

            kr = (np.abs(u_s) / 2) * (tau / h)

            mu_star = np.zeros(n-1)
            pi_star = np.zeros(n-1)

            P = np.zeros(n-1)
            E_star = np.zeros(n-1)

            # Rho
            for i in range(1, n-1):
                if np.isnan(rho_star[i]) or np.isnan(rho_star[i+1]):
                    print(f"[NaN warning] rho_star invalid at i={i}")
                
                h_half = 0.5 * (h[i] + h[i-1])
                rho_0_cell = (rho[i] * u[i] + rho[i+1] * u[i+1]) * 0.5
                rho_implicit_cell = (rho_s[i] * u_s[i] + rho_s[i+1] * u_s[i+1]) * 0.5
                mu_star[i] = 0.5 * (rho_0_cell * u[i] + rho_implicit_cell*u_s[i]) - (nu_rho[i] / h_half) * (rho_star[i+1] - rho_star[i])
            

            mu_star[0] = rho_s1[0] * u_s1[0] - (nu_rho[0] / h[0]) * (rho_star[1] - rho_star[0])
            for i in range(2, n-1):
                h_half = 0.5 * (h[i] + h[i-1])
                rho_s1[i] = rho[i] - (tau / h_half) * (mu_star[i] - mu_star[i-1])
            
            
            # U
            for i in range(1, n-1):
                h_half = 0.5 * (h[i] + h[i-1])
                P[i] = (c0**2) * rho[i]
                pi_star[i] = P[i]/2 - (nu_u[i] / h_half)*(rho_star[i+1]*u_star[i+1] - rho_star[i]*u_star[i])/2

            pi_star[0] = P[0]/2 - (nu_u[0] / h[0])*(rho_star[1]*u_star[1] - rho_star[0]*u_star[0])/2
            for i in range(1, n-1):
                h_half = 0.5 * (h[i] + h[i-1])
                u_s1[i] = (1/rho_s1[i])*(rho[i]* u[i] - (tau / h_half)*(pi_star[i] - pi_star[i-1]) - (tau / h_half)*(mu_star[i]*u_star[i] - mu_star[i-1]*u_star[i-1]))

            

            monotonicity = True
            for i in range(1, n-2):
                nonmono_rho = (rho_s1[i+1] - rho_s1[i]) * (rho_s1[i] - rho_s1[i-1]) < 0
                nonmono_u = (u_s1[i+1] - u_s1[i]) * (u_s1[i] - u_s1[i-1]) < 0
                if nonmono_rho:
                    # print('ok')
                    if beta_rho[i-1] < beta_limiter and beta_rho[i] < beta_limiter:
                        # beta_rho[i] = beta_limiter
                        # beta_rho[i-1] = beta_limiter
                        beta_rho[i-1] += const_beta * kr[i-1]
                        beta_rho[i] += const_beta * kr[i]
                        # print('in')
                        monotonicity = False
                if nonmono_u:
                    if beta_u[i-1] < beta_limiter and beta_u[i] < beta_limiter:
                        # beta_rho[i] = beta_limiter
                        # beta_rho[i-1] = beta_limiter
                        beta_u[i-1] += const_beta * kr[i-1]
                        beta_u[i] += const_beta * kr[i]
                        # print('in')
                        monotonicity = False

            

            # for i in range(1, n-1):
            #     h_half = 0.5 * (h[i] + h[i-1])
            #     E_half = 0.5 * (E[i] + E[i-1])
            #     u_half = 0.5 * (u[i] + u[i-1])

            nu_rho = beta_rho * (h ** 2) / tau
            # print(f"beta = {beta_rho}")
            nu_u = beta_u * (h**2) / tau
            # nu_e
            # print(nu_rho)

        y_max = np.zeros(n)
        conv_rho = np.zeros(n, dtype=bool)
        conv_u = np.zeros(n, dtype=bool)
        for i in range(1, n-1):
            y_max[i] = max(np.max(np.abs(rho_s1[i])), np.max(np.abs(u_s1[i])), np.max(np.abs(E_s1[i])))
            conv_rho[i] = np.all(np.abs(rho_s1[i] - rho_s[i]) < eps_rel * y_max + eps_abs)
            conv_u[i] = np.all(np.abs(u_s1[i] - u_s[i]) < eps_rel * y_max + eps_abs)
            # conv_E = np.all(np.abs(E_s1 - E_s) < eps_rel * y_max + eps_abs)
            # if conv_rho and conv_u and conv_E:
        # if conv_rho:
        if np.all(conv_rho) and np.all(conv_u):
        # if np.all(conv_rho):
            convergence = True
            print('OKK')

        # print(rho_s1)
        print(u_s1)
        s += 1

    return rho_s1, u_s1, s


x_start, x_end = 0.0, 1.0
h_val = 0.01
x = np.arange(x_start, x_end + h_val, h_val)
n = len(x)

h = np.full(n, h_val)
tau = 0.001
T = 1.0
gamma = 1.4
c0 = 1.0

#  New condition
PSW0 = 1.0
PSW1 = 2.0
RSW0 = 1.0
RSW1 = (PSW1 / c0) ** (1.0 / gamma)

D = np.sqrt((PSW1 - PSW0) / (1.0 / RSW0 - 1.0 / RSW1)) / RSW0
W1 = D - np.sqrt((PSW1 - PSW0) / (1.0 / RSW0 - 1.0 / RSW1)) / RSW1

rho = np.ones(n)
# maybe
u = np.zeros(n)
# 
p = np.full(n, PSW0)


rho[0] = RSW1
rho[1] = RSW1
u[0] = W1
u[1] = W1
p[0] = PSW1
p[1] = PSW1

E = p / (gamma - 1) + 0.5 * rho * u**2




# old condition
# rho = np.ones(n)
# rho[0] = 1.740648
# rho[1] = 1.740648
# u = np.full(n, 0.6523404)
# p = np.full(n, 1.0)
# E = p / (gamma - 1) + 0.5 * rho * u**2
# ----------------------------

rho_init = rho.copy()
u_init = u.copy()

rho0 = rho.copy()
u0 = u.copy()
E0 = E.copy()

T_current = 0.0

max_iter_init = 10

while T_current <= 100 * tau:
    print(f"\n== Time step at T = {T_current:.4f} ==")
    rho_s1, u_s1, s = method_1(rho0, u0, E0, h, tau, max_iter=max_iter_init)

    rho0 = rho_s1.copy()
    u0 = u_s1.copy()
    

    T_current += tau

# rho_s1, u1, E1, s = method_1(rho0, u0, E0, h, tau, psi_p=1.0, max_iter=max_iter_init)


fig, (plt1, plt2) = plt.subplots(1, 2, figsize=(10, 6))


plt1.plot(x, rho_init, label='Density at T = 0', linestyle='--', color='gray')
plt1.plot(x, rho_s1, label=f'Density at T = {T:.2f}', color='blue')
plt1.set_xlabel('x')
plt1.set_ylabel('Density')
plt1.set_title('Density Profile Over Time')
plt1.legend()
plt1.grid(True)


plt2.plot(x, u_init, label='Velocity at T = 0', linestyle='--', color='gray')
plt2.plot(x, u_s1, label=f'Velocity at T = {T:.2f}', color='red')
plt2.set_xlabel('x')
plt2.set_ylabel('Velocity')
plt2.set_title('Velocity Profile Over Time')
plt2.legend()
plt2.grid(True)

plt.subplots_adjust(wspace=0.4)

plt.show()
