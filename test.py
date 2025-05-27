import numpy as np
import matplotlib.pyplot as plt

def method_1(rho, u, E, h, tau, psi_p, max_iter, beta_limiter=1.0, const_beta=0.1, eps_rel=1e-4, eps_abs=1e-6):
    n = len(rho)
    rho_s = rho.copy()
    u_s = u.copy()
    E_s = E.copy()
    rho_s1 = rho.copy()
    u_s1 = u.copy()
    E_s1 = E.copy()

    s = 0
    convergence = False

    beta_rho = np.zeros(n)
    nu_rho = np.zeros(n)
    beta_u = np.zeros(n)
    nu_u = np.zeros(n)
    beta_E = np.zeros(n)
    nu_E = np.zeros(n)

    while not convergence and s < max_iter:

        rho_s[:] = rho_s1[:]
        u_s[:] = u_s1[:]
        E_s[:] = E_s1[:]

        monotonicity = False
        
        while not monotonicity:
            print(f"Iteration s = {s}")


            rho_star = psi_p * rho_s1 + (1 - psi_p) * rho_s
            kr = (np.abs(u_s) / 2) * (tau / h)
            nu_rho = beta_rho * (h ** 2) / tau
            mu_star = np.zeros(n-1)
            pi_star = np.zeros(n-1)
            E_star = np.zeros(n-1)
            for i in range(1, n-1):
                h_half = 0.5 * (h[i] + h[i-1])
                rho_half = 0.5 * (rho_star[i] + rho_star[i-1])
                u_half = 0.5 * (u_s[i] + u_s[i-1])
                mu_star[i] = rho_half * u_half - (nu_rho[i] / h_half) * (rho_star[i+1] - rho_star[i])
                if np.isnan(mu_star[i]) or np.isinf(mu_star[i]):
                    print(f"[NaN warning] mu_star at i={i} is nan or inf. rho_half={rho_half}, u_half={u_half}, nu_rho[i]={nu_rho[i]}")
                

            for i in range(1, n-1):
                h_half = 0.5 * (h[i] + h[i-1])
                rho_s1[i] = rho[i] - (tau / h_half) * (mu_star[i] - mu_star[i-1])

            if np.any(np.isnan(rho_s1)):
                print(f"[ERROR] rho_s1 contains NaN at time step T = {T_current}")
                break

            monotonicity = True
            for i in range(1, n-2):
                if (rho_s1[i+1] - rho_s1[i]) * (rho_s1[i] - rho_s1[i-1]) < 0:
                    print('ok')
                    if beta_rho[i-1] < beta_limiter and beta_rho[i] < beta_limiter:
                        beta_rho[i-1] += const_beta * kr[i-1]
                        beta_rho[i] += const_beta * kr[i]
                        print('in')
                        monotonicity = False

            # for i in range(1, n-1):
            #     h_half = 0.5 * (h[i] + h[i-1])
            #     rho_half = 0.5 * (rho_star[i] + rho_star[i-1])
            #     u_half = 0.5 * (u[i] + u[i-1])
            #     P_half = (gamma - 1)*rho_half*eps_abs
            #     pi_star[i] = P_half - (tau / h_half)*(rho[i]*u[i])/2

            # for i in range(1, n-1):
            #     h_half = 0.5 * (h[i] + h[i-1])
            #     E_half = 0.5 * (E[i] + E[i-1])
            #     u_half = 0.5 * (u[i] + u[i-1])

            nu_rho = beta_rho * (h ** 2) / tau
            # nu_u
            # nu_e
            # print(nu_rho)

        
        

        y_max = max(np.max(np.abs(rho_s1)), np.max(np.abs(u_s1)), np.max(np.abs(E_s1)))
        conv_rho = np.all(np.abs(rho_s1 - rho_s) < eps_rel * y_max + eps_abs)
        conv_u = np.all(np.abs(u_s1 - u_s) < eps_rel * y_max + eps_abs)
        conv_E = np.all(np.abs(E_s1 - E_s) < eps_rel * y_max + eps_abs)
        print(f"s = {s}, Δρ = {np.max(np.abs(rho_s1 - rho_s)):.3e}")
        if conv_rho and conv_u and conv_E:
            convergence = True
            print('OKK')

        print(rho_s1)
        s += 1

    return rho_s1, u_s1, E_s1, s


x_start, x_end = 0.0, 1.0
h_val = 0.01
x = np.arange(x_start, x_end + h_val, h_val)
n = len(x)

h = np.full(n, h_val)
tau = 0.001
T = 1.0
gamma = 1.4

rho = np.ones(n)
rho[0] = 1.740648
rho[1] = 1.740648
rho[-1] = 1.0
u = np.full(n, 0.6523404)
p = np.full(n, 1.0)
E = p / (gamma - 1) + 0.5 * rho * u**2

rho_init = rho.copy()

rho0 = rho.copy()
u0 = u.copy()
E0 = E.copy()

T_current = 0.0

max_iter_init = 10

while T_current <= T:
    print(f"\n== Time step at T = {T_current:.4f} ==")
    rho_s1, u1, E1, s = method_1(rho0, u0, E0, h, tau, psi_p=1.0, max_iter=max_iter_init)

    rho0 = rho_s1.copy()

    T_current += tau

# rho_s1, u1, E1, s = method_1(rho0, u0, E0, h, tau, psi_p=1.0, max_iter=max_iter_init)


plt.figure(figsize=(10, 5))
plt.plot(x, rho_init, label='Density at T = 0', linestyle='--', color='gray')
plt.plot(x, rho_s1, label=f'Density at T = {T:.2f}', color='blue')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Density Profile Over Time')
plt.legend()
plt.grid(True)
plt.show()
