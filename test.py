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

    while not convergence:
        rho_s[:] = rho_s1[:]
        u_s[:] = u_s1[:]
        E_s[:] = E_s1[:]

        monotonicity = False
        beta_rho = np.zeros(n)
        nu_rho = np.zeros(n)
        while not monotonicity:
            print(f"Iteration s = {s}")

            rho_star = psi_p * rho_s1 + (1 - psi_p) * rho_s
            kr = (np.abs(u_s) / 2) * (tau / h)
            nu_rho = beta_rho * (h ** 2) / tau
            mu_star = np.zeros(n-1)
            for i in range(1, n-1):
                h_half = 0.5 * (h[i] + h[i-1])
                rho_half = 0.5 * (rho_star[i] + rho_star[i-1])
                u_half = 0.5 * (u_s[i] + u_s[i-1])
                mu_star[i] = rho_half * u_half - (nu_rho[i] / h_half) * (rho_star[i+1] - rho_star[i])
                rho_s1[i] = rho[i] - (tau / h_half) * (mu_star[i] - mu_star[i-1])


            monotonicity = True
            for i in range(1, n-2):
                if (rho_s1[i+1] - rho_s1[i]) * (rho_s1[i] - rho_s1[i-1]) < 0:
                    print('ok')
                    if beta_rho[i-1] < beta_limiter and beta_rho[i] < beta_limiter:
                        beta_rho[i-1] += const_beta * kr[i-1]
                        beta_rho[i] += const_beta * kr[i]
                        print('in')
                        monotonicity = False

            nu_rho = beta_rho * (h ** 2) / tau
            # print(nu_rho)

        y_max = max(np.max(np.abs(rho_s1)), np.max(np.abs(u_s1)), np.max(np.abs(E_s1)))
        conv_rho = np.all(np.abs(rho_s1 - rho_s) < eps_rel * y_max + eps_abs)
        conv_u = np.all(np.abs(u_s1 - u_s) < eps_rel * y_max + eps_abs)
        conv_E = np.all(np.abs(E_s1 - E_s) < eps_rel * y_max + eps_abs)

        # if conv_rho and conv_u and conv_E:
        if np.all(np.abs(rho_s1 - rho_s) < eps_rel * y_max + eps_abs):
            convergence = True
            print(convergence)

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
u = np.full(n, 0.6523404)
p = np.full(n, 1.0)
E = p / (gamma - 1) + 0.5 * rho * u**2

rho0 = rho.copy()
u0 = u.copy()
E0 = E.copy()

# rho_s0 = rho0.copy()
max_iter_init = 10
rho_s1, u1, E1, s = method_1(rho0, u0, E0, h, tau, psi_p=1.0, max_iter=max_iter_init)


plt.figure(figsize=(10, 5))
plt.plot(x, rho, label='Density at s = 0 (initial)', linestyle='--', color='gray')
plt.plot(x, rho_s1, label=f'Density at final s= {s}', color='blue')
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Density Profile')
plt.legend()
plt.grid(True)
plt.show()