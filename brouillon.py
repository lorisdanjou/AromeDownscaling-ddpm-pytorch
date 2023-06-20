import matplotlib.pyplot as plt

mse = [
    1.08,
    1.0,
    1.14,
    1.43,
    1.91,
    1.36,
    1.5,
    0.91,
    1.24,
    1.41,
    1.49,
    1.19,
    1.4,
    1.19,
    1.26,
    1.04,
    0.97,
    1.3,
    0.88,
    1.22,
    1.31,
    0.92,
    1.12,
    1.2,
    1.35,
    1.2,
    1.05,
    0.86,
    1.18,
    0.91,
    0.92,
    0.99,
    0.93,
    0.85,
    1.36,
    1.28,
    1.92,
    1.23,
    1.16,
    1.02,
    0.95,
    1.06,
    1.42,
    1.21,
    1.29,
    1.12,
    1.0,
    1.72,
    0.89,
    1.09
]
steps = range(2000, 100001, 2000)

loss_curve = plt.figure()
plt.plot(steps, mse)
plt.title('model val mse')
plt.ylabel('mse')
plt.xlabel('iter')
plt.savefig("/cnrm/recyf/Data/users/danjoul/ddpm_experiments/expe3/" + 'MSE_curve.png', bbox_inches='tight')