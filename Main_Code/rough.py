import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 🔹 Import your classes
from packages import Trainer, ODEsolver   # change filename if needed

# =========================
# 🔹 CONFIG
# =========================
config = {
    "beta": 8,
    "T0": 298.15,
    "Tsp": 573.15
}

WEIGHTS_PATH = "/home/anantha/Projects/DOP/Pinn_HTL/Experiments/6/model.weights.h5"

# =========================
# 🔹 Load model
# =========================
trainer = Trainer(config["beta"], config["T0"], config["Tsp"])
trainer.model_physics()
trainer.neural_network_physics.load_weights(WEIGHTS_PATH)

# Load experimental data (IMPORTANT)
trainer.load_data()

# =========================
# 🔹 PINN Prediction (smooth)
# =========================
t_test = np.linspace(0, 100, 1000).reshape(-1, 1)
Y = trainer.neural_network_physics.predict(t_test, verbose=0)

Y_org_SR = Y[:, 0]
Y_BC     = Y[:, 1]
Y_org_AP = Y[:, 2]
Y_G      = Y[:, 3]
Y_ash_SR = Y[:, 4]

Y_ash_AP = trainer.Yash0 - Y_ash_SR

Y_SR = Y_org_SR + Y_ash_SR
Y_AP = Y_org_AP + Y_ash_AP

# =========================
# 🔹 ODE Solver (dashed lines)
# =========================
solver = ODEsolver(config)
ode_result = solver.run()

t_ode = ode_result["time"]

# =========================
# 🔹 Plot
# =========================
plt.figure(figsize=(8,5))

# 🔹 PINN (solid)
plt.plot(t_test, Y_SR,  color='blue',  label="PINN Ysr")
plt.plot(t_test, Y_AP,  color='red',   label="PINN Yap")
plt.plot(t_test, Y_BC,  color='black', label="PINN Ybc")
plt.plot(t_test, Y_G,   color='green', label="PINN Yg")

# 🔹 ODE (dashed)
plt.plot(t_ode, ode_result["YSR"], '--', color='blue',  alpha=0.7, label="ODE Ysr")
plt.plot(t_ode, ode_result["YAP"], '--', color='red',   alpha=0.7, label="ODE Yap")
plt.plot(t_ode, ode_result["YBC"], '--', color='black', alpha=0.7, label="ODE Ybc")
plt.plot(t_ode, ode_result["YG"],  '--', color='green', alpha=0.7, label="ODE Yg")

# 🔹 DATA (scatter)
plt.scatter(trainer.time_data, trainer.YSR_data, color='blue',  marker='o', label="Data Ysr")
plt.scatter(trainer.time_data, trainer.YAP_data, color='red',   marker='o', label="Data Yap")
plt.scatter(trainer.time_data, trainer.YBC_data, color='black', marker='o', label="Data Ybc")
plt.scatter(trainer.time_data, trainer.YG_data,  color='green', marker='o', label="Data Yg")

# =========================
# 🔹 Formatting
# =========================
plt.xlabel("Time (min)")
plt.ylabel("Yield")
plt.title(f"PINN vs ODE vs Data ({config['Tsp'] - 273.15:.0f}°C)")
plt.legend()
plt.grid()

plt.savefig("comparison_plot.png", dpi=300, bbox_inches='tight')
plt.show()









# config_2= {
#     "physics_loss_optimizer": "Adam",   
#     "physics_loss_learning_rate": 3e-4,
#     "beta": 8,
#     "T0": 298.15,
#     "Tsp": 573.15,
#     "epochs": 10000,
#     "batch_size": 128,
#     "weight_ic": 0.5,
#     "weight_p": 0.5,
#     "include_data": True,
#     "weight_data": 0.5,
# }

# # PINN Has higher weight
# config_3= {
#     "physics_loss_optimizer": "Adam",   
#     "physics_loss_learning_rate": 3e-4,
#     "beta": 8,
#     "T0": 298.15,
#     "Tsp": 623.15,
#     "epochs": 10000,
#     "batch_size": 128,
#     "weight_ic": 0.5,
#     "weight_p": 10,
#     "include_data": True,
#     "weight_data": 0.5,
# }

# config_4= {
#     "physics_loss_optimizer": "Adam",   
#     "physics_loss_learning_rate": 3e-4,
#     "beta": 8,
#     "T0": 298.15,
#     "Tsp": 573.15,
#     "epochs": 10000,
#     "batch_size": 128,
#     "weight_ic": 0.5,
#     "weight_p": 10,
#     "include_data": True,
#     "weight_data": 0.5,
# }
# Data is not included and equal weight to IC and PINN
config_5= {
    "physics_loss_optimizer": "Adam",   
    "physics_loss_learning_rate": 3e-5,
    "beta": 8,
    "T0": 298.15,
    "Tsp": 623.15,
    "epochs": 100000,
    "batch_size": 128,
    "weight_ic": 10,
    "weight_p": 10,
    "include_data": False,
    "weight_data": 0,
}

config_6= {
    "physics_loss_optimizer": "Adam",   
    "physics_loss_learning_rate": 3e-5,
    "beta": 8,
    "T0": 298.15,
    "Tsp": 573.15,
    "epochs": 100000,
    "batch_size": 128,
    "weight_ic": 10,
    "weight_p": 10,
    "include_data": False,
    "weight_data": 0,
}
# Data is not included and IC weight is more than PINN
config_7= {
    "physics_loss_optimizer": "Adam",   
    "physics_loss_learning_rate": 3e-5,
    "beta": 8,
    "T0": 298.15,
    "Tsp": 623.15,
    "epochs": 100000,
    "batch_size": 128,
    "weight_ic": 10,
    "weight_p": 1,
    "include_data": False,
    "weight_data": 0,
}

config_8= {
    "physics_loss_optimizer": "Adam",   
    "physics_loss_learning_rate": 3e-5,
    "beta": 8,
    "T0": 298.15,
    "Tsp": 573.15,
    "epochs": 100000,
    "batch_size": 128,
    "weight_ic": 10,
    "weight_p": 1,
    "include_data": False,
    "weight_data": 0,
}


# Pipeline(config_2)
# Pipeline(config_3)
# Pipeline(config_4)
Pipeline(config_5)
Pipeline(config_6)
Pipeline(config_7)
Pipeline(config_8)
