import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import os
import os
from pathlib import Path
import matplotlib.pyplot as plt
import json
from scipy.integrate import solve_ivp
import pandas as pd


class Trainer:
    def __init__(self,beta,T0,Tsp):
        self.R = 8.314
        self.logA = np.array([3.44, 3.22, -2.35, 1.60, 1.00, 5.10])
        self.A = (10**self.logA) * 60
        self.Ea = np.array([65.31, 54.64, 15.74, 48.94, 34.36, 139.56]) * 1000
        self.neural_network_physics = None
        self.beta = beta
        self.T0 = T0
        self.Tsp = Tsp
        self.initial_conditions()


    def k_values(self, T):
        return self.A * tf.exp(- self.Ea / (self.R * T))
    
    def temperature(self, t):
        t_ramp = (self.Tsp - self.T0) / self.beta
        return tf.where(t <= t_ramp,self.T0 + self.beta * t,self.Tsp)
        

    def initial_conditions(self):
        self.ash_fraction = 0.2454
        self.Ysr0 = 0.898
        self.Yash0 = self.Ysr0 * self.ash_fraction
        self.Yorg0 = self.Ysr0 - self.Yash0
        # [Y_org_SR, Y_BC, Y_org_AP, Y_G, Y_ash_SR]
        self.Y0 = tf.constant([[self.Yorg0, 0.0, 0.102, 0.0, self.Yash0]], dtype=tf.float32) #ONCE LOOK INTO THIS LINE

    def model_physics(self, optimizer=Adam, learning_rate=3e-4):
        self.neural_network_physics = Sequential([
            Input(shape=(1,)),
            Dense(64, activation='tanh'),
            Dense(64, activation='tanh'),
            Dense(64, activation='tanh'),
            Dense(5, activation='softplus')
        ])

        self.optimizer_physics = optimizer(learning_rate=learning_rate)

    def physics_loss(self,t):
        if self.neural_network_physics == None:
            raise KeyError("CALL MODEL FUNCTION FIRST")
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            Y = self.neural_network_physics(t)
            Y_org_SR = Y[:, 0:1]
            Y_BC     = Y[:, 1:2]
            Y_org_AP = Y[:, 2:3]
            Y_G      = Y[:, 3:4]
            Y_ash_SR = Y[:, 4:5]

        dY_org_SR = tape.gradient(Y_org_SR, t)
        dY_BC     = tape.gradient(Y_BC, t)
        dY_org_AP = tape.gradient(Y_org_AP, t)
        dY_G      = tape.gradient(Y_G, t)
        dY_ash_SR = tape.gradient(Y_ash_SR, t)
        

        Temp = self.temperature(t)
        k = self.k_values(T=Temp)
        k1 = k[:, 0:1]
        k2 = k[:, 1:2]
        k3 = k[:, 2:3]
        k4 = k[:, 3:4]
        k5 = k[:, 4:5]
        k6 = k[:, 5:6]


        eq1 = dY_org_SR + (k1 + k2 + k3) * Y_org_SR - k5 * Y_org_AP
        eq2 = dY_BC     - (k1 * Y_org_SR - k4 * Y_BC)
        eq3 = dY_org_AP - (k2 * Y_org_SR + k4 * Y_BC - (k5 + k6) * Y_org_AP)
        eq4 = dY_G      - (k3 * Y_org_SR + k6 * Y_org_AP)
        t_ramp = (self.Tsp - self.T0) / self.beta
        k_ash_value = 0.0056 if abs(self.Tsp - 573.15) < 1e-6 else 0.0081
        k_ash = tf.ones_like(t) * k_ash_value
        eq5 = dY_ash_SR - tf.where(t > t_ramp,-k_ash * Y_ash_SR,0.0)
        Y_ash_AP = self.Yash0 - Y_ash_SR
        Y_total = Y_org_SR + Y_BC + Y_org_AP + Y_G + Y_ash_SR + Y_ash_AP
        eq_mass = Y_total - 1.0

        loss = tf.reduce_mean(eq1**2 + eq2**2 + eq3**2 + eq4**2 + eq5**2 + eq_mass**2)
        return 10000 * loss
        
    def ic_loss(self):
        t0 = tf.zeros((1,1))
        Y_pred = self.neural_network_physics(t0)
        return 10000 * (tf.reduce_mean((Y_pred - self.Y0)**2))
    def load_data(self):

        if self.Tsp == 573.15:
            self.time_data = tf.constant([0.00,15.5,21.8,28.1,31.3,34.3,44.3,54.3,64.4,74.5,84.5,94.6], dtype=tf.float32)[:, None]

            self.YSR_data = tf.constant([89.9868, 90.7773, 73.3202, 52.5033, 44.2688, 37.8129, 34.2556, 33.3992, 29.0514, 32.6087, 31.1594, 33.3333], dtype=tf.float32)[:, None] / 100.0

            self.YAP_data = tf.constant([10.2767, 9.4203, 21.1462, 39.1963, 41.6996, 38.6693, 36.1660, 36.2319, 35.2437, 32.4111, 32.9381, 35.9684 ], dtype=tf.float32)[:, None] / 100.0

            self.YBC_data = tf.constant([0.0000, 0.1976, 5.9947, 7.5099, 9.4862, 16.0079, 20.6851, 24.5718, 26.6798, 21.5415, 23.9130, 21.3439], dtype=tf.float32)[:, None] / 100.0

            self.YG_data = tf.constant([0.0000, 0.1976, 0.1976, 1.3834, 4.8748, 7.5758, 9.3544, 6.2582, 9.4862, 14.1634, 12.2530, 9.8814], dtype=tf.float32)[:, None] / 100.0

        elif self.Tsp == 623.15:
            self.time_data = tf.constant([0.00,15.6,21.8,28.0,31.3,40.5,50.6,60.4,70.6,80.5,90.6,100.0], dtype=tf.float32)[:, None]

            self.YSR_data = tf.constant([89.7222, 90.9722, 73.1944, 52.5000, 44.4444, 31.3889,27.9167, 25.6944, 22.6389, 24.8611, 26.6667, 26.3889], dtype=tf.float32)[:, None] / 100.0

            self.YAP_data = tf.constant([10.0000, 9.1667, 20.9722, 39.1667, 41.5278, 36.9444,29.4444, 39.3056, 48.7500, 43.1944, 39.0278, 39.1667 ], dtype=tf.float32)[:, None] / 100.0

            self.YBC_data = tf.constant([0.0000, 0.8333, 5.9722, 7.3611, 9.4444, 23.6111, 32.9167, 24.0278, 21.8056, 21.9444, 23.8889, 23.8889 ], dtype=tf.float32)[:, None] / 100.0

            self.YG_data = tf.constant([0.0000, 0.0000, 0.1389, 1.1111, 4.8611, 8.1944, 9.5833, 11.2500, 6.9444, 9.8611, 10.5556, 10.5556], dtype=tf.float32)[:, None] / 100.0
        else:
            raise KeyError("GIVE VALID TSP FOR DATA")  

    
    def data_loss(self):

        Y_pred = self.neural_network_physics(self.time_data)

        Y_org_SR = Y_pred[:, 0:1]
        Y_BC     = Y_pred[:, 1:2]
        Y_org_AP = Y_pred[:, 2:3]
        Y_G      = Y_pred[:, 3:4]
        Y_ash_SR = Y_pred[:, 4:5]

        Y_SR = Y_org_SR + Y_ash_SR
        Y_ash_AP = self.Yash0 - Y_ash_SR
        Y_AP = Y_org_AP + Y_ash_AP

        loss_sr = tf.reduce_mean((Y_SR - self.YSR_data) ** 2)
        loss_ap = tf.reduce_mean((Y_AP - self.YAP_data) ** 2)
        loss_bc = tf.reduce_mean((Y_BC - self.YBC_data) ** 2)
        loss_g  = tf.reduce_mean((Y_G - self.YG_data) ** 2)

        Y_total = Y_org_SR + Y_BC + Y_org_AP + Y_G + Y_ash_SR + Y_ash_AP
        mass_conserv_loss = tf.reduce_mean((Y_total - 1.0) ** 2)

        return 10000*(loss_sr + loss_ap + loss_bc + loss_g + mass_conserv_loss)
    
    def train(self, epochs, batch_size, include_data: bool, weight_data=10.0, weight_p=1.0, weight_ic=10.0):
        self.last_losses = {}
        self.load_data()
        if self.neural_network_physics is None:
            raise KeyError("CALL MODEL FUNCTION FIRST")
        for epoch in range(epochs):
            t_colloc = tf.random.uniform((batch_size,1), 0.0, 100.0)
            with tf.GradientTape() as tape:
                loss_p = self.physics_loss(t_colloc)
                loss_ic = self.ic_loss()
                if include_data:
                    loss_data = self.data_loss()
                    loss = weight_p * loss_p + weight_ic * loss_ic + weight_data * loss_data
                else:
                    loss = weight_p * loss_p + weight_ic * loss_ic

            grads = tape.gradient(loss, self.neural_network_physics.trainable_variables)
            self.optimizer_physics.apply_gradients(zip(grads, self.neural_network_physics.trainable_variables))

            if epoch  == epochs - 1:
                self.last_losses["total_loss"] = float(loss.numpy())
                self.last_losses["physics_loss"] = float(loss_p.numpy())
                self.last_losses["initial_value_loss"] = float(loss_ic.numpy())

                if include_data:
                    self.last_losses["data_loss"] = float(loss_data.numpy())
            if (epoch + 1) % 100 == 0:
                if include_data:
                    print(f"Epoch {epoch + 1} | "
                        f"Total: {loss.numpy():.6e} | "
                        f"Physics: {loss_p.numpy():.6e} | "
                        f"IC: {loss_ic.numpy():.6e} | "
                        f"Data: {loss_data.numpy():.6e}")
                else:
                    print(f"Epoch {epoch + 1} | "
                        f"Total: {loss.numpy():.6e} | "
                        f"Physics: {loss_p.numpy():.6e} | "
                        f"IC: {loss_ic.numpy():.6e}")


class Tester:
    def __init__(self, trainer):
        self.trainer = trainer
        self.model = trainer.neural_network_physics

    def test(self):
        self.t_test = np.linspace(0.0, 100.0, 500).reshape(-1,1)
        self.Y_pred = self.model.predict(self.t_test)

            
class ODEsolver:

    def __init__(self, config):

        self.beta = config.get("beta", 8)
        self.T0_K = config.get("T0", 298.15)
        self.Tsp_K = config.get("Tsp", 573.15)
        self.simulate_minutes = config.get("simulate_minutes", 100)
        self.t_eval = np.linspace(0, self.simulate_minutes, 5000)
        self.R = 8.314
        logA = np.array([3.44, 3.22, -2.35, 1.60, 1.00, 5.10])
        self.A = (10**logA) * 60
        self.Ea = np.array([65.31, 54.64, 15.74, 48.94, 34.36, 139.56]) * 1000
        self.ash_fraction = 0.2454
        self.Ysr0 = 0.898
        self.Yash0 = self.Ysr0 * self.ash_fraction
        self.Yorg0 = self.Ysr0 - self.Yash0
        self.Y0 = [self.Yorg0, 0.102, 0.0, 0.0, self.Yash0]


    def k_values(self, T_K):
        return self.A * np.exp(-self.Ea / (self.R * T_K))

    def temperature(self, t):
        T_ramp_end = (self.Tsp_K - self.T0_K) / self.beta
        if t <= T_ramp_end:
            return self.T0_K + self.beta * t
        else:
            return self.Tsp_K


    def odes(self, t, y):
        Yorg_sr, Yap, Ybc, Yg, Yash_sr = y
        T_K = self.temperature(t)
        k1, k2, k3, k4, k5, k6 = self.k_values(T_K)
        dYorg_sr = -(k1 + k2 + k3) * Yorg_sr + k5 * Yap
        dYbc     = k1 * Yorg_sr - k4 * Ybc
        dYap     = k2 * Yorg_sr + k4 * Ybc - (k5 + k6) * Yap
        dYg      = k3 * Yorg_sr + k6 * Yap

        t_ramp = (self.Tsp_K - self.T0_K) / self.beta
        if t > t_ramp:
            kash = 0.0056 if abs(self.Tsp_K - 573.15) < 1e-6 else 0.0081
            dYash_sr = -kash * Yash_sr
        else:
            dYash_sr = 0
        return [dYorg_sr, dYap, dYbc, dYg, dYash_sr]


    def run(self):
        sol = solve_ivp(self.odes,[0, self.t_eval[-1]], self.Y0, t_eval=self.t_eval, method='Radau',atol=1e-10,rtol=1e-8)
        Yorg_sr, Yap_org, Ybc, Yg, Yash_sr = sol.y
        Yash_ap = self.Yash0 - Yash_sr
        Ysr_total = Yorg_sr + Yash_sr
        Yap_total = Yap_org + Yash_ap
        return {"time": sol.t,"YSR": Ysr_total,"YAP": Yap_total,"YBC": Ybc,"YG": Yg}

    def to_dataframe(self, result):
        return pd.DataFrame(result)

    def save_dataset(self, result, filename="dataset.csv"):
        df = self.to_dataframe(result)
        df.to_csv(filename, index=False)
        print(f"Saved dataset to {filename}")



class SaveModel(ODEsolver):
    def __init__(self,trainer,tester,info):
        BASE_DIR = Path(__file__).resolve().parent.parent
        EXPERIMENTS_DIR = BASE_DIR / "Experiments"
        self.main_weight_directory = EXPERIMENTS_DIR
        self.trainer = trainer
        self.tester = tester
        self.info = info
        os.makedirs(self.main_weight_directory, exist_ok=True) 
        self.main_path = self.make_directory()

    def save(self):
        self.weights()
        self.info_save()
        temp_c = int(self.trainer.Tsp - 273.15)
        solver = ODEsolver(self.info)
        ode_result = solver.run()
        self.graph(f"{temp_c}C_prediction.png",ode_result =  ode_result)

    def make_directory(self):
        self.weights_dir = Path(self.main_weight_directory)
        i = 1
        while True:
            run_dir = self.weights_dir / f"{i}"
            if not run_dir.exists():
                run_dir.mkdir(parents=True)
                return run_dir   
            i += 1

    def weights(self):
        self.weights_path = self.main_path / "model.weights.h5"
        self.trainer.neural_network_physics.save_weights(self.weights_path)
    
    def info_save(self):
        self.info_path = self.main_path / "info.txt"
        with open(self.main_path / "config.json", "w") as f:
            json.dump(self.info, f, indent=4)
        with open(self.info_path, "w") as f:
            f.write("===== PHYSICS MODEL =====\n")
            f.write(f"Optimizer: {self.info['physics_loss_optimizer']}\n")
            f.write(f"Learning Rate: {self.info['physics_loss_learning_rate']}\n")
            f.write(f"Beta: {self.info['beta']}\n")
            f.write(f"T0: {self.info['T0']}\n")
            f.write(f"Tsp: {self.info['Tsp']}\n\n")

            f.write("===== TRAINING =====\n")
            f.write(f"Epochs: {self.info['epochs']}\n")
            f.write(f"Batch Size: {self.info['batch_size']}\n")
            f.write(f"Weight IC: {self.info['weight_ic']}\n")
            f.write(f"Weight Physics: {self.info['weight_p']}\n")
            f.write(f"Include Data: {self.info['include_data']}\n")
            f.write(f"Weight Data: {self.info['weight_data']}\n\n")

            f.write("===== LOSSES =====\n")
            losses = self.trainer.last_losses
            f.write(f"Final Total Loss: {losses['total_loss']:.6e}\n")
            f.write(f"Final Physics Loss: {losses['physics_loss']:.6e}\n")
            f.write(f"Final IC Loss: {losses['initial_value_loss']:.6e}\n")
            if "data_loss" in losses:
                f.write(f"Final Data Loss: {losses['data_loss']:.6e}\n")
            f.write("\n")

    def graph(self, name, ode_result=None):

        graph_path = self.main_path / name
        plt.figure(figsize=(8, 5))

        if not hasattr(self.tester, "Y_pred"):
            raise ValueError("Run tester.test() before saving graph")

        Y = self.tester.Y_pred

        Y_org_SR = Y[:, 0]
        Y_BC     = Y[:, 1]
        Y_org_AP = Y[:, 2]
        Y_G      = Y[:, 3]
        Y_ash_SR = Y[:, 4]

        Y_ash_AP = self.trainer.Yash0 - Y_ash_SR

        Y_SR = Y_org_SR + Y_ash_SR
        Y_AP = Y_org_AP + Y_ash_AP

        t = self.tester.t_test

        plt.plot(t, Y_SR,  color='blue',  label="PINN Ysr")
        plt.plot(t, Y_AP,  color='red',   label="PINN Yap")
        plt.plot(t, Y_BC,  color='black', label="PINN Ybc")
        plt.plot(t, Y_G,   color='green', label="PINN Yg")

        if ode_result is not None:
            t_ode = ode_result["time"]

            plt.plot(t_ode, ode_result["YSR"],  '--', color='blue',  alpha=0.6, label="ODE Ysr")
            plt.plot(t_ode, ode_result["YAP"],  '--', color='red',   alpha=0.6, label="ODE Yap")
            plt.plot(t_ode, ode_result["YBC"],  '--', color='black', alpha=0.6, label="ODE Ybc")
            plt.plot(t_ode, ode_result["YG"],   '--', color='green', alpha=0.6, label="ODE Yg")

        plt.scatter(self.trainer.time_data, self.trainer.YSR_data, color='blue',  marker='o', label="Data Ysr")
        plt.scatter(self.trainer.time_data, self.trainer.YAP_data, color='red',   marker='o', label="Data Yap")
        plt.scatter(self.trainer.time_data, self.trainer.YBC_data, color='black', marker='o', label="Data Ybc")
        plt.scatter(self.trainer.time_data, self.trainer.YG_data,  color='green', marker='o', label="Data Yg")

        plt.xlabel("Time (min)")
        plt.ylabel("Yield")
        plt.title(f"PINN vs ODE vs Data ({self.trainer.Tsp - 273.15:.0f}°C)")
        plt.legend()
        plt.grid()

        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        plt.close()