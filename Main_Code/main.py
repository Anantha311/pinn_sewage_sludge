from packages import Trainer, Tester, SaveModel,ODEsolver
import numpy as np
from tensorflow.keras.optimizers import Adam


def Pipeline(config):
    optimizer_map = {"Adam": Adam}
    optimizer_class = optimizer_map[config["physics_loss_optimizer"]]
    Model_Trainer = Trainer(
        config["beta"],
        config["T0"],
        config["Tsp"]
    )
    Model_Trainer.model_physics(
        optimizer=optimizer_class,
        learning_rate=config["physics_loss_learning_rate"]
    )

    Model_Trainer.train(
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        weight_ic=config["weight_ic"],
        weight_p=config["weight_p"],
        include_data=config["include_data"],
        weight_data=config["weight_data"]
    )
    Model_Tester = Tester(Model_Trainer)
    Model_Tester.test()
    Model_Saver = SaveModel(Model_Trainer,Model_Tester,config)
    Model_Saver.save()




# Change this to change the parameters
config = {
    "physics_loss_optimizer": "Adam",   
    "physics_loss_learning_rate": 3e-4,
    "beta": 8,
    "T0": 298.15,
    "Tsp": 623.15,
    "epochs": 10000,
    "batch_size": 128,
    "weight_ic": 0.5,
    "weight_p": 0.5,
    "include_data": True,
    "weight_data": 0.5,
}


Pipeline(config)
