from coom.train import Trainer

experiment_name = "experiment_0" # Ensure that the configs directory in coom contains a subfolder with the same name, and that it includes the appropriate configuration files.
sub_experiment_name = "trial" # this is in case someone wants to do multiple runs with the same configs.
trainer = Trainer(experiment_name = experiment_name, sub_experiment_name = sub_experiment_name)

trainer.train()