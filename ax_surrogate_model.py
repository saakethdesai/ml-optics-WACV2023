from ax import * 
from ax.modelbridge.registry import Models

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd

from botorch.acquisition.monte_carlo import qExpectedImprovement

#https://ax.dev/tutorials/gpei_hartmann_developer.html

torch.manual_seed(0)

latent_dim = 3 

search_space = SearchSpace(
parameters = [ RangeParameter(name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=-3.0, upper=3.0) for i in range(latent_dim) ] )

class neural_net(nn.Module):
    def __init__(self):
        super(neural_net, self).__init__()
        self.ninputs = 3
        self.noutputs = 1

        self.linear1 = nn.Linear(self.ninputs, 1000)
        self.linear2 = nn.Linear(1000, 1000)
        self.linear3 = nn.Linear(1000, self.noutputs)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        x = self.linear3(x)
        x = torch.relu(x)
        return x


class MyRunner(Runner):
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        pos_list1 = ["-5.0", "-4.5", "-4.0", "-3.5", "-3.0", "-2.5", "-2.0", "-1.5", "-1.0", "-0.5", "0.0"]
        pos_list2 = ["0.5", "1.0", "1.5", "2.0", "2.5", "3.0", "3.5", "4.0", "4.5", "5.0"]
        pos_list = pos_list1 + pos_list2
        model_list = []
        for pos in pos_list:
            model_name = "nn_" + pos + ".pth"
            model_list.append(model_name)
        for arm_name, arm in trial.arms_by_name.items():
            parameters = arm.parameters
            x = np.array([parameters.get(f"x{i}") for i in range(latent_dim)], dtype='float32')
            z = x.reshape((1, -1))
            z = torch.from_numpy(z)
            output_list = []
            for model_name in model_list:
                net = neural_net()
                net.load_state_dict(torch.load(model_name, map_location=torch.device('cpu')))
                output = net(z)
                output_np = output.detach().numpy()[0]
                output_list.append(output_np)
            #evaluate point
            obj = output_list[-6]/sum(output_list)
            trial_metadata["exp_result"] = obj
            print (parameters, obj)
        return trial_metadata


class MyMetric(Metric):
    def fetch_trial_data(self, trial): 
        records = []
        for arm_name, arm in trial.arms_by_name.items():
            params = arm.parameters
            records.append({
                "arm_name": arm_name,
                "metric_name": self.name,
                "trial_index": trial.index,
                #"mean": (params["x1"] + 2*params["x2"] - 7)**2 + (2*params["x1"] + params["x2"] - 5)**2,
                "mean": trial.run_metadata["exp_result"],
                "sem": 0.0
                })
        return Data(df=pd.DataFrame.from_records(records)) 
     


#param_names = [f"x{i}" for i in range(latent_dim)]
optimization_config = OptimizationConfig(objective = Objective(metric=MyMetric(name="mymetric"), minimize=False))
exp = Experiment(name="test", search_space=search_space, optimization_config=optimization_config, runner=MyRunner())

NUM_SOBOL_TRIALS = 1000 
NUM_BOTORCH_TRIALS = 100 

#print(f"Running Sobol initialization trials...")
sobol = Models.SOBOL(search_space=exp.search_space)

for i in range(NUM_SOBOL_TRIALS):
    print(f"Running SOBOL trial {i + 1}/{NUM_SOBOL_TRIALS}...")
    # Produce a GeneratorRun from the model, which contains proposed arm(s) and other metadata
    generator_run = sobol.gen(n=1) #?parallelization
    # Add generator run to a trial to make it part of the experiment and evaluate arm(s) in it
    trial = exp.new_trial(generator_run=generator_run)
    # Start trial run to evaluate arm(s) in the trial
    trial.run()
    # Mark trial as completed to record when a trial run is completed 
    # and enable fetching of data for metrics on the experiment 
    # (by default, trials must be completed before metrics can fetch their data,
    # unless a metric is explicitly configured otherwise)
    trial.mark_completed()

for i in range(NUM_BOTORCH_TRIALS):
    print(f"Running GP+EI optimization trial {i + 1}/{NUM_BOTORCH_TRIALS}...")
    # Reinitialize GP+EI model at each step with updated data.
    gpei = Models.BOTORCH_MODULAR(experiment=exp, data=exp.fetch_data(), botorch_acqf_class=qExpectedImprovement)
    generator_run = gpei.gen(n=1)
    trial = exp.new_trial(generator_run=generator_run)
    trial.run()
    trial.mark_completed()

print("Done!")

