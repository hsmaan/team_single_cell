import wandb

# Lets set some default hyperparameters for our hypothetical sweep
hyperparameter_defaults = dict(
    seed=42,
    a=1,
    b=2,
    c=3,
)

# Lets initialize wandb with our project entity, config, project name, and 
# indicate reinit=True, which will allow us to restart the sweep if it fails or 
# gets preempted
run = wandb.init(
    entity="test_entity",
    config=hyperparameter_defaults,
    project="test_project",
    reinit=True,
)
wargs = wandb.config

# Now we can use the wargs object to access the hyperparameters we want to
# - these will be sampled from the sweep based on the sweep config -
# i.e. the yaml file 

# We'll imagine we have a default config for our model that we will update 
# with the hyperparameters we want to sweep over
default_config = dict()
default_config["seed"] = wargs.seed
default_config["a"] = wargs.a
default_config["b"] = wargs.b
default_config["c"] = wargs.c

# Now we can run our model using this config, based on the selected
# hyperparameters - again our model is hypothetical here, as are the 
# arbitrary train and evaluate classes 
model = model_init()
trained_model = train(model, default_config)
evaluation_results = evaluate(trained_model, default_config)

# Now we can log the results of our model run to wandb
wandb.log({"total_score": evaluation_results[1]})

