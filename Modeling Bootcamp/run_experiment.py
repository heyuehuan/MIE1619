import minizinc
from pathlib import Path
import json
import os
import numpy as np

from mlflow import log_metric, log_param, log_artifacts, start_run, end_run, create_experiment


INSTANCE_FOLDER = "./data/Problems"
SAVE_TO_FOLDER = "./data/instances_mdl1"



def run_minizinc_model(mdl_name, instance_name):
    if not os.path.isfile(mdl_name):
        raise Exception("ERROR: mdl_name not found: ", mdl_name)

    if not os.path.isfile(instance_name):
        raise Exception("ERROR: instance_name not found: ", instance_name)

    print("Processing model {} with instance {}".format(mdl_name, instance_name))


    model = minizinc.Model(["talent_model2.mzn"])
    solver = minizinc.Solver.lookup("gecode")
    instance = minizinc.Instance(solver, model)


    with open(os.path.join(INSTANCE_FOLDER, instance_name), 'r') as f:
        input_data = json.load(f)
        scenesLength, actorsToScenes, actorsCosts = input_data['scenesLength'],input_data['actorsToScenes'], input_data['actorsCosts']

    if "1" in mdl_name:

        actorsToScenes_new = np.zeros((len(actorsCosts), len(scenesLength)))
        for actor, scenes in enumerate(actorsToScenes):
            actorsToScenes_new[actor, scenes] = 1


        data = {
            "num_scenes": len(scenesLength),
            "num_actors": len(actorsCosts),
            "max_cost": max(actorsCosts),
            "duration": scenesLength,
            "actor_cost": actorsCosts,
            "actorsToScenes": actorsToScenes_new.astype(int).tolist()
            }
    else:
        actorsToScenes_new = [set([i + 1 for i in scenes]) for scenes in actorsToScenes]
        data = {
            "num_scenes": len(scenesLength),
            "num_actors": len(actorsCosts),
            "max_cost": max(actorsCosts),
            "max_SceneDuration": max(scenesLength),
            "max_ActorCost":max(actorsCosts),
            "duration": scenesLength,
            "cost": actorsCosts,
            "actorsToScenes": actorsToScenes_new
            }


    for key, val in data.items():
        instance[key] = val

    msol = instance.solve(verbose=True)

    return msol




# with instance.files() as files:
#     store = Path("./tmp")
#     store.mkdir()
#     for f in files:
#         f.link_to(store / f.name)