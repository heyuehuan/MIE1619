import minizinc
from pathlib import Path
import json
import os
import datetime
import numpy as np

from mlflow import log_metric, log_param, log_artifacts, start_run, end_run, create_experiment


INSTANCE_FOLDER = "./data/Problems"
# SAVE_TO_FOLDER = "./data/instances_mdl1"
SOLVER = minizinc.Solver.lookup("gecode")


def run_minizinc_model(mdl_name, instance_name, timeout = 3600):
    instance_path = os.path.join(INSTANCE_FOLDER, instance_name)

    if not os.path.isfile(mdl_name):
        raise Exception("ERROR: mdl_name not found: ", mdl_name)

    if not os.path.isfile(instance_path):
        raise Exception("ERROR: instance_name not found: ", instance_name)

    print("Processing model {} with instance {}".format(mdl_name, instance_name))


    model = minizinc.Model([mdl_name])

    instance = minizinc.Instance(SOLVER, model)


    with open(instance_path, 'r') as f:
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

    msol = instance.solve(verbose=True, timeout=datetime.timedelta(seconds=timeout))

    return msol



if __name__ == "__main__":
    results = {"ids":[], "num_scenes":[], "num_actors":[], "obj":[], "init_time":[], "solve_time":[], "num_vars":[], "sln_status":[], "instance_name":[], "scene_order":[]}
    no_sln = {"ids": [], "num_scenes":[], "num_actors":[], "sln_status": [], "instance_name":[]}
    mdls = ['talent_model1.mzn', 'talent_model2.mzn']
    timeout = 120

    for mdl_i, mdl_name in enumerate(mdls):
        for instance_name in os.listdir(INSTANCE_FOLDER):
            if '.json' in instance_name:
                msol = run_minizinc_model(mdl_name, instance_name, timeout)

                if msol:
                    results['ids'].append(mdl_i)
                    num_actors = instance_name.split("_")[1]
                    num_scenes = instance_name.split("_")[2]
                    results['num_actors'].append(num_actors)
                    results['num_scenes'].append(num_scenes)
                    results["obj"].append(msol.solution.objective)
                    results["init_time"].append(msol.statistics['initTime'].total_seconds())
                    results["solve_time"].append(msol.statistics['solveTime'].total_seconds())

                    results['instance_name'].append(instance_name)
                    results['num_vars'].append(msol.statistics['variables'])
                    results["sln_status"].append(msol.status.name)

                    if "1" in mdl_name:
                        scene_order = msol['scene_order']
                    else:
                        scene_order = msol['order']
                    results['scene_order'].append(scene_order)

                else:
                    no_sln['ids'].append(mdl_i)
                    no_sln['num_scenes'].append(num_scenes)
                    no_sln['num_actors'].append(num_actors)
                    no_sln['instance_name'].append(instance_name)
                    no_sln["sln_status"].append(msol.status.name)


    with open("results.json", 'w+') as out:
        json.dump(results, out)

    with open("no_results.json", 'w+') as out:
        json.dump(no_sln, out)










