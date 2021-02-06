import minizinc
from pathlib import Path
import json
import os
import datetime
import time
import numpy as np
import multiprocessing as mp


INSTANCE_FOLDER = "./data/Problems"
# SAVE_TO_FOLDER = "./data/instances_mdl1"
SOLVER = minizinc.Solver.lookup("gecode")


def run_minizinc_model(params):
    mdl_name, instance_name, timeout = params

    instance_path = os.path.join(INSTANCE_FOLDER, instance_name)

    if not os.path.isfile(mdl_name):
        raise Exception("ERROR: mdl_name not found: ", mdl_name)

    if not os.path.isfile(instance_path):
        raise Exception("ERROR: instance_name not found: ", instance_name)

    print("*** Processing model {} with instance {}".format(mdl_name, instance_name))


    model = minizinc.Model([mdl_name])

    instance = minizinc.Instance(SOLVER, model)


    with open(instance_path, 'r') as f:
        input_data = json.load(f)
        scenesLength, actorsToScenes, actorsCosts = input_data['scenesLength'],input_data['actorsToScenes'], input_data['actorsCosts']


    num_scenes = len(scenesLength)
    num_actors = len(actorsCosts)
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


    # output= {
    #
    # }
    if msol:
        results = [
            msol.status.name,
            msol.solution.objective,
            msol.statistics['initTime'].total_seconds(),
            msol.statistics['solveTime'].total_seconds(),
            msol.statistics['variables']]
        if "1" in mdl_name:
            scene_order = msol['scene_order']
        else:
            scene_order = msol['order']
        success = True


    else:
        results = [msol.status.name if msol else "NOTFOUND"]
        success = False
        scene_order = None

    return success, mdl_name, instance_name, timeout, num_actors, num_scenes, scene_order, results

def get_result(outputs):
    global results, no_sln

    for output in outputs:

        success, mdl_name, instance_name, timeout, num_actors, num_scenes, scene_order, res = output

        if success:
            # results['ids'].append(mdl_i)
            results['num_actors'].append(num_actors)
            results['num_scenes'].append(num_scenes)
            results["obj"].append(res[1])
            results["init_time"].append(res[2])
            results["solve_time"].append(res[3])
            results['mdl_name'].append(mdl_name)
            results['instance_name'].append(instance_name)
            results['num_vars'].append(res[4])
            results["sln_status"].append(res[0])

            results['scene_order'].append(scene_order)

        else:
            # no_sln['ids'].append(mdl_i)
            no_sln['num_scenes'].append(num_scenes)
            no_sln['num_actors'].append(num_actors)
            no_sln['mdl_name'].append(mdl_name)
            no_sln['instance_name'].append(instance_name)
            no_sln["sln_status"].append(res[0])
    # print(results)
    # print(no_sln)

def run(mdls, timeout):
    params = []
    for mdl_i, mdl_name in enumerate(mdls):
        for instance_name in os.listdir(INSTANCE_FOLDER):
            if '.json' in instance_name:
                params.append([mdl_name, instance_name, timeout])
    ts = time.time()

    # params = params[:10]
    pool = mp.Pool(processes=mp.cpu_count()-1)

    outputs = pool.map(run_minizinc_model, params)
    get_result(outputs)

    pool.close()
    pool.join()

    print('Time in parallel:', time.time() - ts)
    print(results)


    with open("results_{}.json".format(timeout), 'w+') as out:
        json.dump(results, out)

    with open("no_results_{}.json".format(timeout), 'w+') as out:
        json.dump(no_sln, out)

if __name__ == "__main__":

    results = {"ids":[], "num_scenes":[], "num_actors":[], "obj":[], "init_time":[], "solve_time":[], "num_vars":[], "sln_status":[], "instance_name":[], "scene_order":[], "mdl_name":[]}
    no_sln = {"ids": [], "num_scenes":[], "num_actors":[], "sln_status": [], "instance_name":[], "mdl_name":[]}

    # Params
    timeout = 300
    mdls = ['talent_model1.mzn', "talent_model1-redundant.mzn", 'talent_model2.mzn']
    run(mdls, timeout)











