Here are several instances of different size in JSON format. The filename format is: “input_#actors_#scenes_maxSceneDuration_maxAcrorCost.json”.

The fields in the JSON file are:
“scenesLength”: a list of all scene lengths (one per scene in order - first is scene 0)
“actorsCost”: the cost per time unit for actor (one per actor in order - first is actor 0)
“actorsToScenes”: a list of lists mapping from actors to scenes indexed by actor (so first list are the scenes in which actor 0 is participating in, etc).

