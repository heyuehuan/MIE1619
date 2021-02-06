include "globals.mzn"; 

% num_actors = 10;
% num_scenes = 8;
% max_SceneDuration = 5;
% max_ActorCost = 4;
% duration = [1, 2, 2, 1, 1, 5, 5, 3];
% actorsToScenes = [{1, 2, 3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 6, 7}, {2}, {8}, {1, 8}, {1, 2, 3, 4, 8}, {7}, {1, 3, 4, 5, 8}, {3, 4, 5, 6, 7, 8}, {1, 2, 3, 4, 5, 8}];
% cost = [4, 2, 3, 3, 2, 1, 1, 2, 4, 4];

% Film 1
% num_actors = 8;
% num_scenes = 20;
% max_SceneDuration = 3;
% max_ActorCost = 40;
% duration = [2, 1, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1];
% actorsToScenes = [{1,2,3,4,6,8,10,11},{1,2,3,7,8,10,13,14,15,17,20},{2,3,5,7,8,13,14,15},{10,11,12,13},{2,7,8,12,14,18,19,20},{14,15,16,17,18},{5,7,8,15},{6,7,8,9}];
% cost = [10,4,5,5,5,40,4,20];

% input_20_18_15_14.json
% num_actors = 20;
% num_scenes = 18;
% max_SceneDuration = 15;
% max_ActorCost = 14;
% duration = [11, 11, 7, 4, 7, 14, 4, 12, 3, 7, 6, 3, 8, 2, 5, 9, 15, 2];
% actorsToScenes = [{2,3,4,8,10,11,12,14,15,17,},{3,10,16,},{5,},{1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,},{1,2,5,6,12,14,},{6,9,11,14,15,},{2,3,5,7,9,10,13,15,16,},{1,2,3,5,7,8,9,10,11,12,13,14,15,17,18,},{1,2,3,4,5,7,8,11,12,14,15,16,},{2,4,5,6,8,9,10,11,12,13,14,15,16,17,18,},{2,3,4,7,8,14,17,},{1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,},{1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,},{1,4,8,12,17,},{1,3,9,14,16,18,},{1,2,4,5,6,7,8,9,12,13,16,18,},{1,2,3,4,5,6,7,8,9,10,11,13,15,16,17,18,},{2,3,5,10,16,17,18,},{1,4,10,13,},{1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,}];
% cost = [13, 7, 6, 13, 6, 1, 9, 11, 8, 11, 10, 5, 2, 4, 7, 13, 2, 4, 1, 12];


% Film 2 (Version 2)
% num_scenes = 13;
% num_actors = 10;
% max_ActorCost = 40;
% max_SceneDuration = 3;
% duration = [1, 1,	1, 	1, 	3, 	1	, 1, 	1, 	1, 	1, 	1, 	1, 	1];
% cost = [40, 20, 20, 10, 5, 10, 5, 4, 5, 4];
% ActorOnScene = array2d(1..num_actors,1..num_scenes,
%      [
%       0,0,1,0,0,0,0,0,1,1,1,1,0,
%       1,1,0,0,1,1,1,1,1,1,1,0,1,
%       0,1,0,0,0,0,0,1,0,0,0,0,0,
%       1,0,0,1,1,1,1,1,1,1,0,0,1,
%       0,0,0,1,0,0,0,0,0,1,0,0,0,
%       1,0,0,0,0,1,1,0,1,1,1,1,0,
%       0,1,0,0,1,0,0,0,1,1,1,0,0,
%       0,0,0,0,0,1,0,0,0,1,0,0,0,
%       0,0,0,0,0,0,0,0,0,0,1,0,1,
%       0,0,0,0,0,0,0,0,1,1,0,0,0
%    ]);
% 7m 42s
% order = [2, 8, 5, 1, 7, 6, 13, 11, 9, 10, 12, 3, 4];

% Version 1          
array[Actors] of set of Scenes: actorsToScenes; 
array[Actors, Scenes] of var 0..1: ActorOnScene;

% Version 2
% array[Actors,Scenes] of 0..1: ActorOnScene; 
% array[Actors] of var set of Scenes: actorsToScenes = [ { j | j in Scenes where ActorOnScene[i,j] == 1} | i in Actors];

% Version 3



int: num_scenes; 
int: num_actors; 
int: max_SceneDuration;
int: max_ActorCost;

set of int: Actors = 1..num_actors;
set of int: Scenes = 1..num_scenes;

array[Scenes] of int: duration;
array[Actors] of int: cost; 
array[Scenes] of var set of Actors: actors_scene = [ { j | j in Actors where ActorOnScene[j,i] == 1} | i in Scenes] ;

array[Scenes] of var Scenes: order;

array[Scenes] of var set of Actors: before;
array[Scenes] of var set of Actors: after; 
array[Scenes] of var set of Actors: during; 
%var int: total_cost;
%var 0..max_SceneDuration*max_ActorCost*num_scenes*num_actors: total_cost;
var 0..sum(duration)*max_ActorCost*num_actors: total_cost;

constraint alldifferent(order);

constraint forall(a in Actors, s in Scenes) (if s in actorsToScenes[a] then ActorOnScene[a,s]=1 else ActorOnScene[a,s]=0 endif);

constraint before[1] = {};       
constraint after[num_scenes] = {};  
constraint 	forall(t in 1..num_scenes-1)(
               before[t+1] = actors_scene[order[t]] union before[t]   /\ 
               after[t] = actors_scene[order[t+1]] union after[t+1] );
               
constraint during[1] = actors_scene[order[1]];
constraint forall (t in 2..num_scenes-1)(
               during[t] = before[t+1] intersect after[t]);
constraint during[num_scenes] = actors_scene[order[num_scenes]];

constraint total_cost = sum(i in Scenes)(
      	       	     sum(j in Actors)(
		     	   cost[j] * duration[order[i]] * bool2int(j in during[i] /\ ActorOnScene[j,order[i]]==0)
		         )
                  );   

%symmetry breaking constraint
constraint order[1] < order[num_scenes];

solve :: int_search(order, first_fail, indomain, complete)
      minimize total_cost;

output ["cost = ", show(total_cost), ";\norder = ", show(order), 
       ";\n"]; 