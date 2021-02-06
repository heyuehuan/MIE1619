% 
% Talent Scheduling Problem Model in Minizinc
% 

include "globals.mzn"; 


% (see rehearsal_smith.dta)
% num_pieces = 9;
% num_players = 5;
% duration = [2, 4, 1, 3, 3, 2, 5, 7, 6];
% rehearsal = array2d(1..num_players, 1..num_pieces, 
%     [
%      1,1,0,1,0,1,1,0,1,
%      1,1,0,1,1,1,0,1,0,
%      1,1,0,0,0,0,1,1,0,
%      1,0,0,0,1,1,0,0,1,
%      0,0,1,0,1,1,1,1,0
%   ]);


% Film 1 

% num_scenes = 20;
% num_actors = 8;
% max_cost = 40;
% duration = [2, 1, 1, 1, 1, 3, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 2, 1, 1];
% actor_cost = [10, 4, 5, 5, 5, 40, 4, 20];
% actorsToScenes = array2d(1..num_actors,1..num_scenes,
%      [
%       1,1,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,0,0,
% 	     1,1,1,0,0,0,1,1,0,1,0,0,1,1,1,0,1,0,0,1,
%       0,1,1,0,1,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,
%       0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,
%       0,1,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,1,1,1,
%       0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,
%       0,0,0,0,1,0,1,1,0,0,0,0,0,0,1,0,0,0,0,0,
%       0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0
%    ]);

% 

% Film 2

% num_scenes = 13;
% num_actors = 10;
% max_cost = 40;
% duration = [1, 1,	1, 	1, 	3, 	1	, 1, 	1, 	1, 	1, 	1, 	1, 	1];
% actor_cost = [40, 20, 20, 10, 5, 10, 5, 4, 5, 4];
% actorsToScenes = array2d(1..num_actors,1..num_scenes,
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

% Finished in 2m 2s, 
% film order: [2, 8, 5, 7, 1, 6, 13, 11, 9, 10, 12, 3, 4]
% waiting_time: [0, 0, 0, 3, 2, 1, 5, 3, 0, 0]
% total_cost: 87

int: num_scenes;
int: num_actors;
int: max_cost;
array[1..num_scenes] of int: duration;
array[1..num_actors, 1..num_scenes] of 0..1: actorsToScenes;



%
% Decision variables
%

array[1..num_scenes] of var 1..num_scenes: scene_order;
array[1..num_actors] of var 0..sum(duration): waiting_time; % waiting time for players
array[1..num_actors] of var 1..num_scenes: s_from; % first rehearsal
array[1..num_actors] of var 1..num_scenes: s_to;   % last rehearsal
array[1..num_actors] of var 1..max_cost: actor_cost;
%var int: total_cost = sum(i in 1..num_actors) (waiting_time[i]*actor_cost[i]); % objective
var 0..(max_cost*sum(duration)*num_actors): total_cost = sum(i in 1..num_actors) (waiting_time[i]*actor_cost[i]); % objective

solve :: int_search(
         scene_order % ++ waiting_time% ++ p_from ++ p_to ++ [total_waiting_time]
        , 
        first_fail, % occurrence, % max_regret, % first_fail, 
        indomain_max, % indomain_max, 
        complete) 
    minimize total_cost;

%solve minimize total_cost;

constraint
  all_different(scene_order) :: domain
  /\


  forall(a in 1..num_actors) (

     % fix the range from..to, i.e. don't count all that start with 0 
     % or ends with 0.
     % This means that we collect the rehearsals with many 0 at the ends
     %
     s_from[a] < s_to[a]
     /\
     
     % compute first scene
     forall(i in 1..num_scenes) (
        i < s_from[a] -> (actorsToScenes[a, scene_order[i]] = 0)
     )
     /\
     % compute last scene
     forall(i in 1..num_scenes) (
        i > s_to[a] -> (actorsToScenes[a, scene_order[i]] = 0)
     )
     /\ % and now: count the waiting time for from..to
     waiting_time[a] = 
         sum(i in 1..num_scenes) (
              duration[scene_order[i]] * bool2int(
                                             i >= s_from[a] /\ i <= s_to[a] 
                                             /\
                                             actorsToScenes[a,scene_order[i]] = 0
                                )
     ) 

  )

  /\ % symmetry breaking
  scene_order[1] < scene_order[num_scenes]
;

output[
  "film order: " , show(scene_order), "\n",
  "waiting_time: ", show(waiting_time), "\n",
  "total_cost: " , show(total_cost), "\n",
] ++ 
[
  if j = 1 then "\n" else " " endif ++
    show(actorsToScenes[a, scene_order[j]]) ++ " "
  | a in 1..num_actors, j in 1..num_scenes, 
] ++ 
["\n"]
;


% 
% This MiniZinc model is adapted from the rehersal scheduling minimiz model by 
% Hakan Kjellerstrand, hakank@gmail.com, http://www.hakank.org/minizinc
%
