function tms016path
% TMS016PATH Set path to the tms016-subdirectories

[p,n,e]=fileparts(which(mfilename));
addpath(genpath(p))
