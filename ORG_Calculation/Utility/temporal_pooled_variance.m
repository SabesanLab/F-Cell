function [ pooled_variance, pooled_count ] = temporal_pooled_variance( framestamps, temporal_data )
% [ ref_variance, ref_times, ref_count ] = temporal_pooled_variance( cell_times, cell_reflectance, series_length )
%   This function calculates the pooled variance of a group of signals, where each time point
%   may have a different number of signals contributing to it.
%
% @params:
%    framestamps: A 1xM cell array of frame indexes. 
%
%    temporal_data: A NxM cell array of photoreceptor reflectances.
%                      Each row contains the reflectance signal from a single photoreceptor.
%
%
% @outputs:
%
%    ref_variance: An array containing the variance at each time point that we have data.
%
%    ref_count: An array as long as ref_times where each array entry contains the number of signals that
%               contributed to each time point.


end

