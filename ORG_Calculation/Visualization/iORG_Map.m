function [iORG_map] = iORG_Map(coords, temporal_profiles, varargin)




max_cells = [];

target_profiles = 32;

    % Determine the window size dynamically for each coordinate, given our
    % constraints defined above.
    neighborcoords=cell(size(coords,1),1);

    parfor c=1:size(coords,1)

        thiswindowsize=0;
        clipped_coords=[];
        while length(clipped_coords) < upper_bound
            thiswindowsize = thiswindowsize+1;
            rowborders = ([coords(c,2)-(thiswindowsize/2) coords(c,2)+(thiswindowsize/2)]);
            colborders = ([coords(c,1)-(thiswindowsize/2) coords(c,1)+(thiswindowsize/2)]);

            rowborders(rowborders<1)=1;
            colborders(colborders<1)=1;
            rowborders(rowborders>maxrowval)=maxrowval;
            colborders(colborders>maxcolval)=maxcolval;

            clipped_coords=coordclip(coords, colborders, rowborders,'i');
        end
        % Track all the neighboring coordinates that we'll be including.
        neighborcoords{c} = clipped_coords;
    end

end

