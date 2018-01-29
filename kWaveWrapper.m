function result = kWaveWrapper(input, mode, kgrid, medium, sensor, varargin)
%KWAVEWRAPPER is a wrapper for calling kWave to simulate the forward, adjoint
%or time reversal PAT operator
%
% DESCRIPTION:
%       kWaveWrapper is a simple wrapper for kWave.
%
% USAGE:
%       result = kWaveWrapper(input, mode, kgrid, medium, sensor)
%       result = kWaveWrapper(input, mode, kgrid, medium, sensor, ...)
%
% INPUTS:
%       input - input vector on which the PAT operator should act.
%               For 'forward' mode, an initial pressure distribution in 2D
%               or 3D. For 'adjoint' or 'time_reversal' a pressure-time
%               matrix
%       mode  - type of PAT operators use, 'forward', 'adjoint' or
%               'time_reversal'
%       kgrid  - see k-Wave documentation
%       medium - see k-Wave documentation
%       sensor - see k-Wave documentation
%
% OPTIONAL INPUTS:
%       Optional 'string', value pairs that may be used to modify the
%       default computational settings. Most will get forwarded to k-Wave
%       routines, see kspaceFirstOrder2D, kspaceFirstOrder3D for avaiable
%       opitions. However, these ones modify the behavior in a different
%       way:
%       'Output' - Boolean controlling whether kWave output should be
%                  displayed (default = false)
%       'Smooth' - Boolean controlling whether smoothing should be applied
%                  (will be done in this function, not in kWave)
%       'kWaveCodeVersion' - 'Matlab', 'C++' or 'CUDA' determining which
%                  kWave code should be executed
%       'SaveToDisk' - if part of the options, the 'SaveToDisk' utility
%                      in kWave will be activated and the result returned
%                      will be a cell containing the full file path of .h5
%                      input and output files (although the output file
%                      does not exist yet. 
%                      
%
% OUTPUTS:
%       result - result of the computation, for mode 'forward' a pressure-time
%                matrix, for 'adjoint' or 'time_reversal' a 2D or 3D
%                pressure distribution
%
% ABOUT:
%       author          - Felix Lucka
%       date            - 3rd March 2017
%       last update     - 16th March 2017
%
% See also kspaceFirstOrder2D, kspaceFirstOrder3D


% set external defaul parameter
output            = false;
code_version      = 'Matlab';
add_smoothing     = false;
save_to_disk_only = false;

% set kWave default parameter
parameter = [];
parameter.DataCast  = 'single';
parameter.PlotSim   = false;
parameter.PlotPML   = false;
parameter.PMLSize   = 10 + (kgrid.dim == 3) * 10;
parameter.PMLInside = false;
parameter.Smooth    = false;
parameter.PMLAlpha  = 2;

% read additional arguments or overwrite default ones (no sanity check!)
if(~isempty(varargin))
    for input_index = 1:2:length(varargin)
        switch varargin{input_index}
            case 'Output'
                output = varargin{input_index + 1};
            case 'kWaveCodeVersion'
                code_version = varargin{input_index + 1};
            case 'Smooth'
                add_smoothing = varargin{input_index + 1};
            case 'SaveToDisk'
                save_to_disk_only = true;
                file_name = varargin{input_index + 1};
            otherwise
                % add to parameter struct (or overwrite fields)
                parameter.(varargin{input_index}) = varargin{input_index + 1};
        end
    end
end


% convert parameter struct to input cell
input_args = {};
optional_inputs = fieldnames(parameter);
for i = 1:length(optional_inputs)
    input_args{end + 1} = optional_inputs{i};
    input_args{end + 1} = parameter.(optional_inputs{i});
end
input_args_binaries = input_args;

% create a random name for input and output files
if(save_to_disk_only)
    code_version = 'Matlab';
    input_args{end+1} = 'SaveToDisk';
    input_args{end+1} = [file_name '_input.h5'];
end

switch code_version
    case {'C++','CUDA'}
        curr_rand_state = rng;
        rng('shuffle')
        file_number = randi(intmax('int32'),1,1);
        rng(curr_rand_state);
        data_name = ['kWaveID' int2str(file_number)];
        input_args_binaries{end+1} = 'DataName';
        input_args_binaries{end+1} = data_name;
end

% create source struct
source = [];

% modify inputs for different modes
switch mode
    case 'forward'
        
        % do smoothing explicitly
        if(add_smoothing)
            source.p0  = smooth(kgrid,input,false,'Blackman');
        else
            source.p0  = input;
        end
        
    case 'adjoint'
        
        % special embeding of the adjoint source (see Arridge, Betcke, Cox,
        % Lucka, Treeby, 2016, On the Adjoint Operator in Photoacoustic
        % Tomography, Inverse Problems 32(11)
        source.p_mask = sensor.mask;
        source.p = [input(:, end:-1:1), zeros(size(input, 1), 1)] + ...
            [zeros(size(input, 1), 1), input(:, end:-1:1)];
        source.p(:, end-1) = source.p(:, end-1) + source.p(:, end);
        source.p = source.p(:, 1:end-1);
        if(isscalar(medium.sound_speed))
            source.p = source.p .* (medium.sound_speed * ...
                kgrid.dx / (4 * kgrid.dt));
        else
            source.p = bsxfun(@times, source.p, medium.sound_speed( ...
                source.p_mask)) .* kgrid.dx / (4 * kgrid.dt);
        end
        
        % rescale from pressure to density
        if(isfield(medium, 'density'))
            if(isscalar(medium.density))
                source.p = medium.density .* source.p;
            else
                source.p = bsxfun(@times, source.p, ...
                    medium.density(source.p_mask));
            end
        else
            medium.density = 1;
        end
        source.p_mode = 'additive';
        
        % we are interested in the pressure at the last time point
        sensor = [];
        sensor.mask = ones(kgrid.Nx, kgrid.Ny, kgrid.Nz);
        sensor.record = {'p_final'};
        
    case 'time_reversal'
        
        % reset the initial pressure
        source.p0 = 0;
        
        % assign the time reversal data
        sensor.time_reversal_boundary_data = input;
        
    otherwise
        error(['Unknown mode: ' mode])
end


% call kWave code for the forward computation
switch code_version
    case 'Matlab'
        if(output)
            switch kgrid.dim
                case 2
                    result = kspaceFirstOrder2D(kgrid, medium, ...
                        source, sensor, input_args{:});
                case 3
                    result = kspaceFirstOrder3D(kgrid, medium, ...
                        source, sensor, input_args{:});
            end
        else
            switch kgrid.dim
                case 2
                    evalc(['result = kspaceFirstOrder2D(kgrid, '...
                        'medium, source, sensor, input_args{:});']);
                case 3
                    evalc(['result = kspaceFirstOrder3D(kgrid, '...
                        'medium, source, sensor, input_args{:});']);
            end
        end
    case 'C++'
        if(output)
            result = kspaceFirstOrder3DC(kgrid, medium, source, ...
                sensor, input_args_binaries{:});
        else
            evalc(['result = kspaceFirstOrder3DC(kgrid, medium,'...
                ' source, sensor, input_args_binaries{:});']);
        end
    case 'CUDA'
        if(output)
            result = kspaceFirstOrder3DG(kgrid, medium, source, ...
                sensor, input_args_binaries{:});
        else
            evalc(['result = kspaceFirstOrder3DG(kgrid, medium,'...
                'source, sensor, input_args_binaries{:});']);
        end
end



% modify result for different modes
if(save_to_disk_only)
    % return path to input file name
    result{1} = [file_name '_input.h5'];
    result{2} = [file_name '_output.h5'];
else
    switch mode
        case 'forward'
            result = gather(result);
        case 'adjoint'
            result = gather(result.p_final) ./ (medium.density .* medium.sound_speed.^2);
            if(add_smoothing)
                result = smooth(kgrid, result, false, 'Blackman');
            end
        case 'time_reversal'
            result = gather(result);
            if(add_smoothing)
                result = smooth(kgrid, result, false, 'Blackman');
            end
    end
end


