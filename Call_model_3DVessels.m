%  Script for calling the kWave operattor for learned 3D photoacoustic
%  imaging. This script initialises the model in k-wave and
%  evaluates the gradients.
% 
%  This is accompanying code for: Hauptmann et al., Model based learning for 
%  accelerated, limited-view 3D photoacoustic tomography, 
%  https://arxiv.org/abs/1708.09832
% 
%  written by Felix Lucka and Andreas Hauptmann, January 2018
%  ==============================================================================

%This part is called as initialisation 
if(compInit)

    % check if kWave is on the path
    if(~exist('kspaceFirstOrder3D.m', 'file'))
       error('kWave toolbox must be on the path to execute this part of the code') 
    end

    % load struct that contains setting information and subSampling mask
    load('setting.mat', 'kgrid', 'subSamplingMask')
    recSize = [kgrid.Nx, kgrid.Ny, kgrid.Nz];

    % for these settings, see kWave documentation. 
    medium               = [];
    medium.sound_speed   = 1580; % [m/s] speed of sound
    sensor               = [];      
    sensor.mask          = false([kgrid.Nx, kgrid.Ny, kgrid.Nz]);
    sensor.mask(1, :, :) = subSamplingMask;

    % for an explanation of the options, see kWaveWrapper.m
    dataCast    =  'gpuArray-single';
    smoothP0    = true;
    codeVersion = 'Matlab'; 

    inputArgs   = {'PMLSize', [8, 8, 8], 'DataCast', dataCast, 'Smooth', smoothP0,...
        'kWaveCodeVersion', codeVersion, 'PlotSim', false, 'Output', false};

    % define function handles for forward and adjoint operator
    A    = @(p0) kWaveWrapper(p0, 'forward', kgrid, medium, sensor, inputArgs{:});
    Aadj = @(f)  kWaveWrapper(f,  'adjoint', kgrid, medium, sensor, inputArgs{:});


    %Load phantom and measurement data
    eval(['load ' testFile ' sino phan ']);
    bSize=size(sino,3);
    imag=single(zeros(recSize(1),recSize(2),recSize(3),bSize));
    grad=single(zeros(recSize(1),recSize(2),recSize(3),bSize));

    %Process all samples in test data
    for iii=1:bSize

        display(['Sample, pre computing: ' num2str(iii)])
        
        %Initial backprojection    
        fNoisy=sino(:,:,iii);
        x0=Aadj(fNoisy);

        %gradient
        gradUp= Aadj(A(x0)-fNoisy);

        % Scaling has been done for learning 
        % to avoid immediate convergence to zero solution
        imag(:,:,:,iii)=10*x0;      
        grad(:,:,:,iii)=10*gradUp;

    end

    %Save data to be read by python
    savePath=['phantomData/paperTest' phantomId '_Iter' num2str(iterNum+1)]
    save(savePath,'-v7.3','grad','imag')

else
%%
    %Initialise variables
    imag=single(zeros(recSize(1),recSize(2),recSize(3),bSize));
    grad=single(zeros(recSize(1),recSize(2),recSize(3),bSize));

    %Load data saved from python
    dataName = ['saveIterates/paperTest' phantomId '_Iter' num2str(iterNum) '.h5'];
    imagCur=h5read(dataName,'/result');

    %Go through batch
    for vvv=1:bSize

        %Compute gradient
        fCur=sino(:,:,vvv);
        xx=imagCur(:,:,:,vvv);
        gradUp= Aadj(A(xx)-fCur);

        imag(:,:,:,vvv)=xx;
        grad(:,:,:,vvv)=gradUp;

    end
    
    %Save data to be read by python
    savePath=['phantomData/paperTest' phantomId '_Iter' num2str(iterNum+1)];
    save(savePath,'-v7.3','grad','imag')

    
end




