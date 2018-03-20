%  Script for evaluating DGD for learned 3D photoacoustic
%  imaging. This script loads the data, initialises the model in k-wave and
%  evaluates the gradients. Networks are called as script in Python.
% 
%  This is accompanying code for: Hauptmann et al., Model based learning for 
%  accelerated, limited-view 3D photoacoustic tomography, 
%  https://arxiv.org/abs/1708.09832
% 
%  written by Andreas Hauptmann, January 2018
%  ==============================================================================

clear all, clc, close all

tic

% Choose phantom: Tumor, Vessel
phantomId='Vessel'; 

%Set 
testFile=['phantomData/paper' phantomId ];
compInit=true;
iterNum=0;
mkdir('saveIterates')

%Placeholder
Call_model_3DVessels


%% Eval iterates and compute new gradient

compInit=false;
for iterNum = 1:4

    %Define network
    filePath=['NetData/3Dgrad_iter' num2str(iterNum) '/3Dgrad.ckpt'];
    %Define current iterate to input into network
    dataSet  = ['phantomData/paperTest' phantomId '_Iter' num2str(iterNum) '.mat'];
    %Define result of network
    fileOutName=['saveIterates/paperTest' phantomId '_Iter' num2str(iterNum) '.h5'];
    
    % Command line to call python script and evaluation
    systemCommand = ['python EVAL_DGD_iteration.py ' filePath ' ' fileOutName ' ' dataSet ]
    [status, result] = system(systemCommand);

    %Placeholder
    Call_model_3DVessels

end

%% Eval fifth iterate

iterNum=5;
%Define network
filePath=['NetData/3Dgrad_iter' num2str(iterNum) '/3Dgrad.ckpt'];
%Define current iterate to input into network
dataSet  = ['phantomData/paperTest' phantomId '_Iter' num2str(iterNum) '.mat'];
%Define result of network
fileOutName=['saveIterates/paperTest' phantomId '_Iter' num2str(iterNum) '.h5'];

% Command line to call python script and evaluation
systemCommand = ['python EVAL_DGD_iteration.py ' filePath ' ' fileOutName ' ' dataSet ]
[status, result] = system(systemCommand);

evalTime=toc;
display(['DGD evaluation done, time: ' num2str(evalTime)])

