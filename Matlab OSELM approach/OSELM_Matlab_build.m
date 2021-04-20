
%%%%%%%%%%% step 1 Initialization Phase
%Training data chunk. Images can be stored in CSV format to follow the below code.
%If using image then need to pass image paths and read those images in batch wise manner 
P0=P(1:N0,:); 
T0=T(1:N0,:);

IW = rand(nHiddenNeurons,nInputNeurons)*2-1;

Bias = rand(1,nHiddenNeurons)*2-1;
H0 = SinActFun(P0,IW,Bias);
    
M = pinv(H0' * H0);
beta = pinv(H0) * T0;
clear P0 T0 H0;

%%%%%%%%%%%%% step 2 Sequential Learning Phase
for n = N0 : Block : nTrainingData
    if (n+Block-1) > nTrainingData
        Pn = P(n:nTrainingData,:);    Tn = T(n:nTrainingData,:);
        Block = size(Pn,1);             %%%% correct the block size
        clear V;                        %%%% correct the first dimention of V 
    else
        Pn = P(n:(n+Block-1),:);    Tn = T(n:(n+Block-1),:);
    end
    
    H = SinActFun(Pn,IW,Bias);
        
    M = M - M * H' * (eye(Block) + H * M * H')^(-1) * H * M; 
    beta = beta + M * H' * (Tn - H * beta);
end
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train        
clear Pn Tn H M;


HTrain = SinActFun(P, IW, Bias);
    
Y=HTrain * beta;
clear HTrain;

%%%%%%%%%%% Performance Evaluation
 
HTest = SinActFun(TV.P, IW, Bias);
   
TY=HTest * beta;
clear HTest;
end_time_test=cputime;
TestingTime=end_time_test-start_time_test  

%%%%%%%%%% Classification rate
MissClassificationRate_Training=0;
MissClassificationRate_Testing=0;

for i = 1 : nTrainingData
    [x, label_index_expected]=max(T(i,:));
    [x, label_index_actual]=max(Y(i,:));
    if label_index_actual~=label_index_expected
        MissClassificationRate_Training=MissClassificationRate_Training+1;
    end
end
TrainingAccuracy=1-MissClassificationRate_Training/nTrainingData
for i = 1 : nTestingData
    [x, label_index_expected]=max(TV.T(i,:));
    [x, label_index_actual]=max(TY(i,:));
    if label_index_actual~=label_index_expected
        MissClassificationRate_Testing=MissClassificationRate_Testing+1;
    end
end
TestingAccuracy=1-MissClassificationRate_Testing/nTestingData  



