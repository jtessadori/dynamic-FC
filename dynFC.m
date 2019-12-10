classdef dynFC < handle
    % December 2019, Jacopo Tessadori
    % Previously existing code is being recoded in class form for ease of
    % maintenance
    
    properties
        dataPath;
        Y;
        p;
        n;
        CMdemean;
        V;
        L;
        eigenLL;
        allfeatures;
        all_res;
    end
    
    methods
        function this=dynFC(path)
            % Constructor for dynFC class.
            % Only argument (required) is absolut path of data file.
            this.dataPath=path;
        end
        
        function fullProcess(this)
            % Perform the whole analysis pipeline, from loading to SMV LOO
            % classification
            this.CD;
            this.computeJointEigenspace(2);
            this.reorderJointEigenspace;
            this.extractFeatures;
            this.svmClassify;
        end
        
        function CD(this)
            % Concatenation and Demeaning
            % Given dynamical FC, concatenation of all the subjects and
            % Demeaning i.e. CM{1} = NxNxT, CM{2} = NxNxT; allCM =
            % cat(3,CM{1,CM{2}];
            disp('Concatenation and Demeaning');
            
%             % Originally..
%             load(sprintf('%s\\CM_original.mat',this.dataPath),'CM');
%             
%             allCM=[];
%             for i = 1:size(CM,2)
%                 allCM = cat(3,allCM,CM{i});
%             end
%             
%             
%             % Removing ROI 75 and 76
%             allCM(75:76,:,:)=[];
%             allCM(:,75:76,:)=[];
%             
%             mm = mean(allCM,3);
%             this.CMdemean = bsxfun(@minus, allCM, mm);

            % Split data according to patient
            inData=load(this.dataPath);
            this.p=size(inData.cm_all_subj_corr{1},1);
            
            % Define labels
            y=zeros(length(inData.idHC)+length(inData.idRR),1);
            y(inData.idHC)=0;
            y(inData.idRR)=1;
            this.Y=y;
            
            % Idea is to pack everything together, remove mean and partly
            % missing channels.
            % Start by packing...
            allPdata=cat(3,inData.cm_all_subj_corr{:});
            this.n=cellfun(@(x)size(x,3),inData.cm_all_subj_corr);
            
            % ...remove entries...
            toBeRemoved=sum(diff(allPdata,[],3)==0,3)>this.p;
            toBeRemoved=sum(toBeRemoved)>this.p/2;
            allPdata(toBeRemoved,:,:)=[];
            allPdata(:,toBeRemoved,:)=[];
            
            % ... remove means
            this.CMdemean=allPdata-repmat(mean(allPdata,3),1,1,size(allPdata,3));
            this.p=size(allPdata,1);
            
            disp('done');
        end
        
        function computeJointEigenspace(this,algorithm)
            % Compute Laplacian and Joint Eigenspace
            
            disp('Computing the Laplacian and the Joint Eigenspace');
            % These Functions are in the JoinDiagonalziation Folder
            % W set of connectivity matrices [(nxn)]x N_subjs
            this.L=ld_ComputeLaplacianGraph(this.CMdemean,'Normalized');
%             this.L=ld_ComputeLaplacianGraph(this.CMdemean,'UnNormalized');
            % L Laplacian matrices [(nxn)x N_subjs]
            % U eigenspace [(nxn)x N_subjs ]
            % E eigenvalues [(nxn)x N_subjs ]
            
            if isempty(this.V)
                if ~exist('algorithm','var')
                    algorithm=2;
                end
                
                switch algorithm
                    case 1
                        % This SHOULD be an upgrade over previous joint
                        % diagonalization code. Code from: "Pierre Ablin,
                        % Jean-François Cardoso, Alexandre Gramfort. Beyond
                        % Pham’s algorithm for joint diagonalization. 2018.
                        % ffhal-01936887f"
                        % Load original python code from author
                        insert(py.sys.path,int32(0),'D:\Code\2019_12_oldPaperJT\qndiag-master\qndiag');
                        mod=py.importlib.import_module('qndiag');
                        
                        % Fix data to use in algorithm
                        C=permute(this.L,[3,1,2]);
                        C(C==0)=eps;
                        Ctest=mod.np.array(C);
                        
                        % Define algorithm options (check python code for
                        % exact meaning of each)
                        B0=mod.np.array(eye(size(this.L,1)));
                        maxIter=int32(2e4);
                        tol=1e-8;
                        lambdaMin=1e-4;
                        maxTries=int32(10);
                        returnBlist=false;
                        verbose=true;
                        
                        % Actually perform joint diagonalization
                        tempV=mod.qndiag(Ctest,B0,maxIter,tol,lambdaMin,maxTries,returnBlist,verbose);
                        tempV=cell(tempV);
                        this.V=double(tempV{1});
                    case 2
                        Q = [];
                        for i = 1:size(this.L,3)
                            Q = cat(3,Q,this.L(:,:,i));
                        end
                        thr = 10^-8;
                        %                 thr = 1e-2;
                        % We used 10^-8. Higher thr makes the algorithm faster but
                        % the joint eigenspace is less close to the original
                        % eispaces of laplacians
                        this.V=jointD(Q,thr); % V is the Joint Eigenspace between two or more laplacians
%                         % The following loads a previously saved V matrix
%                         temp=load(sprintf('%s\\V.mat',this.dataPath),'V');
%                         this.V=temp.V;
                end
            end
            disp('done');
        end
        
        function reorderJointEigenspace(this)
            % Given the joineigensapce....
            disp('Computing the Approximates Eigenvalues');
            
            LambdaTilde=zeros(size(this.V,1),size(this.L,3));
            for j = 1:size(this.L,3)
                LambdaNew= ld_reorderJointEigenspace_v2(this.V,this.L(:,:,j));
                LambdaTilde(:,j) = diag(LambdaNew);
            end
            
            % Originally...
%             ii = 0;
%             step = 205;
%             for i = 1:step:size(LambdaTilde,2)
%                 ii = ii+1;
%                 this.eigenLL{1,ii} = LambdaTilde(:,i:(i+step-1));
%             end

            this.eigenLL=cellfun(@(x)x',mat2cell(LambdaTilde',this.n),'UniformOutput',false)';
            disp('done');
        end
        
        function extractFeatures(this)
            % Extracting Features from eigenvalues Timeseries
            disp('Extracting Features');
            standardDev =[];
            meanEig=[];
            for s = 1:size(this.eigenLL,2)
                tmpS = std(this.eigenLL{s},[],2);
                tmpM = mean(this.eigenLL{s},2);
                standardDev =cat(2,standardDev,tmpS);
                meanEig = cat(2,meanEig,tmpM);
            end
            standardDev=standardDev';
            meanEig = meanEig';
            this.allfeatures = [standardDev  meanEig];
%             this.allfeaturesNorm=featureNormSVM(allfeatures);
            disp('done');
        end
        
        function svmClassify(this)
            % Perform leave one out svm classification
            disp('Performing LOO SVM')
            
%             % Originally...
%             Healthy=sum(this.Y==1);
%             norm =1;
%             [this.all_res,Accuracy,overall] = SVMLOO_vecMS(this.allfeatures,this.Y,Healthy,norm);
            pdf=histcounts(this.Y,length(unique(this.Y)),'Normalization','pdf');
            costMat=[0,1/pdf(2);1/pdf(1),0];
            svmSearch=fitcsvm(this.allfeatures,this.Y,'Standardize',true','Cost',costMat,'OptimizeHyperparameters','BoxConstraint',...
            'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
            'expected-improvement-plus','Kfold',length(this.n)));
            svmLOO=fitcsvm(this.allfeatures,this.Y,'Kfold',length(this.n),'BoxConstraint',mean(svmSearch.BoxConstraints),'Cost',costMat,'Standardize',true');
            LOOacc=1-svmLOO.kfoldLoss('Lossfun','classifError');
            close all
            fprintf('Done. LOO acc: %0.2f\n',LOOacc);
        end
        
        function extractClassifierWeights(this)
            disp('Extracting Classifier weights and bootstraping');
            c = 5;
            loo_model = this.all_res{c};
            for i = 1:size(loo_model,2)
                model = loo_model(i).m;
                w(i,:) = (model.sv_coef' * full(model.SVs));
            end
            
            MeanW= mean(w,1);
            stdW = std(w,1);
            boot = MeanW./stdW;
            zboot = zscore(boot);
            figure;plot(zboot)
            
            zneg  = -1.96;
            zpos = 1.96;
            label = 1:this.p;
            label = label';
            label = [label;label];
            zP = label(zboot>zpos);
            zN = label(zboot<zneg);
            z = sort([zN;zP]);
            z = unique(z);
            keyboard;
        end
    end
end

