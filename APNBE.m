function [idx,K,E,nIter,S]=APNBE(data,NN,lambda,m,maxNumIter,aimCl)
% data: n by p matrix, each row is an observation
% NN: n by m matrix, each row contains the indices of m nearest neighbor of the ith datapoint
% lambda: damping factor
% m: nearest neighbor size
% maxNumIter: maximal number of iterations
% aimCl: the number of clusters we aim to get

% idx: a vector of n elements, represents the exemplar assignment
% K: number of exemplars
% E: list of examplars
% nIter: number of iterations used
% S: similarity matrix


N=size(data,1);

%% construct the matrix Q
NNl=NN(:);
count=histc(NNl,1:N);
acount=zeros(1,N+1);acount(2:N+1)=cumsum(count);acount=acount+1;
[~,p]=sort(NNl);

%% construct the similarity matrix S, storage is reduced by nn
S=zeros(N,m);
for i=1:N, S(i,:)=(pdist2(data(i,:),data(NN(i,:),:)));end;S=-S.^2;
n_sample=ceil(sqrt(N*log(N)/log(2)));
n_sample=min(n_sample,N-1);
seq=randperm(N,n_sample);
D=(pdist2(data(seq,:),data(seq,:)));D=D.^2;
med=median(D(:));Skk=-med*ones(N,1);

%% initialation of iterations
A=zeros(N,m);Akk=zeros(N,1);R=A;Rkk=Akk;E=[];K=0;E0=E;
nIter=1; Ve=zeros(N,1);Se=[];T=m+aimCl+2;

%% running the first ten iterations
for nIter=1:maxNumIter 
    
    % compute max2
    Rold=R;Rkkold=Rkk; 
    NNI=[NN,(1:N)'];
    Ae=repmat(Ve(E0)',[N,1]);
    ind=ismember(repmat(E0',[N,1]),NNI);
    Ae(ind)=-Inf;
    if K==0, Ae=[];Se=[];end
    ASne=[A+S,Akk+Skk,Ae+Se];
    [Y1,I1]=max(ASne,[],2);
    tmpid=sub2ind([N,m+1+K],(1:N)',I1);
    ASne(tmpid)=-Inf; 
    [Y2,~]=max(ASne,[],2);
    
    % compute R
    R=S-repmat(Y1,[1,m]);
    Rkk=Skk-Y1;
    I1_less_than_m=(I1<=m);list=(1:N)';
    rows=list(I1_less_than_m);cols=I1(I1_less_than_m);
    tmpid=sub2ind([N,m],rows,cols);
    R(tmpid)=S(tmpid)-Y2(I1_less_than_m);
    Rkk(~I1_less_than_m)=Skk(~I1_less_than_m)-Y2(~I1_less_than_m);
    
    R=(1-lambda)*R+lambda*Rold;
    Rkk=(1-lambda)*Rkk+lambda*Rkkold;
    if nIter==1
       R=(1-lambda)\R;Rkk=(1-lambda)\Rkk;
    end
    
    % compute A
    Aold=A;Akkold=Akk;
    Rp=max(R,0);
    Akku=zeros(N,1);
    for k=1:N
        Akku(k)=sum(Rp(p(acount(k):acount(k+1)-1)));
    end
    Veold=Ve;V=Rkk+Akku;Ve=min(0,V);
    Au=V(NN);Au=Au-Rp;Au=min(Au,0);
    A=(1-lambda)*Au+lambda*Aold;
    Akk=(1-lambda)*Akku+lambda*Akkold;
    
    % compute Se
    V2=Rkk+Akk;
    E=find((Rkk+Akk)>0);
    K=length(E);
    if K>T, [~,E0]=sort(-V2);E0=E0(1:T);else E0=E;end
    Ve=(1-lambda)*Ve+lambda*Veold;
    Se=pdist2(data,data(E0,:));Se=-Se.^2; 
       
end

Se=-pdist2(data,data(E,:));
for j=1:K,Se(E(j),j)=Skk(E(j)); end
[~,cc]=max(Se,[],2);cc(E)=1:K;idx=E(cc);
 
end
