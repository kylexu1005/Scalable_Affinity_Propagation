function [idx1,idx2seq,K1]=SAP(data,NN,m,lambda,maxNumIter,aimCl)
%% stage one: APNB with Exemplars
[idx1,K1,E1,~,S]=APNBE(data,NN,lambda,m,maxNumIter,aimCl);

%% stage two: spectral clustering
N=size(data,1);data2=data(E1,:);
D=pdist2(data2,data2);D=D.^2;
cut=ceil(K1/(2*aimCl));cut=min(m,cut);
sigma=-S(E1,cut);sigma=sqrt(sigma);
occurs=histc(idx1,E1);

idx2seq=zeros(N,9);
for ker=2:10
    W=(1/(N^2))*(occurs*occurs').*exp(-(1/ker)*D./(sigma*sigma'));
    idx2Tmp=spectral_clustering(W,aimCl,3);
    idx2=zeros(N,1);
    for i=1:N
        idx2(i)=idx2Tmp(E1==idx1(i));
    end
    idx2seq(:,ker-1)=idx2;     
end
