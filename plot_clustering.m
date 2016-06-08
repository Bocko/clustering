clustering_shanghai_20_4
uninetflows = csvread('uninetflows.csv',1,0);
% [pc,score,latent,tsquare] = princomp(uninetflows);
[pc, score] = pca(zscore(uninetflows));
format = { {}; {'Marker', '^', 'MarkerSize', 6}; {'Marker', 's', 'MarkerSize', 6}; {'Marker', '+', 'MarkerSize', 6} };
% format = { {}; {'Marker', '^', 'MarkerSize', 6}; {'Marker', 's', 'MarkerSize', 6}};
biplotG(pc,score,'Groups',clusts,'VarLabels',criteria,'Format',format);

% figure
% hold on
% for i=1:length(uninetflows)
%     if clusts(i) == 0
%         plot(uninetflows(i,1),uninetflows(i,2),'bo')
%     elseif clusts(i) == 1
%         plot(uninetflows(i,1),uninetflows(i,2),'r^')
%     elseif clusts(i) == 2
%         plot(uninetflows(i,1),uninetflows(i,2),'gs')
%     end
% end
