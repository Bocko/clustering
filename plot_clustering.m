clustering
uninetflows = csvread('uninetflows.csv');
[pc,score,latent,tsquare] = princomp(uninetflows);
format = { {}; {'Marker', '^', 'MarkerSize', 6}; {'Marker', 's', 'MarkerSize', 6} };
biplotG(pc,score,'Groups',clusts,'VarLabels',criteria,'Format',format);

