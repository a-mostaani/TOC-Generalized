[X,Y] = meshgrid(1:5,1:5);
X_new = X;
Y_new = Y;
X_new(find(X==1))=0;
Y_new(find(Y==1))=0;
X_new(find(X~=1))=1;
Y_new(find(Y~=1))=1;
Z = 0.975 * Y_new .* X_new;
Z(find(Z==0))=0.099;
surf(X,Y,Z,'FaceAlpha',0.5, 'EdgeColor','none')
hold on 
contour(Z,16)

figure
contour(Z,16)
% figure
% surf(X,Y,Z,'FaceAlpha',0.5)
