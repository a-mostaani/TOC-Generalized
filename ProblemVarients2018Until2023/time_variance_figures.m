%time variance in SAIC vs ESAIC:

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Centralized %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
hold on
%% SAIC
%Numerical SAIC
errorbar([2,3,4] , log(2.5*[6.07, 4554, 186000]),log([0.8, 0.8, 0.8]),"LineWidth",3)
hold on
%Analytical SAIC
plot([2,3,4,5] , log(2.5*[4554/45, 4554, 4554*45, 4554*45*45]),"LineWidth",3)


%%ESAIC
%Numerical ESAIC
hold on
plot([2,3,4,5] , log(2.5*[6.07, 6.07, 6.07, 6.07]),"LineWidth",3)
%Analytical ESAIC
hold on
plot([2,3,4,5] , log(3.6*[6.07, 6.07, 6.07, 6.07]),"LineWidth",3)








%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% End-to-End %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SAIC
figure
hold on
%Numerical SAIC
errorbar([2,3,4] , log([59,4126,18600] + 2.5*[6.07, 4554, 186000]),log([0.8, 0.8, 0.8]),"LineWidth",3)
hold on
%Theoretical SAIC
plot([2,3,4,5] , log([ 4554/45 * (5*9*4^1*2)/(5^2*9^2) ,  4554 * (5*9*4^2*3)/(5^3*9^3), 4554*45 * (5*9*4^3*4)/(5^4*9^4),  4554*45*45 * (5*9*4^4*5)/(5^5*9^5) ] + 2.5*[4554/45, 4554, 4554*45, 4554*45*45]),"LineWidth",3)

%% ESAIC
hold on
%Numerical ESAIC
plot([2,3,4,5] , log( [ 59 , 4467, 4467*4*4/3 * 1.5, 4467*4^2*5/3 * 1.5^2 ]  + 2.5*[6.07, 6.07, 6.07, 6.07]),"LineWidth",3)
%Analytical ESAIC:
plot([2,3,4,5] , log( [ 4467*4^-1*2/3 , 4467, 4467*4*4/3, 4467*4^2*5/3 ]  + 3.2*[6.07, 6.07, 6.07, 6.07]),"LineWidth",3)