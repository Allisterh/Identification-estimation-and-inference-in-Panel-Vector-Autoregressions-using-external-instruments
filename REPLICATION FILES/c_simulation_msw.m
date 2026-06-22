clear; clc;
addpath('.'); addpath('matlab_functions');

load('results/sim.mat');

columnnames = {'Fiscal policy','GDP'};
horizons    = 20;
scale       = 1;
NWlags      = 0;
auxparamMC  = 5.5^.5;
MCdraws     = 1000;
burnout     = 500;
confidence  = 0.95;

MC.mu    = sim.mu';
MC.AL    = sim.AR;
MC.Sigma = sim.Sigma;
MC.norm  = 1;

MC.Gamma    = sim.trueirf(:,1);
MC.alphaaux = sim.trueirf(1,1);
B1          = MC.Gamma ./ MC.alphaaux;
sigmae1     = abs(MC.alphaaux) ./ ((MC.Gamma'*(MC.Sigma^(-1))*MC.Gamma)^.5);
Baux        = null(((MC.Sigma^(-1/2))*B1)');
MC.B        = [B1, (MC.Sigma^(1/2))*Baux];
MC.n        = size(MC.AL,1);
MC.D        = diag([sigmae1^2, ones(1,MC.n-1)], 0);

MC.WHat         = sim.WHat;
MC.sigma2Gamma1 = MC.WHat((MC.n^2*sim.p)+1, (MC.n^2*sim.p)+1);
MC.T            = sim.T;
MC.i            = sim.i;
MC.muZ          = sim.muZ;
MC.varZ         = sim.stdZ^2;
MC.alpha        = MC.alphaaux;
MC.sigmav       = (MC.varZ - ((auxparamMC*MC.alpha).^2 / MC.D(1,1)))^.5;
MC.p            = sim.p;
MC.Y0           = sim.Y0;
MC.horizons     = horizons;
MC.x            = scale;

Caux = [eye(MC.n), MARep(MC.AL, MC.p, MC.horizons)];
C    = reshape(Caux, [MC.n, MC.n, MC.horizons+1]);
Ccum = cumsum(C, 3);
B1   = MC.x * MC.B(:,1) ./ MC.B(1,1);
MC.IRFZ(:,:,1) = reshape(sum(bsxfun(@times,C,B1'),2),    [MC.n, MC.horizons+1]);
MC.IRFZ(:,:,2) = reshape(sum(bsxfun(@times,Ccum,B1'),2), [MC.n, MC.horizons+1]);

coverageMCMSW     = zeros(MC.n, MC.horizons+1, MCdraws, 2);
coverageMCdmethod = zeros(MC.n, MC.horizons+1, MCdraws, 2);

fprintf('Running %d MC draws...\n', MCdraws);

for mcdraw = 1:MCdraws

    if mod(mcdraw,100)==0, fprintf('  draw %d / %d\n', mcdraw, MCdraws); end

    for jj = 1:MC.i
        lagsDATAMC{jj}        = zeros(MC.n, MC.p, 1+MC.T+burnout);
        lagsDATAMC{jj}(:,:,1) = fliplr(MC.Y0');
        DATAMCaux{jj}         = zeros(MC.n, MC.T+burnout);
        EIVMCaux{jj}          = zeros(1, MC.T+burnout);
        residuals{jj}         = randn(1+MC.n, MC.T+burnout);

        for ix = 1:MC.T+burnout
            rformresiduals = MC.B * (MC.D^.5) * residuals{jj}(1:MC.n, ix);
            DATAMCaux{jj}(:,ix) = MC.mu + ...
                MC.AL * reshape(lagsDATAMC{jj}(:,:,ix), [MC.n*MC.p, 1]) + rformresiduals;
            lagsDATAMC{jj}(:,1,ix+1)      = DATAMCaux{jj}(:,ix);
            lagsDATAMC{jj}(:,2:MC.p,ix+1) = lagsDATAMC{jj}(:,1:MC.p-1,ix);
            EIVMCaux{jj}(1,ix) = real(MC.muZ + ...
                (auxparamMC*MC.alpha / MC.D(1,1)^.5) * residuals{jj}(1,ix) + ...
                MC.sigmav * residuals{jj}(MC.n+1, ix));
        end

        YMC{jj} = DATAMCaux{jj}(:, (burnout-MC.p)+1:end);
        YLC{jj} = DATAMCaux{jj}(:, (burnout-MC.p):end-MC.p);
        ZMC{jj} = EIVMCaux{jj}(1, (burnout-MC.p)+1:end);
    end

    MCdata.Y    = cell2mat(YMC)';
    MCdata.Z    = cell2mat(ZMC)';
    MCdata.Ylag = cell2mat(YLC)';

    depx   = MCdata.Y;
    indepx = [ones(length(depx),1), MCdata.Ylag];
    [RFormMC.mu, RFormMC.AL, RFormMC.Sigma, RFormMC.eta, RFormMC.X, RFormMC.Y] = ...
        RForm_PVAR(depx, indepx);

    RFormMC.Gamma = RFormMC.eta * MCdata.Z / size(RFormMC.eta, 2);
    RFormMC.n     = MC.n;
    RFormMC.p     = MC.p;

    [~, RFormMC.WHat, RFormMC.V] = ...
        CovAhat_Sigmahat_Gamma(MC.p, RFormMC.X, MCdata.Z, RFormMC.eta, NWlags);

    [InferenceMSWMC, ~, ~] = MSWfunction(confidence, MC.norm, MC.x, MC.horizons, RFormMC, 0);

    for i = 1:MC.n
        aux1 = MC.IRFZ(i,:,1).^2 .* InferenceMSWMC.ahat(i,:) + ...
               MC.IRFZ(i,:,1)    .* InferenceMSWMC.bhat(i,:) + ...
               InferenceMSWMC.chat(i,:);
        if i == 1, aux1(1,1) = -inf; end
        coverageMCMSW(i,:,mcdraw,1) = aux1 <= 0;

        aux1_cum = MC.IRFZ(i,:,2).^2 .* InferenceMSWMC.ahatcum(i,:) + ...
                   MC.IRFZ(i,:,2)    .* InferenceMSWMC.bhatcum(i,:) + ...
                   InferenceMSWMC.chatcum(i,:);
        if i == 1, aux1_cum(1,1) = -inf; end
        coverageMCMSW(i,:,mcdraw,2) = aux1_cum <= 0;

        coverageMCdmethod(i,:,mcdraw,1) = ...
            (MC.IRFZ(i,:,1) <= InferenceMSWMC.Dmethodubound(i,:)) .* ...
            (InferenceMSWMC.Dmethodlbound(i,:) <= MC.IRFZ(i,:,1));

        coverageMCdmethod(i,:,mcdraw,2) = ...
            (MC.IRFZ(i,:,2) <= InferenceMSWMC.Dmethoduboundcum(i,:)) .* ...
            (InferenceMSWMC.Dmethodlboundcum(i,:) <= MC.IRFZ(i,:,2));
    end

    clear MCdata RFormMC InferenceMSWMC

end

if ~exist('results','dir'), mkdir('results'); end
save('results/d_simulation_results.mat','coverageMCMSW','coverageMCdmethod','MC','MCdraws','confidence');

plots.order = 1:MC.n;

for i = 1:2

    figure;

    for iplot = 1:MC.n

        if MC.n > ceil(sqrt(MC.n)) * floor(sqrt(MC.n))
            subplot(ceil(sqrt(MC.n)), ceil(sqrt(MC.n)), plots.order(1,iplot));
        else
            subplot(ceil(sqrt(MC.n)), floor(sqrt(MC.n)), plots.order(1,iplot));
        end

        plot(0:horizons, mean(coverageMCMSW(iplot,:,:,i),3),    'o'); hold on
        plot(0:horizons, mean(coverageMCdmethod(iplot,:,:,i),3), 'rx'); hold off

        axis([0 horizons 0 1]);
        xticks(0:horizons)
        xlabel('Years after the shock');
        ylabel('MC Coverage');
        title(columnnames(iplot));

        if iplot == 1
            legend(strcat('CS^{AR} (',num2str(100*confidence),'%)'), 'CS^{plug-in}')
            legend('location','southeast')
        end

    end

    if i == 1
        saveas(gcf,'results/d_coverage_noncum.eps','epsc')
        saveas(gcf,'results/d_coverage_noncum.png','png')
    else
        saveas(gcf,'results/d_coverage_cum.eps','epsc')
        saveas(gcf,'results/d_coverage_cum.png','png')
    end

end
