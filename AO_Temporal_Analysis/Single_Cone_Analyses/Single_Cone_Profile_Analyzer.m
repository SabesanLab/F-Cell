% 2018-08-29 Robert F Cooper
%
% This script analyzes the output from Rel_FF_Single_Cone_Analyses.
%


%% Load our data, and calculate some basic stats.
clear;
close all;

saveplots = true;
logmode = true;
DENSTOMETRY_THRESHOLD = 0.15; %0.03 for 11002, 0.15 for 11049
RESPONSE_THRESHOLD = 0.3;

profile_dir = uigetdir(pwd);

single_cone_mat_files = read_folder_contents(profile_dir,'mat');

load(fullfile(profile_dir,single_cone_mat_files{1}),'allcoords');

stimAmp = nan(size(allcoords,1), length(single_cone_mat_files));
stimMedian = nan(size(allcoords,1), length(single_cone_mat_files));
stimTTP = nan(size(allcoords,1), length(single_cone_mat_files));
stimRespRange = nan(size(allcoords,1), length(single_cone_mat_files));
Prestim = nan(size(allcoords,1), length(single_cone_mat_files));

controlAmp = nan(size(allcoords,1), length(single_cone_mat_files));
controlMedian = nan(size(allcoords,1), length(single_cone_mat_files));

single_cone_response = nan(size(allcoords,1), length(single_cone_mat_files));
single_cone_control_response = nan(size(allcoords,1), length(single_cone_mat_files));
trial_validity = false(size(allcoords,1), length(single_cone_mat_files));

% Determine the order we should load the files in (low to high)
for i=1:length(single_cone_mat_files)
    for s=split(single_cone_mat_files{i},'_')'        
       if contains(s,'nW')
          stimPower(i) = str2double(strrep(s,'nW',''));
       end
    end
end

[stimPower, loadinds]=sort(stimPower);

for i=1:length(single_cone_mat_files)

    single_cone_mat_files{loadinds(i)}
    load(fullfile(profile_dir,single_cone_mat_files{loadinds(i)}));
    
    stimAmp(:,i) = AmpResp;
    stimMedian(:,i) = MedianResp;
    stimTTP(:,i) = TTPResp;
    
    %     stimRespRange(:,i) = stim_resp_range;
%     Prestim(:,i) = median(stim_prestim_means,2,'omitnan');
    
%     controlAmp(:,i) = ControlAmpResp;
%     controlMedian(:,i) = ControlMedianResp;    
    
    single_cone_response(:,i) = AmpResp;%+abs(MedianResp);
%     single_cone_control_response(:,i) = ControlAmpResp;%;+abs(ControlMedianResp);
    trial_validity(:,i) = valid;
    full_signal_response{i}= stim_cell_var;
    
    
    if logmode
        % Clamp our less than 0 values to 0.
    single_cone_response(single_cone_response<0) = 0;
        single_cone_response(:,i) = log10(single_cone_response(:,i)+1);
%         single_cone_control_response(:,i) = log10(single_cone_control_response(:,i)+1);
    end
end

valid = all(~isnan(single_cone_response),2) & all(trial_validity,2);


return;

%% Individual Spatal maps
close all;
thresholds = [0 0.8];

generate_spatial_map(single_cone_response, allcoords, valid, thresholds, single_cone_mat_files(loadinds),'_stimulus', saveplots);

% generate_spatial_map(single_cone_control_response, allcoords, valid, thresholds, single_cone_mat_files,'_control', saveplots);

%% Slope between each power level.
close all;
logpow= log10([stimPower]')
% allresps = [single_cone_control_response(:,1) single_cone_response];
allresps = [single_cone_response(valid,:)];


figure;
plot(repmat( logpow,[1 length(allresps)]), allresps');
xlabel('Log10 stimulus irradiance');
ylabel('Cone response');
title(['Cone responses from ' num2str(min(stimPower)) ' to ' num2str(max(stimPower)) 'nW.' ]);
if saveplots
     set(gcf, 'renderer', 'painters');
    saveas(gcf, 'dose_response_curve.svg'); 
end

regressions = zeros(1,length(allresps));
xlin = [ logpow];

for i=1:length(allresps)
   
    regressions(:,i) = xlin\allresps(i,:)';
    
end
regressions=regressions';
figure;
reghist = histogram(regressions,'BinWidth',log10(1.025),'Normalization','pdf');
axis([0 0.45 0 10]);
title('Slope histogram of dose response curve (intercept 0)');
if saveplots
    saveas(gcf, 'dose_response_slope_histo.svg'); 
end

mainpeakind = mode(discretize(regressions, reghist.BinEdges));
mainpeakval = reghist.Values(mainpeakind);
mainpeak = reghist.BinEdges(mainpeakind);

% Fit a gaussian mixture model to determine which is which.
sigmar = std(regressions)
sigmar =repmat(sigmar,[1 1 2]);
sigmar(1,1,1) = 0.0025;
startstruct = struct('mu',[min(reghist.BinEdges); mainpeak],'Sigma', sigmar,'ComponentProportion',[.15; .85]);

gausfits=fitgmdist(regressions,2,'Start', startstruct)


% range = min(regressions):.001:max(regressions);
range = 0:.001:.45;
leftmix=pdf('Normal',range',gausfits.mu(1),sqrt(gausfits.Sigma(1)));
rightmix=pdf('Normal',range',gausfits.mu(2),sqrt(gausfits.Sigma(2)));
mewomix=pdf(gausfits,range');

normleftmix = gausfits.ComponentProportion(1).*leftmix;
normrightmix = gausfits.ComponentProportion(2).*rightmix;
normmewomix = normleftmix+normrightmix;


RESPONSE_THRESHOLD = max(range(normleftmix>normrightmix))

hold on;
plot(range,normleftmix); 
plot(range,normrightmix);
plot(range,normmewomix);
if saveplots
    saveas(gcf, 'dose_response_slope_histo_wfit.svg'); 
end


%% Slope with densitometry overlay
close all;
allinds= 1:length(densitometry_fit_amplitude);
allvalidensresps = [densitometry_fit_amplitude(valid,:)];
lessdensitometry_threshold = (allvalidensresps<=DENSTOMETRY_THRESHOLD);
lessregression_threshold = regressions<=RESPONSE_THRESHOLD;
validinds= allinds(valid);

figure; hold on;
histogram(regressions,'BinWidth',log10(1.025));
histogram(regressions(lessdensitometry_threshold),'BinWidth',log10(1.025));
hold off;


densitometry_vect_ref_valid=densitometry_vect_ref(valid);
densitometry_vect_times_valid=densitometry_vect_times(valid);

densitometry_vect_ref_valid= densitometry_vect_ref_valid((lessdensitometry_threshold & regressions>=RESPONSE_THRESHOLD));
densitometry_vect_times_valid=densitometry_vect_times_valid((lessdensitometry_threshold & regressions>=RESPONSE_THRESHOLD));

highintrinsic_lowdensitometry_inds = validinds(lessdensitometry_threshold & regressions>=RESPONSE_THRESHOLD)

% for i=1:length(densitometry_vect_ref_valid)
%     figure; subplot(2,1,1);
%     plot(densitometry_vect_times_valid{i}, densitometry_vect_ref_valid{i},'.');
%     subplot(2,1,2);
%     plot( full_signal_response{4}(highintrinsic_lowdensitometry_inds(i),:) );
% end


% Dice calculation
unionofboth = sum(lessdensitometry_threshold&(regressions<=RESPONSE_THRESHOLD))
goldstd = sum(lessdensitometry_threshold);
ourmetric = sum(regressions<=RESPONSE_THRESHOLD)

dice_wslope = (2*unionofboth)/(goldstd+ourmetric)

% axis([0 0.45 0 70]);
xlabel('Slope');
title('Slope histogram of dose response curve (intercept 0), labelled with densitometry.');
% axis([-1 .5 -.5 1]);

if saveplots
    saveas(gcf, 'dose_response_slope_histo_wdens.svg'); 
end
hold off;

figure; hold on;

histogram(allvalidensresps,'BinWidth',.015);
histogram(allvalidensresps(lessregression_threshold),'BinWidth',.015);
hold off;

figure;
valid4th= full_signal_response{4}(valid,:);
% plot((valid4th(lessregression_threshold & allvalidensresps>DENSTOMETRY_THRESHOLD,:)-mean(control_cell_var,'omitnan'))')


dens_s_cone = lessdensitometry_threshold;
intrinsic_s_cone = (regressions<=RESPONSE_THRESHOLD);


agree_s_cones = sum(dens_s_cone & intrinsic_s_cone);

agree_lm_cones = sum(~dens_s_cone & ~intrinsic_s_cone);

intrinsic_s_dens_lm = sum( ~dens_s_cone & intrinsic_s_cone )
intrinsic_lm_dens_s = sum( dens_s_cone & ~intrinsic_s_cone )

individual_agreement = (agree_s_cones +agree_lm_cones)./size(dens_s_cone,1)

%     S_L/M_
%   S|
% L/M|

% Where top is denisometry, left is intrinsic:
individual_truth = [ agree_s_cones  intrinsic_s_dens_lm;
                    intrinsic_lm_dens_s agree_lm_cones]

%% Slope and intercept fitting.
close all;
logpow= log10([stimPower]')
% allresps = [single_cone_control_response(:,1) single_cone_response];
allresps = [single_cone_response(valid,:)];

regressions = zeros(2,length(allresps));
xlin = [ones(length(logpow),1) logpow];


for i=1:length(allresps)
   
    regressions(:,i) = xlin\allresps(i,:)';
    
end



figure;
plot(regressions(1,:), regressions(2,:), 'k.');
xlabel('Intercept');
ylabel('Slope');
title('Slope vs intercept of dose response curve.');
% axis([-1 .5 -.5 1]);
axis square;
if saveplots
    saveas(gcf, 'dose_response_slope_v_int.svg'); 
end

% Project our regressions along our second eigenvector- COEFFS are the
% eiegenvectors, latent the eigenvalues
[COEFF, regressionscore,~,~,~,mu] = pca(regressions');

figure;
reghist = histogram(regressionscore(:,2),'BinWidth',log10(1.025));
title('Slope and Intercept data projected on second eigenvector');
if saveplots
    saveas(gcf, 'second_eigen_projected_data.png'); 
end

% regtable = array2table(regressions');
% regtable.Class = categorical(lessdensitometry_threshold)

mainpeakind = mode(discretize(regressionscore(:,2), reghist.BinEdges));
mainpeakval = reghist.Values(mainpeakind);
mainpeak = reghist.BinEdges(mainpeakind);

% Fit a gaussian mixture model to determine which is which.
sigmar = std(regressionscore(:,2),'omitnan')
sigmar =repmat(sigmar,[1 1 2]);
startstruct = struct('mu',[min(reghist.BinEdges); mainpeak],'Sigma', sigmar,'ComponentProportion',[.15; .85]);

gausfits=fitgmdist(regressionscore(:,2),2,'Start', startstruct)


range = min(regressionscore(:,2)):.001:max(regressionscore(:,2));
leftmix=pdf('Normal',range',gausfits.mu(1),sqrt(gausfits.Sigma(1)));
rightmix=pdf('Normal',range',gausfits.mu(2),sqrt(gausfits.Sigma(2)));
mewomix=pdf(gausfits,range');

normleftmix = mainpeakval.*gausfits.ComponentProportion(1).*leftmix./max(leftmix);
normrightmix = mainpeakval.*gausfits.ComponentProportion(2).*rightmix./max(rightmix);
normmewomix = mainpeakval.*mewomix./max(mewomix);


% RESPONSE_THRESHOLD = max(range(normleftmix>normrightmix))

hold on;
plot(range,normleftmix); 
plot(range,normrightmix);
plot(range,normmewomix);

%% Slope vs intercept with densitometry overlay

allvalidensresps = [densitometry_fit_amplitude(valid,:)];
lessdensitometry_threshold = (allvalidensresps<=DENSTOMETRY_THRESHOLD);
figure; hold on;
wut=0;
for i=1:length(allresps)    
   if lessdensitometry_threshold(i) || isnan(allvalidensresps(i))
       plot(regressions(1,i), regressions(2,i),'b.');       
   else
       plot(regressions(1,i), regressions(2,i),'r.');       
   end    
end
xlabel('Intercept');
ylabel('Slope');
title('Slope vs intercept of dose response curve, labelled with densitometry.');
% axis([-1 .5 -.5 1]);
axis square;
if saveplots
    saveas(gcf, 'dose_response_slope_v_int_wdens.svg'); 
end
hold off;


lowdens = regressions(:,lessdensitometry_threshold);
lowdensind = find(lessdensitometry_threshold);
projlowdens = (lowdens'-mu)*COEFF;

highintrinsic_lowdens = lowdensind(projlowdens(:,2) > -0.07);

figure(3); hold on;
reghist = histogram(projlowdens(:,2),'BinWidth',log10(1.025)); % 'Normalization','cumcount'
title('Slope and Intercept data projected on second eigenvector');
if saveplots
    saveas(gcf, 'second_eigen_projected_data.png'); 
end

figure; % Look at intrinsic responses of low densitometry cells
for interind=highintrinsic_lowdens'
    clf;
    subplot(2,1,1);
    plot(densitometry_vect_times{interind}, densitometry_vect_ref{interind}, '.');
    subplot(2,1,2);
    plot(stim_cell_var(interind,:));
    pause;
end

%% Look at intrinsic responses of low responding intrinsic cells with higher densitometry
confirmlowdens = lowdensind(projlowdens(:,2) <= -0.07);

lowintrinsic_lowdens = false(size(lessdensitometry_threshold));
lowintrinsic_lowdens(confirmlowdens) = true;

lowintrinsic_nodens = xor((regressionscore(:,2)<=-0.07), lowintrinsic_lowdens);
lowintrinsic_nodens =find(lowintrinsic_nodens );
figure; 
for interind=lowintrinsic_nodens'
    clf;
    subplot(2,1,1);
    plot(densitometry_vect_times{interind}, densitometry_vect_ref{interind}, '.');
    subplot(2,1,2);
    plot(stim_cell_var(interind,:));
    pause;
end

% Start pulling apart what about those single cones causes those spikes?


%% Amplitude vs Median response
figure; hold on;
for i=1:length(single_cone_mat_files)
    plot(stimTTP(:,i), stimAmp(:,i), '.');
end
xlabel('Std dev reponse');
ylabel('Absolute Mean reponse');
legend(strrep(single_cone_mat_files(loadinds),'_','\_'));
if saveplots
    saveas(gcf, ['comparative_responses.png']); 
end

%% Bland-Altman plot

response_compare = [single_cone_response(:,1) single_cone_control_response(:,1)]
meanresp = mean(response_compare,2);
diffresp = diff(response_compare,[],2);

diffbias = mean(diffresp(:),'omitnan');
stddiff = 1.96*std(diffresp(:),'omitnan');
LOA = [diffbias-stddiff diffbias-stddiff;
       diffbias+stddiff diffbias+stddiff];
  
figure; plot(meanresp,diffresp,'.'); hold on;
currentsize = axis;
plot(currentsize(1:2), LOA(1,:),'r-.')
plot(currentsize(1:2), [diffbias diffbias],'b')
plot(currentsize(1:2), LOA(2,:),'r-.'); hold off;



%% Boxplot of the amplitudes from each intensity.
figure;
boxplot(single_cone_response,'notch','on');
ylabel('Stimulus amplitude');

if saveplots
    saveas(gcf, 'allresps_boxplot.png');
end

%% Lognormal fitting of data.
hold on;
xvals=0:0.01:12;
for i=1:length(single_cone_mat_files)
    if any(single_cone_response(valid,i)<=0)
        fitted(i)=fitdist(single_cone_response(valid,i)-min(single_cone_response(valid,i))+.01,'Weibull');
        plot(xvals+min(single_cone_response(valid,i)),pdf(fitted(i),xvals));
    else        
        fitted(i)=fitdist(single_cone_response(valid,i),'Weibull');
        plot(xvals,pdf(fitted(i),xvals));
    end
    
    
end

%% Histograms of the response from each mat file.
close all;
% lessdensitometry_threshold = (densitometry_fit_amplitude<=DENSTOMETRY_THRESHOLD) & valid & valid_densitometry;
figure; hold on;
for i=1:length(single_cone_mat_files)
        
    if logmode        
        histogram(single_cone_response(valid,i),'BinWidth',log10(1.035),'Normalization','pdf');
%         histogram(single_cone_response(valid,i),'BinWidth',log10(1.025),'FaceColor','red');        
%         histogram(single_cone_response(lessdensitometry_threshold&valid,i),'BinWidth',log10(1.025),'FaceColor','blue');
    else
        histogram(single_cone_response(valid,i),'BinWidth',0.1,'Normalization','pdf');
    end
    title(strrep(single_cone_mat_files{loadinds(i)},'_','\_'));
    
%     if saveplots
%     saveas(gcf, [single_cone_mat_files{loadinds(i)} '_allresps_histogramsplot.png']);
%     saveas(gcf, [single_cone_mat_files{loadinds(i)} '_allresps_histogramsplot.svg']);
%     end
end
legend(strrep(single_cone_mat_files(loadinds),'_','\_'));

xlabel('Aggregate Response');
ylabel('Number of Cones');
 if logmode   
    axis([-0.2 1.2 0 45])
 else
     axis([-0.5 13 0 1.2])
 end
if saveplots
    saveas(gcf, [single_cone_mat_files{loadinds(i)} '_allresps_histogramsplot_nonlog.png']);
    saveas(gcf, [single_cone_mat_files{loadinds(i)} '_allresps_histogramsplot_nonlog.svg']);
end



%% Vs plots
close all;
for i=1:length(single_cone_mat_files)
    figure; hold on;
    plot(single_cone_control_response(:,i), single_cone_response(:,i),'k.');
    if logmode
        plot([-0.5 1.5],[-0.5 1.5],'k');
        axis equal;axis([-0.5 1.5 -0.5 1.5]); 
    else
        plot([-10 10],[-10 10],'k');
        axis equal; axis([-0.5 2 -0.5 10]); 
    end
    xlabel('Control Response'); ylabel('Stmulus Response')
    if saveplots
        saveas(gcf, [single_cone_mat_files{loadinds(i)}(1:end-4) '_VS_plot.png']);
    end
end

%% Display Cells under the 1:1 line
lessdensitometry_threshold = (densitometry_fit_amplitude<=DENSTOMETRY_THRESHOLD) & valid & valid_densitometry;
% lessdensitometry_threshold = diffvalid < abs(min(diffvalid))*0.8 & valid;

% lessdensitometry_threshold= (single_cone_response<single_cone_control_response) & valid;

% lessdensitometry_threshold = (single_cone_response<0.3) & valid & valid_densitometry;
% valid = valid & valid_densitometry;

for i=1:length(single_cone_mat_files)
    figure; hold on;    
    plot(single_cone_control_response(valid,i),single_cone_response(valid,i),'k.');
    plot([-10 10],[-10 10],'k');
    plot(single_cone_control_response(lessdensitometry_threshold(:,i)&valid,i),single_cone_response(lessdensitometry_threshold(:,i)&valid,i),'r.');
         
    thelessthan{i} = find(lessdensitometry_threshold(:,i)==1);
    if logmode
        axis square;axis([-0.5 1.5 -0.5 1.5]); 
        xlabel('Log Control Response (mean control subtracted)')
        ylabel('Log Stimulus Response (mean control subtracted)');
    else
        axis equal;axis([-1 5 -1 20]);
        xlabel('Control Response (mean control subtracted)')
        ylabel('Stimulus Response (mean control subtracted)');
    end
    
    if saveplots
        saveas(gcf, [single_cone_mat_files{loadinds(i)}(1:end-4) '_VS_lessthan_plot.png']);
    end
    
    generate_spatial_map(single_cone_response(:,i), allcoords, lessdensitometry_threshold(:,i), single_cone_mat_files(loadinds(i)), '_VS', saveplots);
end

%% Angular coordinate histograms of VS plots.

for i=1:length(single_cone_mat_files)
    

    figure; hold on;
    histogram(single_cone_response(valid,i),40);
    title(strrep(single_cone_mat_files{loadinds(i)}(1:end-4),'_','\_'))
    if logmode
        xlabel('Log Stimulus response');
    else
        xlabel('Stimulus response');
    end
    ylabel('Number of cones');
    
    if saveplots
        saveas(gcf, [single_cone_mat_files{loadinds(i)}(1:end-4) '_resp_histogram.png']);
    end
end

%% Display results vs densitometry
% load('/local_data/Dropbox/General_Postdoc_Work/Dynamic_Densitometry/11049/Dynamic_Densitometry_combined_4_sec_545b25nm_3uW_20_single_cone_signals.mat')
close all;
lessdensitometry_threshold = (densitometry_fit_amplitude<=DENSTOMETRY_THRESHOLD) & valid;

% lessdensitometry_threshold = (single_cone_response(:,4)<RESPONSE_THRESHOLD) 
% lessdensitometry_threshold = all(lessdensitometry_threshold,2);


for i=3:length(single_cone_mat_files)
    
    figure; hold on;
    plot(single_cone_control_response(valid,i),single_cone_response(valid,i),'k.');
    plot(single_cone_control_response(lessdensitometry_threshold,i),single_cone_response(lessdensitometry_threshold,i),'r.');
    plot([-10 10],[-10 10],'k');
         
    thelessthan{i} = find(lessdensitometry_threshold==1);
    if logmode
        axis square;axis([-0.5 1.5 -0.5 1.5]); 
        xlabel('Log Control Response (mean control subtracted)')
        ylabel('Log Stimulus Response (mean control subtracted)');
    else
        axis equal;axis([-0.5 2 0 6]);
        xlabel('Control Response (mean control subtracted)')
        ylabel('Stimulus Response (mean control subtracted)');
    end
    
    if saveplots
        saveas(gcf, [single_cone_mat_files{loadinds(i)}(1:end-4) '_VS_plot.png']);
        saveas(gcf, [single_cone_mat_files{loadinds(i)}(1:end-4) '_VS_plot.svg']);
    end
    
    generate_spatial_map(single_cone_response(:,i), allcoords, valid & valid_densitometry,[0 .7], single_cone_mat_files(loadinds(i)), '_VS', saveplots, lessdensitometry_threshold);
    
end

%% 

% lessdensitometry_threshold = (single_cone_response<RESPONSE_THRESHOLD) & (densitometry_fit_amplitude<=DENSTOMETRY_THRESHOLD) & valid & valid_densitometry;
 lessdensitometry_threshold = (densitometry_fit_amplitude<=DENSTOMETRY_THRESHOLD) & valid_densitometry;
numlowresp = 1;
figure; clf;
[V,C] = voronoin(allcoords,{'QJ'});
colors = ['rgbym'];
for j=1:length(single_cone_mat_files)
    
    for i=1:size(allcoords,1)
        
        vertices = V(C{i},:);

        if all(vertices(:,1)<max(allcoords(:,1))) && all(vertices(:,2)<max(allcoords(:,1))) ... % [xmin xmax ymin ymax] 
                                && all(vertices(:,1)>0) && all(vertices(:,2)>0) %&& ~isnan(single_cone_response(i,j))

            if all(lessdensitometry_threshold(i))
                patch(V(C{i},1),V(C{i},2),ones(size(V(C{i},1))),'FaceColor', 'w');
                numlowresp = numlowresp + 1;
            elseif lessdensitometry_threshold(i)
                patch(V(C{i},1),V(C{i},2),ones(size(V(C{i},1))),'FaceColor', colors(j));
            end
        end
    end
end
axis image;
axis([0 max(allcoords(:,1)) 0 max(allcoords(:,2)) ])    
 set(gca,'Color','k');     

% if saveplots
%     saveas(gcf, [single_cone_mat_files{loadinds(i)}(1:end-4) '_agreement_plot.png']);
% end

%% Repeatability of timepoints 1 and 2.
lessdensitometry_threshold = (densitometry_fit_amplitude<=DENSTOMETRY_THRESHOLD) & valid & valid_densitometry;

lownotdens = find((lessdensitometry_threshold < (valid & valid_densitometry) ) & ...
             (single_cone_response(:,1)<0.45 & single_cone_response(:,2)<0.45));


figure; hold on;
plot(single_cone_response(:,1),single_cone_response(:,2),'k.');
plot([-10 10],[-10 10],'k');
axis equal; axis([-0.5 2 -0.5 15]);
axis([-0.5 10 -0.5 10]);
xlabel('Timepoint 1'); ylabel('Timepoint 2');
title('Responses between both time points.')
% plot(single_cone_response(lownotdens,1),single_cone_response(lownotdens,2),'b.');
plot(single_cone_response(lessdensitometry_threshold,1),single_cone_response(lessdensitometry_threshold,2),'r.');

if saveplots
    saveas(gcf, [single_cone_mat_files{1}(1:end-4) '_Repeat_plot_thresh_' num2str(DENSTOMETRY_THRESHOLD) '.png']);
end

% figure; hold on;
% % plot(mean(single_cone_response(lownotdens,:),2), densitometry_fit_amplitude(lownotdens),'b*')
% plot(mean(single_cone_response(lessdensitometry_threshold,:),2), densitometry_fit_amplitude(lessdensitometry_threshold),'r*')
% xlabel('Mean cone response (std dev + median)');
% ylabel('Densitometry fit amplitude');

if saveplots
    saveas(gcf, [single_cone_mat_files{1}(1:end-4) '_dens_vs_mag_plot_thresh_' num2str(DENSTOMETRY_THRESHOLD) '.png']);
end

%% Densitometry gaussian mixture model stuff

% densitometry_fit_amplitude=densitometry_fit_amplitude(densitometry_fit_amplitude>-0.1);
% densitometry_fit_amplitude=densitometry_fit_amplitude(valid);

close all;
figure;
reghist = histogram(densitometry_fit_amplitude','BinWidth',.015, 'Normalization', 'pdf');



mainpeakind = mode(discretize(densitometry_fit_amplitude, reghist.BinEdges));
mainpeakval = reghist.Values(mainpeakind);
mainpeak = reghist.BinEdges(mainpeakind);

% Fit a gaussian mixture model to determine which is which.
sigmar = std(densitometry_fit_amplitude,'omitnan')
sigmar =repmat(sigmar,[1 1 2]);
startstruct = struct('mu',[min(reghist.BinEdges); mainpeak],'Sigma', sigmar,'ComponentProportion',[.15; .85]);

gausfits=fitgmdist(densitometry_fit_amplitude,2,'Start', startstruct)

range = min(densitometry_fit_amplitude):.001:max(densitometry_fit_amplitude);
leftmix=pdf('Normal',range',gausfits.mu(1),sqrt(gausfits.Sigma(1)));
rightmix=pdf('Normal',range',gausfits.mu(2),sqrt(gausfits.Sigma(2)));
mewomix=pdf(gausfits,range');

normleftmix = gausfits.ComponentProportion(1).*leftmix;
normrightmix = gausfits.ComponentProportion(2).*rightmix;
% normleftmix = mainpeakval.*gausfits.ComponentProportion(1).*leftmix./max(leftmix);
% normrightmix = mainpeakval.*gausfits.ComponentProportion(2).*rightmix./max(rightmix);
% normmewomix = mainpeakval.*mewomix./max(mewomix);


hold on;
plot(range,normleftmix); 
plot(range,normrightmix);
% plot(range,mewomix);
plot(range,normleftmix+normrightmix);
axis([-0.2 .6 0 7])
if saveplots
    saveas(gcf, 'densitometry_amplitudes.svg');
end

DENSTOMETRY_THRESHOLD = max(range(normleftmix>normrightmix))

% 
% populationratio = normleftmix./normrightmix;
% 
% DENSTOMETRY_THRESHOLD = max(range(populationratio>=2.5))

%% Intrinsic gaussian mixture model stuff


close all;
figure;

response_of_interest = single_cone_response(valid,4);
% response_of_interest(response_of_interest<.34 &
% response_of_interest>.328)=[]; % Screwing around with getting it to converge.
% response_of_interest(response_of_interest<.3884 & response_of_interest>.3735)=[];

% reghist = histogram(response_of_interest,'BinWidth',log10(1.035), 'Normalization', 'pdf');
reghist = histogram(response_of_interest,36, 'Normalization', 'pdf');



mainpeakind = mode(discretize(response_of_interest, reghist.BinEdges));
mainpeakval = reghist.Values(mainpeakind);
mainpeak = reghist.BinEdges(mainpeakind);

% Fit a gaussian mixture model to determine which is which.
sigmar = std(response_of_interest,'omitnan')
sigmar =repmat(sigmar,[1 1 2]);
sigmar(1,1,1) = 0.005;
startstruct = struct('mu',[min(reghist.BinEdges); mainpeak],'Sigma', sigmar,'ComponentProportion',[.05; .95],'Display','iter');

gausfits=fitgmdist(response_of_interest,2,'Start', startstruct)

range = min(response_of_interest):.001:max(response_of_interest);
leftmix=pdf('Normal',range',gausfits.mu(1),sqrt(gausfits.Sigma(1)));
rightmix=pdf('Normal',range',gausfits.mu(2),sqrt(gausfits.Sigma(2)));
mewomix=pdf(gausfits,range');

normleftmix = gausfits.ComponentProportion(1).*leftmix;
normrightmix = gausfits.ComponentProportion(2).*rightmix;
% normleftmix = mainpeakval.*gausfits.ComponentProportion(1).*leftmix./max(leftmix);
% normrightmix = mainpeakval.*gausfits.ComponentProportion(2).*rightmix./max(rightmix);
% normmewomix = mainpeakval.*mewomix./max(mewomix);


hold on;
plot(range,normleftmix); 
plot(range,normrightmix);
% plot(range,mewomix);
plot(range,normleftmix+normrightmix);
axis([0 1.2 0 3])
if saveplots
    saveas(gcf, 'intrinsic_response_amplitudes_fit.svg');
end

RESPONSE_THRESHOLD = max(range(normleftmix>normrightmix))

RESPONSE_THRESHOLD=0.32

dens_s_cone = lessdensitometry_threshold;
intrinsic_s_cone = (response_of_interest<=RESPONSE_THRESHOLD);


agree_s_cones = sum(dens_s_cone & intrinsic_s_cone);

agree_lm_cones = sum(~dens_s_cone & ~intrinsic_s_cone);

intrinsic_s_dens_lm = sum( ~dens_s_cone & intrinsic_s_cone )
intrinsic_lm_dens_s = sum( dens_s_cone & ~intrinsic_s_cone )

individual_agreement = (agree_s_cones +agree_lm_cones)./size(dens_s_cone,1)

%     S_L/M_
%   S|
% L/M|

% Where top is denisometry, left is intrinsic:
individual_truth = [ agree_s_cones  intrinsic_s_dens_lm;
                    intrinsic_lm_dens_s agree_lm_cones]

% unionofboth = sum(lessdensitometry_threshold&(response_of_interest<=RESPONSE_THRESHOLD))
% goldstd = sum(lessdensitometry_threshold);
% ourmetric = sum(response_of_interest<=RESPONSE_THRESHOLD);
% 
% dice_individual = (2*unionofboth)/(goldstd+ourmetric)
