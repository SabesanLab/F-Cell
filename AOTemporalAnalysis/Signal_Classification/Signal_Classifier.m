
clear;
close all force;


baseDir = pwd; %uigetdir(pwd);
profileDataNames = read_folder_contents(baseDir ,'mat');

wavelet = 'gaus3';

stimd=0;
stimless=0;
x45=[];
x92=[];
x04=[];

for j=1:length(profileDataNames)

    load(fullfile(baseDir, profileDataNames{j}));
    
    % Remove the empty cells
    norm_stim_cell_reflectance = norm_stim_cell_reflectance( ~cellfun(@isempty,norm_stim_cell_reflectance) );
    stim_cell_times            = stim_cell_times(  ~cellfun(@isempty,stim_cell_times) );
    norm_control_cell_reflectance = norm_control_cell_reflectance( ~cellfun(@isempty,norm_control_cell_reflectance)  );
    control_cell_times            = control_cell_times( ~cellfun(@isempty,control_cell_times) );
    
    all_spect = zeros(length(norm_stim_cell_reflectance), 256);
    cutoff = 0;
    for i=1:length(norm_control_cell_reflectance)
        
        times = control_cell_times{i};
        signal = norm_control_cell_reflectance{i}(times>=cutoff);
        times = times(times>=cutoff);        
        % The profile itself

        
        % Spectrogram calculation
%         figure(2); 
%         spectrogram(signal,8,'yaxis');
%         Nx = length(signal);
%         nsc = floor(Nx/16);
%         nov = floor(nsc/2);
%         nff = max(256,2^nextpow2(nsc));

%         figure(3);
%         spectrogram(signal,hamming(nsc),nov,nff,'yaxis', 16.6);
        

%         figure;
        figure(1); hold on; title('Control cones');
        
        %Continuous
        D5 = cwt(signal,16,wavelet);
        
        if length(D5) >= 190
        D5 = D5(1:190);
        
        %Discrete
%         [C, L] = wavedec(signal,5,wavelet);
%         [cD1,cD2,cD3,cD4,cD5] = detcoef(C,L,1:5);
% 
%         D4 = wrcoef('d',C,L,wavelet,4);
%         D5 = wrcoef('d',C,L,wavelet,5);

%         subplot(5,1,1); plot(D1);
%         subplot(5,1,2); plot(D2);
%         subplot(5,1,3); plot(D3);
%         subplot(5,1,4); plot(D4);
%         subplot(5,1,5);
         plot(D5);
        
        

        controlcoeffs(i,:) = D5;
        plot(D5); axis([0 250 -10 10])
%         hp = pcolor(1:length(signal),f,abs(cfs)); hp.EdgeColor = 'none';
%         set(gca,'YScale','log');
%         xlabel('Sample'); ylabel('log10(f)');
        end
%         pause(0.1);
    end
    
    simplethreshold = 3*std(controlcoeffs);
    
    stimless=[];
    stimd=[];
    for i=1:length(norm_stim_cell_reflectance)
        times  = stim_cell_times{i};
        signal = norm_stim_cell_reflectance{i}(times>=cutoff);
        times = times(times>=cutoff);
        
        figure(10);
        plot(times, signal); hold on;
        plot(66:98,ones(33,1)*max(signal), 'r*'); hold off;

        %Continuous
        D5 = cwt(signal,16,wavelet);
        
        if length(D5) >= 190

            D5 = D5(1:190);

            %Discrete
    %         [C, L] = wavedec(signal,5,wavelet);
    %         [cD1,cD2,cD3,cD4,cD5] = detcoef(C,L,1:5);
    %         
    %         D4 = wrcoef('d',C,L,wavelet,4);
    %         D5 = wrcoef('d',C,L,wavelet,5);

            if all(D5<simplethreshold & D5>-simplethreshold)
                stimless = [stimless i];

                figure(3);  title('StimLESS cones'); hold on;            
                plot(D5); 
                axis([0 250 -10 10])
            else            
                stimd = [stimd i];

                figure(2);  title('Stim cones'); hold on;
                plot(D5); axis([0 250 -10 10])
            end
        end
    end

    cind = randperm(length(norm_control_cell_reflectance), 11);
    aind = randperm(length(stimless), 11);
    sind = randperm(length(stimd), 11);
    
    x45 = [x45; control_cell_times(cind) norm_control_cell_reflectance(cind)];    
    x92 = [x92; stim_cell_times(stimless(aind)) norm_stim_cell_reflectance(stimless(aind))];
    x04 = [x04; stim_cell_times(stimd(sind)) norm_stim_cell_reflectance(stimd(sind))];
%     imagesc(avg_spect);
end