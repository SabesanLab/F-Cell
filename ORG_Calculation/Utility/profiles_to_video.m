function []=profiles_to_video(temporal_profiles, video_size, output_filename)


vidObj = VideoWriter(output_filename,'Uncompressed AVI');
open(vidObj);

minall = quantile(temporal_profiles(:),0.01);
maxall = quantile(temporal_profiles(:)-minall,0.99);
for i=1:video_size(3)

    scaled_norm_chg_im = reshape(temporal_profiles(:,i)-minall, video_size(2), video_size(1));
    scaled_norm_chg_im = scaled_norm_chg_im ./maxall;

    % Clamp the values over the 95th percentile.
    scaled_norm_chg_im(scaled_norm_chg_im>1) = 1;
    scaled_norm_chg_im(scaled_norm_chg_im<0) = 0;
    writeVideo(vidObj, scaled_norm_chg_im);

end

close(vidObj);