% track_green_gripper.m
clear; close all; clc;

%% ---------------- USER PARAMETERS ----------------
videoFile = 'ProtoSwing.mp4';   % video filename
outputCSV  = 'tracked_green_angles2.csv';

% desired sampling time vector for comparison with simulation
Time = 0:0.01:1.19;  % 120 samples (0 .. 1.19 step 0.01)

% HSV thresholds for green (tune if necessary)
hueMin = 0.20;  % e.g. ~0.2
hueMax = 0.45;  % e.g. ~0.45
satMin = 0.30;  % 0..1
valMin = 0.20;  % 0..1

% Morphological and detection params
minAreaPx = 80;        % minimum blob area (pixels) to accept
searchRadius = 80;     % pixels: local search radius for ROI mask
templateSize = 41;     % odd size template patch (px) used for normxcorr2 fallback

% Smoothing (Savitzky-Golay)
sgolayWin = 15;   % must be odd and <= number of valid samples
sgolayPoly = 3;

%% ---------------- Open video and read first frame ----------------
if ~isfile(videoFile)
    error('Video file not found: %s', videoFile);
end

vr = VideoReader(videoFile);
fps = vr.FrameRate;
nFrames = floor(vr.Duration * fps);
fprintf('Video: %s  fps=%.2f  frames=%d\n', videoFile, fps, nFrames);

firstFrame = readFrame(vr);          % read first frame for pivot & init detection
frameH = size(firstFrame,1); frameW = size(firstFrame,2);

%% ---------------- choose pivot ----------------
figure(1); imshow(firstFrame); title('Click pivot point (joint/base) and press Enter');
[xs, ys] = ginput(1);
pivot = round([xs(1), ys(1)]);
close(1);
fprintf('Pivot at px = %d, py = %d\n', pivot(1), pivot(2));

%% ---------------- initial detection of green marker ----------------
% Convert to HSV and detect green blob on first frame
hsv0 = rgb2hsv(firstFrame);
H0 = hsv0(:,:,1); S0 = hsv0(:,:,2); V0 = hsv0(:,:,3);
mask0 = (H0 >= hueMin) & (H0 <= hueMax) & (S0 >= satMin) & (V0 >= valMin);
mask0 = medfilt2(mask0, [3 3]);
mask0 = bwareaopen(mask0, minAreaPx);

% If multiple regions: choose the largest
stats = regionprops(mask0, 'Area', 'Centroid', 'BoundingBox');
if isempty(stats)
    % If initial detection fails, ask user to click the marker position
    figure; imshow(firstFrame); title('Could not auto-detect: click the green marker and press Enter');
    [cx, cy] = ginput(1);
    initPoint = round([cx, cy]);
    close;
else
    areas = [stats.Area];
    [~, idx] = max(areas);
    initPoint = round(stats(idx).Centroid);
end
fprintf('Initial marker at px = %d, py = %d\n', initPoint(1), initPoint(2));

% extract a template patch around initPoint for template matching fallback
tplSize = templateSize;
x0 = max(1, initPoint(1)-floor(tplSize/2));
y0 = max(1, initPoint(2)-floor(tplSize/2));
x1 = min(frameW, x0+tplSize-1);
y1 = min(frameH, y0+tplSize-1);
tpl = im2gray(firstFrame(y0:y1, x0:x1, :));   % initial template (grayscale)
tplW = size(tpl,2); tplH = size(tpl,1);

%% ---------------- Prepare logging ----------------
positions = nan(nFrames,2);
angles = nan(nFrames,1);
times = (0:nFrames-1)'/fps;

% reset video to beginning so loop reads from frame 1
vr.CurrentTime = 0;

%% ---------------- tracking loop ----------------
frameIdx = 1;
fprintf('Tracking... (press Ctrl-C to abort)\n');
while hasFrame(vr) && frameIdx <= nFrames
    frame = readFrame(vr);
    hsv = rgb2hsv(frame);
    H = hsv(:,:,1); S = hsv(:,:,2); V = hsv(:,:,3);

    % Primary: determine search center (previous valid position or init)
    if frameIdx==1
        searchCx = initPoint(1); searchCy = initPoint(2);
    else
        if ~isnan(positions(frameIdx-1,1))
            searchCx = positions(frameIdx-1,1);
            searchCy = positions(frameIdx-1,2);
        else
            % fallback to last valid index if available
            lastValid = find(~isnan(positions(1:frameIdx-1,1)), 1, 'last');
            if ~isempty(lastValid)
                searchCx = positions(lastValid,1);
                searchCy = positions(lastValid,2);
            else
                searchCx = initPoint(1); searchCy = initPoint(2);
            end
        end
    end

    % ROI bounds (clamped)
    rx1 = max(1, round(searchCx - searchRadius)); rx2 = min(frameW, round(searchCx + searchRadius));
    ry1 = max(1, round(searchCy - searchRadius)); ry2 = min(frameH, round(searchCy + searchRadius));
    if rx2 < rx1 || ry2 < ry1
        rx1 = 1; rx2 = frameW; ry1 = 1; ry2 = frameH;
    end

    Hroi = H(ry1:ry2, rx1:rx2);
    Sroi = S(ry1:ry2, rx1:rx2);
    Vroi = V(ry1:ry2, rx1:rx2);

    mask = (Hroi >= hueMin) & (Hroi <= hueMax) & (Sroi >= satMin) & (Vroi >= valMin);
    mask = medfilt2(mask, [3 3]);
    mask = bwareaopen(mask, minAreaPx);

    % regionprops on ROI (choose largest blob centroid)
    stats = regionprops(mask, 'Area', 'Centroid');
    if ~isempty(stats)
        areas = [stats.Area];
        [~, idx] = max(areas);
        c = stats(idx).Centroid;
        % convert centroid to full-frame coordinates
        cx = rx1 - 1 + c(1);
        cy = ry1 - 1 + c(2);
        positions(frameIdx,:) = [cx, cy];
        angles(frameIdx) = atan2(cy - pivot(2), cx - pivot(1));
    else
        % Fallback #1: template matching (normxcorr2) in a larger search patch
        grayF = im2gray(frame);
        bigX1 = max(1, round(searchCx - 2*searchRadius)); bigX2 = min(frameW, round(searchCx + 2*searchRadius));
        bigY1 = max(1, round(searchCy - 2*searchRadius)); bigY2 = min(frameH, round(searchCy + 2*searchRadius));
        searchPatch = grayF(bigY1:bigY2, bigX1:bigX2);
        foundMatch = false;

        try
            crr = normxcorr2(tpl, searchPatch);
            [~, imax] = max(abs(crr(:)));
            [ypeak, xpeak] = ind2sub(size(crr), imax(1));
            corr_offset = [xpeak - tplW, ypeak - tplH];
            match_x = bigX1 + corr_offset(1) + floor(tplW/2);
            match_y = bigY1 + corr_offset(2) + floor(tplH/2);
            % sanity check: ensure match is within image
            if match_x >=1 && match_x <= frameW && match_y >=1 && match_y <= frameH
                positions(frameIdx,:) = [match_x, match_y];
                angles(frameIdx) = atan2(match_y - pivot(2), match_x - pivot(1));
                foundMatch = true;
            end
        catch
            % normxcorr2 can fail on very small patches - ignore and fallback
            foundMatch = false;
        end

        if ~foundMatch
            positions(frameIdx,:) = [nan, nan];
        end
    end

    % Final fallback: linear prediction from last two valid positions
    if isnan(positions(frameIdx,1))
        validIdx = find(~isnan(positions(1:frameIdx-1,1)));
        if length(validIdx) >= 2
            k1 = validIdx(end-1); k2 = validIdx(end);
            p1 = positions(k1,:); p2 = positions(k2,:);
            dt = (k2 - k1) / fps;
            if dt > 0
                vel = (p2 - p1) / dt;
                pred = p2 + vel * (1 / fps);
                % clamp to image
                pred(1) = min(max(pred(1),1),frameW);
                pred(2) = min(max(pred(2),1),frameH);
                positions(frameIdx,:) = pred;
                angles(frameIdx) = atan2(pred(2) - pivot(2), pred(1) - pivot(1));
            else
                positions(frameIdx,:) = [nan, nan];
            end
        else
            positions(frameIdx,:) = [nan, nan];
        end
    end

    % Optional: Update template occasionally from current frame if valid (to adapt appearance)
    if ~isnan(positions(frameIdx,1)) && mod(frameIdx,20) == 0
        cx = round(positions(frameIdx,1)); cy = round(positions(frameIdx,2));
        tx0 = max(1, cx - floor(tplSize/2));
        ty0 = max(1, cy - floor(tplSize/2));
        tx1 = min(frameW, tx0 + tplSize - 1);
        ty1 = min(frameH, ty0 + tplSize - 1);
        % ensure crop is valid
        if ty1 > ty0 && tx1 > tx0
            tpl = im2gray(frame(ty0:ty1, tx0:tx1, :));
            tplW = size(tpl,2); tplH = size(tpl,1);
        end
    end

    % display overlay every N frames
    if mod(frameIdx,5) == 0
        imshow(frame); hold on;
        plot(pivot(1), pivot(2), 'g+', 'MarkerSize',10, 'LineWidth',2);
        if ~isnan(positions(frameIdx,1))
            plot(positions(frameIdx,1), positions(frameIdx,2), 'ro', 'MarkerSize',8, 'LineWidth',2);
            line([pivot(1), positions(frameIdx,1)], [pivot(2), positions(frameIdx,2)], 'Color','y', 'LineWidth',2);
        else
            text(10,30,'TRACK LOST','Color','r','FontSize',14,'FontWeight','bold');
        end
        title(sprintf('Frame %d / %d (t=%.3fs)', frameIdx, nFrames, (frameIdx-1)/fps));
        hold off; drawnow;
    end

    frameIdx = frameIdx + 1;
end

%% ---------------- Post-process & save ----------------
processed = min(nFrames, frameIdx-1);
positions = positions(1:processed,:);
angles = angles(1:processed);
times = (0:processed-1)'/fps;

% unwrap and smooth angle
validAngles = ~isnan(angles);
angles_unwrap = angles;
if any(validAngles)
    angles_unwrap(validAngles) = unwrap(angles(validAngles));  % unwrap only valid segments
end

if sum(validAngles) >= sgolayWin
    tmp = angles_unwrap(validAngles);
    tmp_sm = sgolayfilt(tmp, sgolayPoly, sgolayWin);
    angles_smoothed = nan(size(angles_unwrap));
    angles_smoothed(validAngles) = tmp_sm;
else
    angles_smoothed = angles_unwrap;
end

% Save CSV (original output)
T = table(times, positions(:,1), positions(:,2), angles, angles_smoothed, ...
    'VariableNames', {'t_s','x_px','y_px','angle_raw_rad','angle_smooth_rad'});
writetable(T, outputCSV);
fprintf('Saved tracked data to %s (frames processed: %d)\n', outputCSV, processed);

% interpolate onto requested Time vector and save sampled CSV
sampleCSV = 'tracked_green_angles_sampled2.csv';
% Use only frames that have valid smoothed angles and positions for interpolation
validPosIdx = find(~isnan(angles_smoothed) & ~isnan(positions(:,1)));

if numel(validPosIdx) >= 2
    t_valid = times(validPosIdx);
    x_valid = positions(validPosIdx,1);
    y_valid = positions(validPosIdx,2);
    theta_valid = angles_smoothed(validPosIdx);

    % Interpolate using pchip; extrapolate with NaN beyond data range for safety
    x_interp = interp1(t_valid, x_valid, Time, 'pchip', NaN);
    y_interp = interp1(t_valid, y_valid, Time, 'pchip', NaN);
    theta_interp = interp1(t_valid, theta_valid, Time, 'pchip', NaN);
else
    % Not enough valid data -> fill with NaNs
    x_interp = nan(size(Time));
    y_interp = nan(size(Time));
    theta_interp = nan(size(Time));
    warning('Not enough valid tracked points to interpolate onto requested Time vector. Sampled output will be NaN.');
end

Ts = table(Time', x_interp', y_interp', theta_interp', ...
    'VariableNames', {'t_s','x_px','y_px','angle_smooth_rad'});
writetable(Ts, sampleCSV);
fprintf('Saved sampled tracked data to %s (samples=%d)\n', sampleCSV, numel(Time));

% Plot final angle vs time
figure; hold on;
plot(times, angles_unwrap, 'b.-', 'DisplayName','raw (unwrap)');
plot(times, angles_smoothed, 'r-', 'LineWidth',1.5, 'DisplayName','smoothed');
xlabel('Time (s)'); ylabel('Angle (rad)'); legend; grid on;
title('Tracked angle about pivot');

