
close all; clear all; clc;
 
LAPLACIAN_THRESHOLD = 0.9;
STD_THRESHOLD       = 0.27;
MEAN_DROP_THRESHOLD = 0.27;
DETECT_DELAY_SEC    = 1.0;
FPS                 = 15;
GAUSS_SIGMA         = 3;
BATCH_SIZE          = 50;
MAT_FILE            = 'output_emilie.mat';
COORD_FILE          = 'parking_spaces.mat';
 
COLOR_GREEN = uint8([0   255 0  ]);
COLOR_BLUE  = uint8([0   0   255]);
COLOR_WHITE = uint8([255 255 255]);
COLOR_RED   = uint8([255 0   0  ]);
 
MIN_FRAME_GAP = 30;
 
fprintf("Opening %s...", MAT_FILE);
mf = matfile(MAT_FILE);
 
whos_info  = whos(mf);
video_name = '';
for i = 1:numel(whos_info)
    video_name = whos_info(i).name;
    fullSize   = whos_info(i).size;
    fprintf('  Found "%s"  full size: %s\n', video_name, mat2str(fullSize));
end
if isempty(video_name)
    error('No variable found in %s', MAT_FILE);
end
 
image_height = fullSize(1);
image_width  = fullSize(2);
n_total      = fullSize(3);
 
space_data = [];
if isfile(COORD_FILE)
    ans_ = input('Found parking_spaces.mat. Reuse it? [y/n]: ','s');
    if strcmpi(strtrim(ans_),'y')
        tmp        = load(COORD_FILE);
        space_data = tmp.spaces_data;
        fprintf('Loaded %d spaces from file.\n', numel(space_data));
    end
end
if isempty(space_data)
    ret_frame_raw = loadSlice(mf, video_name, 1);
    space_data    = coordinatesGenerator(ret_frame_raw, image_height, image_width, COLOR_BLUE, COORD_FILE);
end
 
n_spaces = numel(space_data);
if n_spaces == 0
    error('No spaces defined.');
end
fprintf('Using %d parking spaces over %d frames.\n', n_spaces, n_total);
 
bounds = zeros(n_spaces, 4);
masks  = cell(n_spaces, 1);
for s = 1:n_spaces
    coords  = space_data(s).coordinates;
    x1=min(coords(:,1)); x2=max(coords(:,1));
    y1=min(coords(:,2)); y2=max(coords(:,2));
    bounds(s,:) = [x1 y1 x2-x1+1 y2-y1+1];
    lc = coords;
    lc(:,1) = coords(:,1) - x1 + 1;
    lc(:,2) = coords(:,2) - y1 + 1;
    masks{s} = poly2mask(lc(:,1), lc(:,2), y2-y1+1, x2-x1+1);
end
 
blurred0  = imgaussfilt(double(mf.(video_name)(:,:,1)), GAUSS_SIGMA);
mean_init = zeros(n_spaces, 1);
mean_ref  = zeros(n_spaces, 1);
for s = 1:n_spaces
    roi = genRoi(blurred0, bounds(s,:), masks{s}, image_height, image_width);
    mean_init(s) = mean(roi(masks{s}));
    mean_ref(s)  = mean_init(s);
end
 
statuses_std = false(n_spaces, 1);
statuses_lap = false(n_spaces, 1);
pending_std  = nan(n_spaces, 1);
pending_lap  = nan(n_spaces, 1);
 
mu_log     = zeros(n_spaces, n_total);
sigma_log  = zeros(n_spaces, n_total);
std_to_ref = zeros(n_spaces, n_total);
changes_lap = zeros(n_spaces, n_total);
changes_std = zeros(n_spaces, n_total);
 
vw_std = VideoWriter('parking_std.avi','Motion JPEG AVI');
vw_std.FrameRate = FPS;
open(vw_std);
 
vw_lap = VideoWriter('parking_laplacian.avi','Motion JPEG AVI');
vw_lap.FrameRate = FPS;
open(vw_lap);
 
match_frames       = [];
mismatch_frames    = [];
match_annotated    = {};
mismatch_annotated = {};
 
fprintf('\nProcessing %d frames in batches of %d...\n', n_total, BATCH_SIZE);
total_batches = ceil(n_total / BATCH_SIZE);
 
for batch_id = 1:total_batches
    f_start  = (batch_id-1)*BATCH_SIZE + 1;
    f_end    = min(batch_id*BATCH_SIZE, n_total);
    n_batch  = f_end - f_start + 1;
    fprintf('  Batch %d/%d  (frames %d-%d)...\n', batch_id, total_batches, f_start, f_end);
    batch_data = mf.(video_name)(:,:,f_start:f_end);
 
    for local_f = 1:n_batch
        global_f = f_start + local_f - 1;
        frame    = batchSlice(batch_data, local_f);
        blurred  = imgaussfilt(double(frame), GAUSS_SIGMA);
        posSec   = global_f / FPS;
 
        new_std = false(n_spaces, 1);
        new_lap = false(n_spaces, 1);
 
        for s = 1:n_spaces
            roi      = genRoi(blurred, bounds(s,:), masks{s}, image_height, image_width);
            roi_flat = roi(masks{s});
 
            cur_mean = mean(roi_flat);
            cur_std  = getSTD(roi_flat, cur_mean);
 
            mu_log(s,    global_f) = cur_mean;
            sigma_log(s, global_f) = cur_std;
            std_to_ref(s,global_f) = abs(cur_mean - mean_ref(s)) / mean_ref(s);
 
            new_std(s) = applySTD(blurred, bounds(s,:), masks{s}, ...
                                  STD_THRESHOLD,MEAN_DROP_THRESHOLD, image_height, image_width, mean_init(s));
            new_lap(s) = applyLaplacian(blurred, bounds(s,:), masks{s}, ...
                                        LAPLACIAN_THRESHOLD, image_height, image_width);
            changes_lap(s, global_f) = new_lap(s);
            changes_std(s, global_f) = new_std(s);

 
            if new_std(s)
                mean_init(s) = cur_mean;
            end
        end
 
        for s = 1:n_spaces
            [statuses_std(s), pending_std(s)] = debounce(statuses_std(s), new_std(s), pending_std(s), posSec, DETECT_DELAY_SEC);
            [statuses_lap(s), pending_lap(s)] = debounce(statuses_lap(s), new_lap(s), pending_lap(s), posSec, DETECT_DELAY_SEC);
        end
 
        ann_std = annotateFrame(frame, space_data, statuses_std, n_spaces, COLOR_GREEN, COLOR_RED, COLOR_WHITE, image_height, image_width);
        ann_lap = annotateFrame(frame, space_data, statuses_lap, n_spaces, COLOR_GREEN, COLOR_RED, COLOR_WHITE, image_height, image_width);
 
        writeVideo(vw_std, ann_std);
        writeVideo(vw_lap, ann_lap);
 
        free_std = sum(statuses_std);
        free_lap = sum(statuses_lap);
 
        if free_std == free_lap
            if numel(match_frames) < 2 && isSpacedEnough(global_f, match_frames, MIN_FRAME_GAP)
                match_frames(end+1)    = global_f;            %#ok<AGROW>
                match_annotated{end+1} = {ann_std, ann_lap};  %#ok<AGROW>
            end
        else
            if numel(mismatch_frames) < 2 && isSpacedEnough(global_f, mismatch_frames, MIN_FRAME_GAP)
                mismatch_frames(end+1)    = global_f;             %#ok<AGROW>
                mismatch_annotated{end+1} = {ann_std, ann_lap};   %#ok<AGROW>
            end
        end
    end
 
    clear batch_data;
    fprintf('    done.  STD free: %d  LAP free: %d  /  %d\n', ...
            sum(statuses_std), sum(statuses_lap), n_spaces);
end
 
close(vw_std);
close(vw_lap);
fprintf('\nVideos saved: parking_std.avi  parking_laplacian.avi\n');
 
plotComparison(match_annotated, mismatch_annotated, match_frames, mismatch_frames);
plotStats(mu_log, sigma_log, std_to_ref, n_spaces, n_total, FPS, changes_lap, changes_std);
 
 
%% ── Local functions ──────────────────────────────────────────────────────────
 
function ok = isSpacedEnough(candidate, existing, min_gap)
    if isempty(existing)
        ok = true;
        return;
    end
    ok = all(abs(candidate - existing) >= min_gap);
end
 
function plotStats(mu_log, sigma_log, std_to_ref, n_spaces, n_total, fps, changes_lap, changes_std)
    t      = (1:n_total) / fps;
    n_cols = 3;
    n_rows = n_spaces;
    colors = lines(n_spaces);

    % --- Figure 1: Mean Intensity | Std Dev | Deviation from Reference ---
    figure('Name','Per-Space Statistics','NumberTitle','off', ...
        'Position',[100 100 1400 260*n_rows]);

    for s = 1:n_spaces
        sp_label = sprintf('Space %d', s);
        c        = colors(s,:);

        ax1 = subplot(n_rows, n_cols, (s-1)*n_cols + 1);
        plot(t, mu_log(s,:), 'Color', c, 'LineWidth', 1.2);
        xlabel('Time (s)');
        ylabel('Mean intensity');
        title(sprintf('%s — Mean Intensity', sp_label));
        grid on;

        ax2 = subplot(n_rows, n_cols, (s-1)*n_cols + 2);
        plot(t, sigma_log(s,:), 'Color', c, 'LineWidth', 1.2);
        xlabel('Time (s)');
        ylabel('\sigma');
        title(sprintf('%s — Std Dev', sp_label));
        grid on;

        ax3 = subplot(n_rows, n_cols, (s-1)*n_cols + 3);
        plot(t, std_to_ref(s,:), 'Color', c, 'LineWidth', 1.2);
        yline(0.25, 'r--', 'Threshold (0.25)', ...
            'LabelHorizontalAlignment','left', 'LineWidth', 1.2);
        xlabel('Time (s)');
        ylabel('|mean - ref| / ref');
        title(sprintf('%s — Deviation from Reference Mean', sp_label));
        grid on;

        linkaxes([ax1 ax2 ax3], 'x');
    end

    sgtitle('Per-Space Mean Intensity  |  Std Dev  |  Deviation from Reference Mean', ...
        'FontSize', 13, 'FontWeight', 'bold');

    % --- Figure 2: Binary Occupied State ---
    n_cols2 = 2;
    figure('Name','Binary Occupied','NumberTitle','off', ...
        'Position',[100 100 1000 260*n_rows]);

    for s = 1:n_spaces
        sp_label = sprintf('Space %d', s);
        c        = colors(s,:);

        ax1 = subplot(n_rows, n_cols2, (s-1)*n_cols2 + 1);
        plot(t, changes_lap(s,:), 'Color', c, 'LineWidth', 1.2);
        xlabel('Time (s)');
        ylabel('Occupancy');
        title(sprintf('%s — Binary Occupied State LAP', sp_label));
        grid on;

        ax2 = subplot(n_rows, n_cols2, (s-1)*n_cols2 + 2);
        plot(t, changes_std(s,:), 'Color', c, 'LineWidth', 1.2);
        xlabel('Time (s)');
        ylabel('Occupancy');
        title(sprintf('%s — Binary Occupied State STD', sp_label));
        grid on;

        linkaxes([ax1 ax2], 'x');
    end

    sgtitle('Per-Space Binary Occupied State  |  LAP  |  STD', ...
        'FontSize', 13, 'FontWeight', 'bold');
end
 
function plotComparison(match_ann, mismatch_ann, match_fr, mismatch_fr)
    n_match    = numel(match_ann);
    n_mismatch = numel(mismatch_ann);
 
    if n_match == 0 && n_mismatch == 0
        fprintf('No frames collected for comparison figure.\n');
        return;
    end
 
    total_pairs = n_match + n_mismatch;
    figure('Name','Method Comparison','NumberTitle','off', ...
           'Position',[50 50 1400 400*total_pairs]);
 
    row = 1;
    for i = 1:n_match
        subplot(total_pairs, 2, (row-1)*2 + 1);
        imshow(match_ann{i}{1});
        title(sprintf('MATCH  frame %d  —  STD method', match_fr(i)), ...
              'Color','g','FontWeight','bold');
 
        subplot(total_pairs, 2, (row-1)*2 + 2);
        imshow(match_ann{i}{2});
        title(sprintf('MATCH  frame %d  —  Laplacian method', match_fr(i)), ...
              'Color','g','FontWeight','bold');
        row = row + 1;
    end
 
    for i = 1:n_mismatch
        subplot(total_pairs, 2, (row-1)*2 + 1);
        imshow(mismatch_ann{i}{1});
        title(sprintf('MISMATCH  frame %d  —  STD method', mismatch_fr(i)), ...
              'Color','r','FontWeight','bold');
 
        subplot(total_pairs, 2, (row-1)*2 + 2);
        imshow(mismatch_ann{i}{2});
        title(sprintf('MISMATCH  frame %d  —  Laplacian method', mismatch_fr(i)), ...
              'Color','r','FontWeight','bold');
        row = row + 1;
    end
 
    sgtitle('Green title = methods agree  |  Red title = methods disagree', ...
            'FontSize', 13);
end
 
function annotated = annotateFrame(frame, space_data, statuses, n_spaces, ...
                                   COLOR_GREEN, COLOR_RED, COLOR_WHITE, H, W)
    annotated = repmat(frame, [1 1 3]);
    for s = 1:n_spaces
        clr = COLOR_GREEN;
        if ~statuses(s), clr = COLOR_RED; end
        annotated = drawContours(annotated, space_data(s).coordinates, ...
                                 num2str(space_data(s).id+1), ...
                                 COLOR_WHITE, clr, H, W);
    end
end
 
function [new_status, new_pending] = debounce(status, candidate, pending, posSec, delay)
    new_status  = status;
    new_pending = pending;
    if ~isnan(pending) && candidate == status
        new_pending = nan;
        return;
    end
    if ~isnan(pending) && candidate ~= status
        if posSec - pending >= delay
            new_status  = candidate;
            new_pending = nan;
        end
        return;
    end
    if isnan(pending) && candidate ~= status
        new_pending = posSec;
    end
end
 
function spaces_data = coordinatesGenerator(ref_frame_raw, H, W, ~, coord_file)
    fprintf('\nCoordinates Generator\n');
    fprintf('Click 4 corners per space then press ENTER. Answer n when done.\n\n');
    spaces_data = struct('id',{},'coordinates',{});
    spaceId = 0;
    fig = figure('Name','Mark Parking Spaces','NumberTitle','off', ...
                 'Position',[100 80 min(round(W*1.2),1400) min(round(H*1.2),900)]);
    imshow(ref_frame_raw); hold on;
    title({'Click 4 corners, press ENTER','Answer n to finish'},'FontSize',11);
    continueLoop = true;
    while continueLoop
        try
            [xc, yc] = ginput(4);
        catch
            break;
        end
        if numel(xc) < 4, break; end
        xc = min(max(round(xc),1),W);
        yc = min(max(round(yc),1),H);
        fill(xc([1 2 3 4 1]), yc([1 2 3 4 1]), 'r', ...
             'FaceAlpha',0.15,'EdgeColor',[1 0 0],'LineWidth',1.5);
        entry.id          = spaceId;
        entry.coordinates = [xc yc];
        spaces_data(end+1) = entry; %#ok<AGROW>
        spaceId = spaceId + 1;
        ans_ = input(sprintf('Space %d saved. Add another? [y/n]: ',spaceId),'s');
        if ~strcmpi(strtrim(ans_),'y'), continueLoop = false; end
    end
    close(fig);
    save(coord_file,'spaces_data');
    fprintf('Saved %d spaces to %s\n\n', numel(spaces_data), coord_file);
end
 
function raw = loadSlice(mf, var_name, f)
    raw = squeeze(mf.(var_name)(:,:,f));
end
 
function raw = batchSlice(batch_data, local_f)
    raw = squeeze(batch_data(:,:,local_f));
end
 
function is_free = applyLaplacian(blurred, rect, mask, threshold, H, W)
    roi = genRoi(blurred, rect, mask, H, W);
    lap = del2(roi) * 4;
    [mh,mw] = size(mask);
    [rh,rw] = size(roi);
    if mh~=rh || mw~=rw
        mask = imresize(mask,[rh rw],'nearest');
    end
    is_free = mean(abs(lap(mask))) < threshold;
end
 
function is_free = applySTD(blurred, rect, mask, threshold_std, threshold_m H, W, mean_init)
    roi          = genRoi(blurred, rect, mask, H, W);
    current_mean = mean(roi(mask));
    std_ratio    = getSTD(roi(mask), mean_init) / mean_init;
    mean_drop    = (mean_init - current_mean) / mean_init;
    is_free      = std_ratio < threshold_std && mean_drop < threshold_m;
end
 
function roi = genRoi(blurred, rect, ~, H, W)
    bound_x = round(rect(1)); bound_y = round(rect(2));
    bound_w = round(rect(3)); bound_h = round(rect(4));
    c1 = max(1, bound_x);    c2 = min(W, bound_x+bound_w-1);
    r1 = max(1, bound_y);    r2 = min(H, bound_y+bound_h-1);
    if r2 < r1 || c2 < c1, roi = zeros(1); return; end
    roi = blurred(r1:r2, c1:c2);
end
 
function img = drawContours(img, coords, ~, ~, border_color, H, W)
    pts = [coords; coords(1,:)];
    for k = 1:4
        img = drawLine(img, round(pts(k,1)), round(pts(k,2)), ...
                            round(pts(k+1,1)), round(pts(k+1,2)), ...
                            border_color, 2, H, W);
    end
end
 
function img = drawLine(img, x1,y1,x2,y2, color, thickness, H, W)
    msk = lineSegMask(H,W,y1,x1,y2,x2);
    if thickness > 1, msk = imdilate(msk, strel('disk', floor(thickness/2))); end
    for ch = 1:3
        layer = img(:,:,ch); layer(msk) = color(ch); img(:,:,ch) = layer;
    end
end
 
function mask = lineSegMask(H,W,r0,c0,r1,c1)
    mask = false(H,W);
    r0=round(r0); c0=round(c0); r1=round(r1); c1=round(c1);
    dr=abs(r1-r0); dc=abs(c1-c0);
    sr=sign(r1-r0); sc=sign(c1-c0); err=dr-dc;
    while true
        if r0>=1&&r0<=H&&c0>=1&&c0<=W, mask(r0,c0)=true; end
        if r0==r1&&c0==c1, break; end
        e2=2*err;
        if e2>-dc, err=err-dc; r0=r0+sr; end
        if e2< dr, err=err+dr; c0=c0+sc; end
    end
end
 
function s_dev = getSTD(data, mean_val)
    s_dev = sqrt(sum((data - mean_val).^2) / length(data));
end
 

