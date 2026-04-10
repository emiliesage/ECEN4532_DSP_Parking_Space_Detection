close all; clear all; clc;

%{
    Car Detection Algorithm uses Laplace Transfom to find variation
    in pxels in a predefined parking space. The spaces are 
    Pre-Defined and a Gaussian Blur is applied to the mask
    of the parking space. The data in the video file is gathered in
    chunks as to preserve memory. The detection is then saved to a video 
    timelapse where the detection is registered.

    Credit
    Certain functions deasling with graphics in line drawing and annotation
    The methododlogy for this project was taken from 
%}

%% Define Cosntants

LAPLACIAN_THRESHOLD = 0.9;   % threshold determined through expirementation
DETECT_DELAY_SEC    = 1.0;    
FPS                 = 15;     
GAUSS_SIGMA         = 3;      
BATCH_SIZE          = 50;   
MAT_FILE            = 'output_emilie.mat';
COORD_FILE          = 'parking_spaces.mat';

COLOR_GREEN = uint8([0   255 0  ]);   % free space
COLOR_BLUE  = uint8([0   0   255]);   % occupied
COLOR_WHITE = uint8([255 255 255]);   % label text
COLOR_RED   = uint8([255 0   0  ]);   % coordinate overlay

%% Open file 
fprintf("Opening %s, not loading into memory...",MAT_FILE);
mf = matfile(MAT_FILE);

% gather data on owner and size of the file from matfile()
whos_info = whos(mf);
video_name = '';
for i = 1:numel(whos_info)
    
    video_name = whos_info(i).name;
    fullSize  = whos_info(i).size;
    fprintf('  Found "%s"  full size: %s\n', video_name, mat2str(fullSize));
        
end
if isempty(video_name)
    error('No 3-D/4-D numeric variable found in %s', MAT_FILE);
end

dimensions = numel(fullSize);
image_height  = fullSize(1);
image_width  = fullSize(2);

n_total  = fullSize(3);


%% Gather or generate coordinate data for parking spaces

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
    space_data  = coordinatesGenerator(ret_frame_raw, image_height, image_width, COLOR_BLUE, COORD_FILE);
end

n_spaces = numel(space_data);
if n_spaces == 0
    error('No spaces defined. Re-run and mark at least one space.');
end
fprintf('Using %d parking spaces over %d frames.\n', n_spaces, n_total);

fprintf('Building space masks...\n');

bounds = zeros(n_spaces, 4);  
masks  = cell(n_spaces, 1);
mu  = zeros(n_spaces, n_total);
sigma = zeros(n_spaces, n_total);
space_changes = zeros(n_spaces,n_total);

for s = 1:n_spaces
    coords = space_data(s).coordinates;   
    x1=min(coords(:,1)); x2=max(coords(:,1));
    y1=min(coords(:,2)); y2=max(coords(:,2));
    bound_x=x1; bound_y=y1; bound_w=x2-x1+1; bound_h=y2-y1+1;
    bounds(s,:) = [bound_x bound_y bound_w bound_h];

    local_coords = coords;
    local_coords(:,1) = coords(:,1) - bound_x + 1;
    local_coords(:,2) = coords(:,2) - bound_y + 1;
    masks{s} = poly2mask(local_coords(:,1), local_coords(:,2), bound_h, bound_w);
end


%% Itterate through the file in batches


%{
    itterate through batches of files, find ones containing filled spaces
    and anotate video, creating a video vile that shows the change over
    time
%}

fprintf('\nProcessing %d frames in batches of %d...\n', n_total, BATCH_SIZE);


statuses    = false(n_spaces, 1);
pending_time = nan(n_spaces, 1);


v_writer = VideoWriter('parking_detected.avi','Motion JPEG AVI');
v_writer.FrameRate = FPS;
open(v_writer);

total_batches   = ceil(n_total / BATCH_SIZE);

for batch_id = 1:total_batches

    f_start = (batch_id-1)*BATCH_SIZE + 1;
    f_end   = min(batch_id*BATCH_SIZE, n_total);
    n_batch = f_end - f_start + 1;

    fprintf('  Batch %d/%d  (frames %d–%d)...\n',batch_id, total_batches, f_start, f_end);       
    batch_data = mf.(video_name)(:,:,f_start:f_end);


    for local_f = 1:n_batch
        global_f = f_start + local_f - 1;

        frame = batchSlice(batch_data, local_f);

        blurred = imgaussfilt(double(frame), GAUSS_SIGMA);

        positionSec = global_f / FPS;

        new_statuses = false(n_spaces,1);
        for s = 1:n_spaces
            new_statuses(s) = applyLaplacian(blurred, bounds(s,:), masks{s}, ...
                                            LAPLACIAN_THRESHOLD, image_height, image_width);
            roi = genRoi(blurred,bounds(s,:),masks{s},image_height,image_width);
            mu(s, global_f) = mean(roi(masks{s}));  
            sigma(s,global_f) = std(roi(masks{s}));
            space_changes(s,global_f) = ~new_statuses(s);

            
        end


        for s = 1:n_spaces
            st = new_statuses(s);
            if ~isnan(pending_time(s)) && st == statuses(s)
                pending_time(s) = nan;
                continue;
            end
            if ~isnan(pending_time(s)) && st ~= statuses(s)
                if positionSec - pending_time(s) >= DETECT_DELAY_SEC
                    statuses(s)    = st;
                    pending_time(s) = nan;
                end
                continue;
            end
            if isnan(pending_time(s)) && st ~= statuses(s)
                pending_time(s) = positionSec;
            end
        end

        annotated = repmat(frame, [1 1 3]); 
        for s = 1:n_spaces
            clr = COLOR_GREEN;
            if ~statuses(s), clr = COLOR_RED; end
            annotated = drawContours(annotated, space_data(s).coordinates, ...
                                     num2str(space_data(s).id+1), ...
                                     COLOR_WHITE, clr, image_height, image_width);
        end


        if exist('insertText','file')
            nFree = sum(statuses);
            annotated = insertText(annotated, [5 5], ...
                sprintf('Frame %d/%d  |  Free: %d/%d', global_f, nTotal, nFree, nSpaces), ...
                'FontSize',13,'BoxColor','black','BoxOpacity',0.5,'TextColor','white');
        end

        writeVideo(v_writer, annotated);
    end


    clear batch_data;
    fprintf('    done.  Free spaces: %d/%d\n', sum(statuses), n_spaces);
end

% Design a low-pass filter (e.g. cutoff ~0.05 of Nyquist)
lpFilt = designfilt('lowpassfir', ...
    'PassbandFrequency',  0.01, ...
    'StopbandFrequency',  0.05, ...
    'PassbandRipple',     1, ...
    'StopbandAttenuation',60);
sigma_hp = zeros(size(sigma));
mu_hp = zeros(size(mu));
for s = 1:n_spaces
    lp = filtfilt(lpFilt, mu(s,:));   % zero-phase low-pass
    mu_hp(s,:) = lp;  % subtract to get high-pass
    lp = filtfilt(lpFilt, sigma(s,:));   % zero-phase low-pass
    sigma_hp(s,:) = lp;  % subtract to get high-pass
end

close(v_writer);
fprintf('\nVideo saved: parking_detected.avi\n');
figure;
subplot(3,2,1);
plot(mu_hp(1,:));
subplot(3,2,2);
plot(mu_hp(2,:));
subplot(3,2,3);
plot(mu_hp(3,:));
subplot(3,2,4);
plot(mu_hp(4,:));
subplot(3,2,5);
plot(mu_hp(5,:));
% 
% figure;
% subplot(3,2,1);
% plot(sigma_hp(1,:));
% subplot(3,2,2);
% plot(sigma_hp(2,:));
% subplot(3,2,3);
% plot(sigma_hp(3,:));
% subplot(3,2,4);
% plot(sigma_hp(4,:));
% subplot(3,2,5);
% plot(sigma_hp(5,:));
% 
% figure;
% subplot(3,2,1);
% plot(mu(1,:));
% subplot(3,2,2);
% plot(mu(2,:));
% subplot(3,2,3);
% plot(mu(3,:));
% subplot(3,2,4);
% plot(mu(4,:));
% subplot(3,2,5);
% plot(mu(5,:));
% 
% figure;
% subplot(3,2,1);
% plot(sigma(1,:));
% subplot(3,2,2);
% plot(sigma(2,:));
% subplot(3,2,3);
% plot(sigma(3,:));
% subplot(3,2,4);
% plot(sigma(4,:));
% subplot(3,2,5);
% plot(sigma(5,:));
% 
% figure;
% subplot(3,2,1);
% plot(space_changes(1,:));
% subplot(3,2,2);
% plot(space_changes(2,:));
% subplot(3,2,3);
% plot(space_changes(3,:));
% subplot(3,2,4);
% plot(space_changes(4,:));
% subplot(3,2,5);
% plot(space_changes(5,:));




%% Functions

%{  
    coordinatesGenerator generates coordinates based on user input.
    the user draws the lines of the parking spaces.
    Unable to automatically generate them yet
%}
function spaces_data = coordinatesGenerator(ref_frame_raw, H, W, ~, coord_file)

    fprintf('\nCoordinates Generator\n');
    fprintf('For each parking space: click its 4 corners, then press ENTER.\n');
    fprintf('Answer "n" when you have marked all spaces.\n\n');

    spaces_data = struct('id',{},'coordinates',{});
    spaceId    = 0;

    fig = figure('Name','Mark Parking Spaces – 4 clicks then ENTER per space', ...
                 'NumberTitle','off', ...
                 'Position',[100 80 min(round(W*1.2),1400) min(round(H*1.2),900)]);
    imshow(ref_frame_raw);
    hold on;
    title({'Click 4 corners of a space, then press ENTER', ...
           'Answer "n" in Command Window when finished'}, 'FontSize',11);

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

        % Draw filled semi-transparent quad + red border
        fill(xc([1 2 3 4 1]), yc([1 2 3 4 1]), 'r', ...
             'FaceAlpha',0.15,'EdgeColor',[1 0 0],'LineWidth',1.5)

        entry.id          = spaceId;
        entry.coordinates = [xc yc];   % 4x2
        spaces_data(end+1) = entry;      %#ok<AGROW>
        spaceId = spaceId + 1;

        fprintf('  Space %d: [%d,%d] [%d,%d] [%d,%d] [%d,%d]\n', ...
            spaceId, xc(1),yc(1),xc(2),yc(2),xc(3),yc(3),xc(4),yc(4));

        ans_ = input(sprintf('Space %d saved. Add another space? [y/n]: ',spaceId),'s');
        if ~strcmpi(strtrim(ans_),'y')
            continueLoop = false;
        end
    end

    close(fig);
    save(coord_file,'spaces_data');
    fprintf('Saved %d spaces to %s\n\n', numel(spaces_data), coord_file);
end

% function for loading a single frame from the .mat file
function raw = loadSlice(mf, var_name, f)  
        raw = squeeze(mf.(var_name)(:,:,f));
end

function raw = batchSlice(batch_data, local_f)
        raw = squeeze(batch_data(:,:,local_f));
end

function is_free = applyLaplacian(blurred, rect, mask, threshold, H, W)

    roi = genRoi(blurred,rect,mask,H,W);
    lap = del2(roi) * 4;

    [mh,mw] = size(mask);
    [rh,rw] = size(roi);
    if mh~=rh || mw~=rw
        mask = imresize(mask,[rh rw],'nearest');
    end

    is_free = mean(abs(lap(mask))) < threshold;
end

function roi = genRoi(blurred, rect, msk,H,W)
    bound_x=round(rect(1)); bound_y=round(rect(2));
    bound_w=round(rect(3)); bound_h=round(rect(4));
    column_1=max(1,bound_x);      column_2=min(W,bound_x+bound_w-1);
    row_1=max(1,bound_y);      row_2=min(H,bound_y+bound_h-1);
    if row_2<row_1 || column_2<column_1, is_free=false; return; end

    roi = blurred(row_1:row_2, column_1:column_2);
end

function img = drawContours(img, coords, label, font_color, border_color, H, W)

    pts = [coords; coords(1,:)];
    for k = 1:4
        img = drawLine(img, round(pts(k,1)),round(pts(k,2)), ...
                            round(pts(k+1,1)),round(pts(k+1,2)), ...
                            border_color, 2, H, W);
    end
    
end

function img = drawLine(img, x1,y1,x2,y2, color, thickness, H, W)
    msk = lineSegMask(H,W,y1,x1,y2,x2);
    if thickness>1, msk=imdilate(msk,strel('disk',floor(thickness/2))); end
        for ch=1:3
        layer=img(:,:,ch); layer(msk)=color(ch); img(:,:,ch)=layer;
    end
end

function mask = lineSegMask(H,W,r0,c0,r1,c1)
    mask=false(H,W);
    r0=round(r0);c0=round(c0);r1=round(r1);c1=round(c1);
    dr=abs(r1-r0);dc=abs(c1-c0);
    sr=sign(r1-r0);sc=sign(c1-c0);err=dr-dc;
    while true
        if r0>=1&&r0<=H&&c0>=1&&c0<=W, mask(r0,c0)=true; end
        if r0==r1&&c0==c1, break; end
        e2=2*err;
        if e2>-dc, err=err-dc;r0=r0+sr; end
        if e2< dr, err=err+dr;c0=c0+sc; end
    end
end

function s_dev = getSTD(data, mean)
    
s_dev = sqrt(sum((data - myMean).^2) / length(data));

end