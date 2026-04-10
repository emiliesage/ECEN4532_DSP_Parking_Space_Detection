/*
 * parking_detection.c
 *
 * Parking space occupancy detector using OpenCV.
 * Converted from MATLAB implementation.
 *
 * Build:
 *   gcc parking_detection.c -o parking_detection \
 *       $(pkg-config --cflags --libs opencv4) -lm
 *
 * Usage:
 *   ./parking_detection <input_video> [coords_file]
 *
 *   If coords_file is omitted, the program opens the first frame and lets
 *   you click 4 corners per space (press ENTER after each space, 'q' when done).
 *   Coordinates are saved to parking_spaces.txt for reuse.
 *
 * Dependencies: OpenCV 4.x, standard C99
 */

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <unistd.h>

/* ── tuneable constants ───────────────────────────────────────────────── */
#define STD_THRESHOLD       0.24    /* coefficient-of-variation threshold  */
#define MEAN_DROP_THRESHOLD 0.25    /* fraction drop → occupied            */
#define DETECT_DELAY_SEC    1.0     /* debounce: seconds before flip       */
#define GAUSS_SIGMA         3.0     /* Gaussian blur sigma                 */
#define GAUSS_KSIZE         7       /* kernel size (must be odd)           */
#define MAX_SPACES          32      /* max parking spaces supported        */
#define COORD_FILE_DEFAULT  "parking_spaces.txt"

/* ── colour constants (BGR) ──────────────────────────────────────────── */
static const cv::Scalar COLOR_GREEN (  0, 255,   0);
static const cv::Scalar COLOR_RED   (  0,   0, 255);
static const cv::Scalar COLOR_WHITE (255, 255, 255);
static const cv::Scalar COLOR_BLACK (  0,   0,   0);

/* ══════════════════════════════════════════════════════════════════════ */
/*  Data structures                                                       */
/* ══════════════════════════════════════════════════════════════════════ */

typedef struct {
    cv::Point corners[4];   /* pixel coordinates of the 4 corners        */
    cv::Mat   mask;         /* binary mask (same size as bounding rect)   */
    cv::Rect  bbox;         /* bounding rectangle                         */
} ParkingSpace;

/* ══════════════════════════════════════════════════════════════════════ */
/*  Maths helpers                                                         */
/* ══════════════════════════════════════════════════════════════════════ */

/* Population standard deviation of all non-zero-mask pixels in roi      */
static double computeSTD(const cv::Mat &roi, const cv::Mat &mask, double mean_val)
{
    double sum_sq = 0.0;
    int    count  = 0;
    for (int r = 0; r < roi.rows; ++r)
        for (int c = 0; c < roi.cols; ++c)
            if (mask.at<uchar>(r, c)) {
                double d = roi.at<double>(r, c) - mean_val;
                sum_sq  += d * d;
                ++count;
            }
    return (count > 0) ? sqrt(sum_sq / count) : 0.0;
}

/* Mean of masked pixels                                                  */
static double computeMean(const cv::Mat &roi, const cv::Mat &mask)
{
    double sum   = 0.0;
    int    count = 0;
    for (int r = 0; r < roi.rows; ++r)
        for (int c = 0; c < roi.cols; ++c)
            if (mask.at<uchar>(r, c)) {
                sum += roi.at<double>(r, c);
                ++count;
            }
    return (count > 0) ? sum / count : 0.0;
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Detection                                                             */
/* ══════════════════════════════════════════════════════════════════════ */

/*
 * Returns true  → space is FREE
 *         false → space is OCCUPIED
 *
 * Criteria (both must hold for FREE):
 *   1. coefficient of variation  < STD_THRESHOLD
 *   2. mean has NOT dropped by more than MEAN_DROP_THRESHOLD vs reference
 */
static bool applySTD(const cv::Mat &blurred,
                     const ParkingSpace &space,
                     double mean_ref)
{
    /* clip roi to image bounds */
    cv::Rect safe = space.bbox & cv::Rect(0, 0, blurred.cols, blurred.rows);
    if (safe.width <= 0 || safe.height <= 0)
        return false;

    cv::Mat roi_full = blurred(safe);

    /* the mask may be larger than safe rect if bbox was clipped – resize */
    cv::Mat mask_use = space.mask;
    if (mask_use.size() != roi_full.size())
        cv::resize(mask_use, mask_use, roi_full.size(), 0, 0, cv::INTER_NEAREST);

    double cur_mean = computeMean(roi_full, mask_use);
    if (cur_mean < 1e-6) return false;          /* avoid div-by-zero      */

    double std_val   = computeSTD(roi_full, mask_use, cur_mean);
    double std_ratio = std_val / cur_mean;
    double mean_drop = (mean_ref - cur_mean) / mean_ref;

    return (std_ratio < STD_THRESHOLD) && (mean_drop < MEAN_DROP_THRESHOLD);
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Coordinate I/O                                                        */
/* ══════════════════════════════════════════════════════════════════════ */

static bool saveCoords(const char *path,
                       const ParkingSpace *spaces, int n)
{
    FILE *fp = fopen(path, "w");
    if (!fp) { perror(path); return false; }
    fprintf(fp, "%d\n", n);
    for (int s = 0; s < n; ++s)
        for (int k = 0; k < 4; ++k)
            fprintf(fp, "%d %d\n", spaces[s].corners[k].x,
                                   spaces[s].corners[k].y);
    fclose(fp);
    printf("Saved %d spaces to %s\n", n, path);
    return true;
}

static int loadCoords(const char *path, ParkingSpace *spaces, int max_spaces)
{
    FILE *fp = fopen(path, "r");
    if (!fp) return 0;
    int n = 0;
    if (fscanf(fp, "%d", &n) != 1 || n <= 0 || n > max_spaces) {
        fclose(fp); return 0;
    }
    for (int s = 0; s < n; ++s)
        for (int k = 0; k < 4; ++k)
            if (fscanf(fp, "%d %d",
                       &spaces[s].corners[k].x,
                       &spaces[s].corners[k].y) != 2) {
                fclose(fp); return 0;
            }
    fclose(fp);
    printf("Loaded %d spaces from %s\n", n, path);
    return n;
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Build mask + bbox for a space from its 4 corners                     */
/* ══════════════════════════════════════════════════════════════════════ */

static void buildSpaceMask(ParkingSpace &sp)
{
    int x1 = sp.corners[0].x, x2 = sp.corners[0].x;
    int y1 = sp.corners[0].y, y2 = sp.corners[0].y;
    for (int k = 1; k < 4; ++k) {
        x1 = std::min(x1, sp.corners[k].x);
        x2 = std::max(x2, sp.corners[k].x);
        y1 = std::min(y1, sp.corners[k].y);
        y2 = std::max(y2, sp.corners[k].y);
    }
    sp.bbox = cv::Rect(x1, y1, x2 - x1 + 1, y2 - y1 + 1);

    /* build polygon mask in local coordinates */
    cv::Point local[4];
    for (int k = 0; k < 4; ++k)
        local[k] = cv::Point(sp.corners[k].x - x1,
                             sp.corners[k].y - y1);

    sp.mask = cv::Mat::zeros(sp.bbox.height, sp.bbox.width, CV_8U);
    const cv::Point *pts = local;
    int              npt = 4;
    cv::fillPoly(sp.mask, &pts, &npt, 1, cv::Scalar(255));
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Interactive coordinate picker                                         */
/* ══════════════════════════════════════════════════════════════════════ */

struct PickState {
    cv::Mat          display;
    std::vector<cv::Point> current;
    ParkingSpace    *spaces;
    int             *n_spaces;
};

static void onMouse(int event, int x, int y, int /*flags*/, void *userdata)
{
    PickState *ps = (PickState *)userdata;
    if (event != cv::EVENT_LBUTTONDOWN) return;

    ps->current.push_back(cv::Point(x, y));
    cv::circle(ps->display, cv::Point(x, y), 4, COLOR_RED, -1);

    int n = (int)ps->current.size();
    if (n > 1)
        cv::line(ps->display, ps->current[n-2], ps->current[n-1],
                 COLOR_RED, 2);
    if (n == 4) {
        cv::line(ps->display, ps->current[3], ps->current[0],
                 COLOR_RED, 2);

        /* fill with semi-transparent overlay */
        cv::Mat overlay = ps->display.clone();
        const cv::Point *pts = ps->current.data();
        int npt = 4;
        cv::fillPoly(overlay, &pts, &npt, 1, cv::Scalar(0, 0, 180));
        cv::addWeighted(overlay, 0.2, ps->display, 0.8, 0, ps->display);
    }
    cv::imshow("Mark Parking Spaces", ps->display);
}

static int coordinatesGenerator(const cv::Mat &ref_frame,
                                 ParkingSpace *spaces,
                                 const char   *coord_file)
{
    printf("\nCoordinate Generator\n");
    printf("Click 4 corners of each space. Press ENTER to confirm, 'q' to finish.\n\n");

    PickState ps;
    cv::cvtColor(ref_frame, ps.display, cv::COLOR_GRAY2BGR);
    ps.spaces   = spaces;
    ps.n_spaces = new int(0);

    cv::namedWindow("Mark Parking Spaces", cv::WINDOW_AUTOSIZE);
    cv::setMouseCallback("Mark Parking Spaces", onMouse, &ps);
    cv::imshow("Mark Parking Spaces", ps.display);

    int n = 0;
    while (n < MAX_SPACES) {
        ps.current.clear();
        printf("Space %d: click 4 corners, then press ENTER (or 'q' to quit).\n", n+1);

        /* wait until 4 clicks collected */
        while ((int)ps.current.size() < 4) {
            int key = cv::waitKey(50);
            if (key == 'q' || key == 27) goto done;
        }

        /* confirm with ENTER */
        printf("  Press ENTER to confirm space %d, 'r' to redo, 'q' to quit.\n", n+1);
        for (;;) {
            int key = cv::waitKey(0);
            if (key == 13 || key == 10) break;   /* ENTER */
            if (key == 'r') { ps.current.clear(); break; }
            if (key == 'q' || key == 27) goto done;
        }
        if ((int)ps.current.size() < 4) continue;  /* redo */

        for (int k = 0; k < 4; ++k)
            spaces[n].corners[k] = ps.current[k];
        buildSpaceMask(spaces[n]);
        printf("  Space %d saved.\n", n+1);
        ++n;

        printf("Add another space? (ENTER=yes / 'q'=done): ");
        int key = cv::waitKey(0);
        if (key == 'q' || key == 27) break;
    }

done:
    cv::destroyWindow("Mark Parking Spaces");
    delete ps.n_spaces;
    saveCoords(coord_file, spaces, n);
    return n;
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  Annotation helpers                                                    */
/* ══════════════════════════════════════════════════════════════════════ */

static void drawSpaceContour(cv::Mat &frame,
                              const ParkingSpace &sp,
                              const cv::Scalar &color)
{
    for (int k = 0; k < 4; ++k)
        cv::line(frame,
                 sp.corners[k],
                 sp.corners[(k+1) % 4],
                 color, 2, cv::LINE_AA);
}

/* ══════════════════════════════════════════════════════════════════════ */
/*  main                                                                  */
/* ══════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <input_video> [coords_file]\n", argv[0]);
        return 1;
    }
    const char *video_path = argv[1];
    const char *coord_file = (argc >= 3) ? argv[2] : COORD_FILE_DEFAULT;

    /* ── open video ─────────────────────────────────────────────────── */
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        fprintf(stderr, "Cannot open video: %s\n", video_path);
        return 1;
    }
    int    frame_w   = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int    frame_h   = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    int    n_total   = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    double fps       = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 15.0;

    printf("Video: %s  (%dx%d, %.1f fps, %d frames)\n",
           video_path, frame_w, frame_h, fps, n_total);

    /* ── output video ───────────────────────────────────────────────── */
    cv::VideoWriter vw("parking_detected.avi",
                       cv::VideoWriter::fourcc('M','J','P','G'),
                       fps,
                       cv::Size(frame_w, frame_h));
    if (!vw.isOpened()) {
        fprintf(stderr, "Cannot open output video writer.\n");
        return 1;
    }

    /* ── parking spaces ─────────────────────────────────────────────── */
    ParkingSpace spaces[MAX_SPACES];
    int n_spaces = 0;

    /* try loading existing coords */
    if (access(coord_file, F_OK) == 0) {
        printf("Found %s. Reuse? [y/n]: ", coord_file);
        char ans[8] = {0};
        if (fgets(ans, sizeof(ans), stdin) && (ans[0]=='y'||ans[0]=='Y')) {
            n_spaces = loadCoords(coord_file, spaces, MAX_SPACES);
            if (n_spaces > 0)
                for (int s = 0; s < n_spaces; ++s)
                    buildSpaceMask(spaces[s]);
        }
    }

    if (n_spaces == 0) {
        cv::Mat first;
        cap >> first;
        if (first.empty()) { fprintf(stderr, "Cannot read first frame.\n"); return 1; }
        cv::Mat gray;
        if (first.channels() == 3)
            cv::cvtColor(first, gray, cv::COLOR_BGR2GRAY);
        else
            gray = first;
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);   /* rewind */

        n_spaces = coordinatesGenerator(gray, spaces, coord_file);
        if (n_spaces == 0) {
            fprintf(stderr, "No spaces defined. Exiting.\n");
            return 1;
        }
    }
    printf("Using %d parking spaces.\n", n_spaces);

    /* ── initialise reference means from frame 0 ────────────────────── */
    double mean_init[MAX_SPACES] = {0};
    {
        cv::Mat f0_raw;
        cap >> f0_raw;
        cap.set(cv::CAP_PROP_POS_FRAMES, 0);

        cv::Mat f0_gray, f0_blur, f0_d;
        if (f0_raw.channels() == 3)
            cv::cvtColor(f0_raw, f0_gray, cv::COLOR_BGR2GRAY);
        else
            f0_gray = f0_raw;
        cv::GaussianBlur(f0_gray, f0_blur,
                         cv::Size(GAUSS_KSIZE, GAUSS_KSIZE), GAUSS_SIGMA);
        f0_blur.convertTo(f0_d, CV_64F);

        for (int s = 0; s < n_spaces; ++s) {
            cv::Rect safe = spaces[s].bbox &
                            cv::Rect(0, 0, f0_d.cols, f0_d.rows);
            if (safe.width <= 0 || safe.height <= 0) continue;
            cv::Mat roi = f0_d(safe);
            cv::Mat msk = spaces[s].mask;
            if (msk.size() != roi.size())
                cv::resize(msk, msk, roi.size(), 0, 0, cv::INTER_NEAREST);
            mean_init[s] = computeMean(roi, msk);
        }
    }

    /* ── detection state ────────────────────────────────────────────── */
    bool   statuses[MAX_SPACES]     = {false};
    double pending_time[MAX_SPACES];
    for (int s = 0; s < n_spaces; ++s)
        pending_time[s] = -1.0;   /* -1 = no pending change */

    /* ── main loop ──────────────────────────────────────────────────── */
    cv::Mat frame_raw, frame_gray, frame_blur, frame_d;
    int global_f = 0;

    printf("\nProcessing %d frames...\n", n_total);

    while (true) {
        cap >> frame_raw;
        if (frame_raw.empty()) break;
        ++global_f;

        /* grayscale + blur */
        if (frame_raw.channels() == 3)
            cv::cvtColor(frame_raw, frame_gray, cv::COLOR_BGR2GRAY);
        else
            frame_gray = frame_raw;
        cv::GaussianBlur(frame_gray, frame_blur,
                         cv::Size(GAUSS_KSIZE, GAUSS_KSIZE), GAUSS_SIGMA);
        frame_blur.convertTo(frame_d, CV_64F);

        double pos_sec = (double)global_f / fps;

        /* ── evaluate each space ──────────────────────────────────── */
        bool new_statuses[MAX_SPACES];
        for (int s = 0; s < n_spaces; ++s) {
            new_statuses[s] = applySTD(frame_d, spaces[s], mean_init[s]);

            /* update reference only when free */
            if (new_statuses[s]) {
                cv::Rect safe = spaces[s].bbox &
                                cv::Rect(0, 0, frame_d.cols, frame_d.rows);
                if (safe.width > 0 && safe.height > 0) {
                    cv::Mat roi = frame_d(safe);
                    cv::Mat msk = spaces[s].mask;
                    if (msk.size() != roi.size())
                        cv::resize(msk, msk, roi.size(), 0, 0, cv::INTER_NEAREST);
                    mean_init[s] = computeMean(roi, msk);
                }
            }
        }

        /* ── debounce ─────────────────────────────────────────────── */
        for (int s = 0; s < n_spaces; ++s) {
            bool st = new_statuses[s];

            if (pending_time[s] >= 0.0 && st == statuses[s]) {
                pending_time[s] = -1.0;
                continue;
            }
            if (pending_time[s] >= 0.0 && st != statuses[s]) {
                if (pos_sec - pending_time[s] >= DETECT_DELAY_SEC) {
                    statuses[s]     = st;
                    pending_time[s] = -1.0;
                }
                continue;
            }
            if (pending_time[s] < 0.0 && st != statuses[s])
                pending_time[s] = pos_sec;
        }

        /* ── annotate & write ─────────────────────────────────────── */
        cv::Mat annotated;
        cv::cvtColor(frame_gray, annotated, cv::COLOR_GRAY2BGR);

        int n_free = 0;
        for (int s = 0; s < n_spaces; ++s) {
            drawSpaceContour(annotated, spaces[s],
                             statuses[s] ? COLOR_GREEN : COLOR_RED);
            if (statuses[s]) ++n_free;
        }

        /* HUD overlay */
        char hud[128];
        snprintf(hud, sizeof(hud), "Frame %d/%d  |  Free: %d/%d",
                 global_f, n_total, n_free, n_spaces);
        cv::putText(annotated, hud, cv::Point(8, 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BLACK, 3, cv::LINE_AA);
        cv::putText(annotated, hud, cv::Point(8, 22),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 1, cv::LINE_AA);

        vw.write(annotated);

        if (global_f % 100 == 0)
            printf("  Frame %d/%d  free: %d/%d\n",
                   global_f, n_total, n_free, n_spaces);
    }

    cap.release();
    vw.release();
    printf("\nDone. Output: parking_detected.avi\n");
    return 0;
}
