#ifndef BOX_H
#define BOX_H

typedef struct{
    float x, y, w, h;
} box;

// from: box.h
typedef struct detection {
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;

typedef struct{
    float dx, dy, dw, dh;
} dbox;

box float_to_box(float *f);
float box_iou(box a, box b);
float box_rmse(box a, box b);
dbox diou(box a, box b);
void do_nms(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort_v2(box *boxes, float **probs, int total, int classes, float thresh);
void do_nms_sort(detection *dets, int total, int classes, float thresh); // v3
box decode_box(box b, box anchor);
box encode_box(box b, box anchor);

#endif
