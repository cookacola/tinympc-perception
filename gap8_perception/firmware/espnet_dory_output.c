#include "gap8_perception_output.h"

#include <string.h>

#define CORNER_W 40
#define CORNER_H 40
#define CORNER_C 4
#define CORNER_BYTES (CORNER_W * CORNER_H * CORNER_C)
#define GATE_BYTES (CORNER_W * CORNER_H)
#define DANGER_W 10
#define DANGER_H 10
#define GATE_OFFSET CORNER_BYTES
#define DANGER_OFFSET (CORNER_BYTES + GATE_BYTES)

static const uint8_t corner_threshold[4] = {
    GAP8_CORNER_Q_THRESHOLD_0, GAP8_CORNER_Q_THRESHOLD_1,
    GAP8_CORNER_Q_THRESHOLD_2, GAP8_CORNER_Q_THRESHOLD_3,
};

static float cross2(float ax, float ay, float bx, float by,
                    float px, float py) {
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax);
}

void gap8_decode_corner_argmax(const uint8_t *packed,
                               float corners_xy[8],
                               uint8_t confidence[4]) {
    for (int channel = 0; channel < 4; ++channel) {
        uint8_t best = 0;
        int best_x = 0, best_y = 0;
        for (int y = 0; y < CORNER_H; ++y) {
            for (int x = 0; x < CORNER_W; ++x) {
                const uint8_t value =
                    packed[(y * CORNER_W + x) * CORNER_C + channel];
                if (value > best) {
                    best = value;
                    best_x = x;
                    best_y = y;
                }
            }
        }
        corners_xy[2 * channel] = (best_x + 0.5f) * 4.0f;
        corners_xy[2 * channel + 1] = (best_y + 0.5f) * 4.0f;
        confidence[channel] = best;
    }
}

int gap8_validate_or_recover_gate(float corners[8],
                                  const uint8_t confidence[4]) {
    int confident = 0;
    int missing = -1;
    for (int corner = 0; corner < 4; ++corner) {
        if (confidence[corner] >= corner_threshold[corner]) confident++;
        else missing = corner;
    }
    if (confident < 3) return -1;
    if (confident == 3) {
        const int opposite = (missing + 2) & 3;
        const int previous = (missing + 3) & 3;
        const int following = (missing + 1) & 3;
        corners[2 * missing] = corners[2 * previous] + corners[2 * following]
                             - corners[2 * opposite];
        corners[2 * missing + 1] =
            corners[2 * previous + 1] + corners[2 * following + 1]
            - corners[2 * opposite + 1];
        if (corners[2 * missing] < 0.0f || corners[2 * missing] >= 160.0f ||
            corners[2 * missing + 1] < 0.0f ||
            corners[2 * missing + 1] >= 160.0f) return -1;
    }
    if (!(corners[0] < corners[2] && corners[6] < corners[4] &&
          corners[1] < corners[7] && corners[3] < corners[5])) return -1;
    float signed_area2 = 0.0f;
    float sign = 0.0f;
    for (int edge = 0; edge < 4; ++edge) {
        const int next = (edge + 1) & 3;
        const int after = (edge + 2) & 3;
        signed_area2 += corners[2 * edge] * corners[2 * next + 1]
                      - corners[2 * next] * corners[2 * edge + 1];
        const float side = cross2(
            corners[2 * edge], corners[2 * edge + 1],
            corners[2 * next], corners[2 * next + 1],
            corners[2 * after], corners[2 * after + 1]);
        if (edge == 0) sign = side;
        if (side * sign <= 0.0f) return -1;
    }
    const float area = signed_area2 < 0.0f
        ? -0.5f * signed_area2 : 0.5f * signed_area2;
    if (area < 128.0f || area > 23000.0f) return -1;
    const float width = 0.5f * (
        corners[2] - corners[0] + corners[4] - corners[6]);
    const float height = 0.5f * (
        corners[7] - corners[1] + corners[5] - corners[3]);
    if (width <= 0.0f || height <= 0.0f ||
        width / height < 0.35f || width / height > 2.85f) return -1;
    return missing < 0 ? 4 : missing;
}

void gap8_pool_control_maps(const uint8_t *packed,
                            uint8_t obstacle_presence[400],
                            uint8_t inverse_range[400],
                            uint8_t uncertainty[400],
                            uint8_t gate_opening[400]) {
    const uint8_t *gate = packed + GATE_OFFSET;
    const uint8_t *danger = packed + DANGER_OFFSET;
    float corners[8];
    uint8_t confidence[4];
    gap8_decode_corner_argmax(packed, corners, confidence);
    memset(inverse_range, 0, 400);
    memset(uncertainty, 0, 400);
    memset(gate_opening, 0, 400);
    for (int y = 0; y < 20; ++y) {
        for (int x = 0; x < 20; ++x) {
            obstacle_presence[y * 20 + x] =
                danger[(y / 2) * DANGER_W + x / 2]
                    >= GAP8_DANGER_Q_THRESHOLD ? 255 : 0;
        }
    }
    if (gap8_validate_or_recover_gate(corners, confidence) < 0) return;
    const float sign = cross2(
        corners[0], corners[1], corners[2], corners[3],
        corners[4], corners[5]);
    float center_x = 0.0f, center_y = 0.0f, inset[8];
    for (int corner = 0; corner < 4; ++corner) {
        center_x += 0.25f * corners[2 * corner];
        center_y += 0.25f * corners[2 * corner + 1];
    }
    for (int corner = 0; corner < 4; ++corner) {
        inset[2 * corner] = 0.86f * corners[2 * corner] + 0.14f * center_x;
        inset[2 * corner + 1] =
            0.86f * corners[2 * corner + 1] + 0.14f * center_y;
    }
    for (int y = 0; y < 20; ++y) {
        for (int x = 0; x < 20; ++x) {
            int gate_mask_confident = 1;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    if (gate[(2 * y + dy) * 40 + 2 * x + dx]
                        < GAP8_GATE_Q_THRESHOLD) gate_mask_confident = 0;
                }
            }
            if (!gate_mask_confident) continue;
            const float px = (x + 0.5f) * 8.0f;
            const float py = (y + 0.5f) * 8.0f;
            int inside = 1;
            for (int edge = 0; edge < 4; ++edge) {
                const int next = (edge + 1) & 3;
                if (cross2(
                    inset[2 * edge], inset[2 * edge + 1],
                    inset[2 * next], inset[2 * next + 1], px, py) * sign < 0.0f) {
                    inside = 0;
                    break;
                }
            }
            if (inside) gate_opening[y * 20 + x] = 255;
        }
    }
}
