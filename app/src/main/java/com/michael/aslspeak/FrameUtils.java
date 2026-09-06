package com.michael.aslspeak;

/**
 * Pixel-format plumbing only (NV21 -> RGB + nearest-neighbor downscale to the model's
 * fixed input size). This is not a detection/segmentation stage — every pixel is just
 * reshaped so the classifier can read it; the model does all the actual interpretation.
 */
public final class FrameUtils {

    private FrameUtils() {}

    public static byte[] nv21ToRgbSquare(byte[] nv21, int width, int height, int outSize) {
        byte[] out = new byte[outSize * outSize * 3];

        // center-crop to a square region of the source frame, then nearest-neighbor
        // sample that square down to outSize x outSize.
        int cropSize = Math.min(width, height);
        int cropX = (width - cropSize) / 2;
        int cropY = (height - cropSize) / 2;

        for (int oy = 0; oy < outSize; oy++) {
            int sy = cropY + (oy * cropSize) / outSize;
            for (int ox = 0; ox < outSize; ox++) {
                int sx = cropX + (ox * cropSize) / outSize;

                int yIndex = sy * width + sx;
                int y = nv21[yIndex] & 0xFF;

                int uvRow = sy / 2;
                int uvCol = sx / 2;
                int uvIndex = width * height + uvRow * width + uvCol * 2;
                int v = nv21[uvIndex] & 0xFF;
                int u = nv21[uvIndex + 1] & 0xFF;

                int r = (int) (y + 1.370705f * (v - 128));
                int g = (int) (y - 0.337633f * (u - 128) - 0.698001f * (v - 128));
                int b = (int) (y + 1.732446f * (u - 128));

                r = clamp(r); g = clamp(g); b = clamp(b);

                int outIdx = (oy * outSize + ox) * 3;
                out[outIdx] = (byte) r;
                out[outIdx + 1] = (byte) g;
                out[outIdx + 2] = (byte) b;
            }
        }
        return out;
    }

    private static int clamp(int v) {
        if (v < 0) return 0;
        if (v > 255) return 255;
        return v;
    }
}
