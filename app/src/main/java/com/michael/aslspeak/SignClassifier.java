package com.michael.aslspeak;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Thin wrapper around a tiny quantized TFLite model. Pure ML classification only —
 * no hand-crafted detection/segmentation stage. The model is expected to take the
 * whole (downscaled) frame and output a softmax over LABELS, including a "blank"
 * class for "no recognizable sign in frame" since there's no separate detector.
 */
public class SignClassifier {

    // A-Z fingerspelling alphabet + blank ("no sign") + space (end-of-word gesture).
    public static final String[] LABELS = {
            "A","B","C","D","E","F","G","H","I","J","K","L","M",
            "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
            "SPACE","BLANK"
    };

    public static final int INPUT_SIZE = 96; // small input keeps CPU cost low on Note 2
    private static final int PIXEL_SIZE = 3; // RGB
    private static final int QUANT_ZERO_POINT = 0; // adjust to match trained model's quant params
    private static final float QUANT_SCALE = 1f / 255f;

    private final Interpreter interpreter;
    private final ByteBuffer inputBuffer;
    private final byte[][] outputBuffer;

    public SignClassifier(AssetManager assets, String modelFile) throws IOException {
        MappedByteBuffer modelBuffer = loadModelFile(assets, modelFile);

        Interpreter.Options options = new Interpreter.Options();
        // Note 2 is a quad-core Exynos 4412; leave one core free for camera/UI work.
        options.setNumThreads(3);
        options.setUseXNNPACK(true);

        interpreter = new Interpreter(modelBuffer, options);

        inputBuffer = ByteBuffer.allocateDirect(INPUT_SIZE * INPUT_SIZE * PIXEL_SIZE);
        inputBuffer.order(ByteOrder.nativeOrder());

        outputBuffer = new byte[1][LABELS.length];
    }

    private static MappedByteBuffer loadModelFile(AssetManager assets, String modelFile) throws IOException {
        AssetFileDescriptor fd = assets.openFd(modelFile);
        FileInputStream inputStream = new FileInputStream(fd.getFileDescriptor());
        FileChannel channel = inputStream.getChannel();
        long startOffset = fd.getStartOffset();
        long declaredLength = fd.getDeclaredLength();
        return channel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    /** rgb: INPUT_SIZE*INPUT_SIZE*3 bytes, row-major, already resized/cropped by caller. */
    public Result classify(byte[] rgb) {
        inputBuffer.rewind();
        inputBuffer.put(rgb);

        interpreter.run(inputBuffer, outputBuffer);

        int bestIdx = 0;
        int bestScore = Integer.MIN_VALUE;
        for (int i = 0; i < LABELS.length; i++) {
            int score = outputBuffer[0][i] & 0xFF; // uint8 quantized score
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }

        float confidence = bestScore / 255f;
        return new Result(LABELS[bestIdx], confidence);
    }

    public void close() {
        interpreter.close();
    }

    public static class Result {
        public final String label;
        public final float confidence;

        Result(String label, float confidence) {
            this.label = label;
            this.confidence = confidence;
        }
    }
}
