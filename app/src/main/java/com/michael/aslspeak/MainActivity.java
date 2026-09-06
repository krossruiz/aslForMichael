package com.michael.aslspeak;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

public class MainActivity extends AppCompatActivity implements SurfaceHolder.Callback {

    private static final String TAG = "ASLSpeak";
    private static final int REQ_CAMERA = 100;
    private static final String MODEL_FILE = "asl_alphabet.tflite";

    // Inference cadence: the Note 2's Exynos 4412 can't sustain 30fps CNN inference,
    // so we deliberately throttle instead of processing every preview frame.
    private static final long INFERENCE_INTERVAL_MS = 200; // ~5 inferences/sec
    private static final float CONFIDENCE_THRESHOLD = 0.75f;
    private static final int STABLE_FRAMES_REQUIRED = 4; // debounce: same label N times before committing

    private SurfaceView surfaceView;
    private TextView predictedLetterView;
    private TextView spokenBufferView;

    private Camera camera;
    private SignClassifier classifier;
    private TextToSpeech tts;

    private HandlerThread inferenceThread;
    private Handler inferenceHandler;
    private final Handler mainHandler = new Handler();

    private volatile long lastInferenceTime = 0;
    private String lastLabel = "";
    private int stableCount = 0;
    private final StringBuilder wordBuffer = new StringBuilder();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        surfaceView = findViewById(R.id.camera_preview);
        predictedLetterView = findViewById(R.id.predicted_letter);
        spokenBufferView = findViewById(R.id.spoken_buffer);

        surfaceView.getHolder().addCallback(this);

        tts = new TextToSpeech(this, status -> {
            if (status == TextToSpeech.SUCCESS) {
                tts.setLanguage(Locale.US);
            } else {
                Log.e(TAG, "TTS init failed");
            }
        });

        inferenceThread = new HandlerThread("inference");
        inferenceThread.start();
        inferenceHandler = new Handler(inferenceThread.getLooper());

        try {
            classifier = new SignClassifier(getAssets(), MODEL_FILE);
        } catch (IOException e) {
            Log.e(TAG, "Failed to load model: " + MODEL_FILE, e);
            Toast.makeText(this,
                    "Model not found: place a trained " + MODEL_FILE + " in app/src/main/assets/ (see training/README.md)",
                    Toast.LENGTH_LONG).show();
        }

        requestCameraPermissionIfNeeded();
    }

    private void requestCameraPermissionIfNeeded() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, REQ_CAMERA);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_CAMERA && grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            if (surfaceView.getHolder().getSurface() != null && surfaceView.getHolder().getSurface().isValid()) {
                openCamera(surfaceView.getHolder());
            }
        } else {
            Toast.makeText(this, "Camera permission is required.", Toast.LENGTH_LONG).show();
        }
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_GRANTED) {
            openCamera(holder);
        }
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        // preview size fixed at open time; no-op
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        releaseCamera();
    }

    @SuppressWarnings("deprecation")
    private void openCamera(SurfaceHolder holder) {
        try {
            camera = Camera.open(); // legacy API: only reliable path on Note 2 firmware
            Camera.Parameters params = camera.getParameters();

            Camera.Size bestSize = chooseSmallestUsablePreviewSize(params.getSupportedPreviewSizes());
            params.setPreviewSize(bestSize.width, bestSize.height);
            params.setPreviewFormat(android.graphics.ImageFormat.NV21);
            camera.setParameters(params);

            camera.setPreviewDisplay(holder);
            camera.setPreviewCallback((data, cam) -> onPreviewFrame(data, bestSize.width, bestSize.height));
            camera.startPreview();
        } catch (IOException e) {
            Log.e(TAG, "Failed to open camera", e);
        }
    }

    /** Smallest preview size >= model input resolution: less data to convert/copy per frame. */
    @SuppressWarnings("deprecation")
    private Camera.Size chooseSmallestUsablePreviewSize(List<Camera.Size> sizes) {
        Camera.Size best = sizes.get(0);
        for (Camera.Size s : sizes) {
            boolean usable = s.width >= SignClassifier.INPUT_SIZE && s.height >= SignClassifier.INPUT_SIZE;
            boolean smaller = (s.width * s.height) < (best.width * best.height);
            if (usable && (smaller || best.width < SignClassifier.INPUT_SIZE)) {
                best = s;
            }
        }
        return best;
    }

    private void onPreviewFrame(byte[] data, int width, int height) {
        long now = System.currentTimeMillis();
        if (now - lastInferenceTime < INFERENCE_INTERVAL_MS) {
            return; // throttle: skip this frame entirely, don't queue up work
        }
        lastInferenceTime = now;

        if (classifier == null) return;

        inferenceHandler.post(() -> {
            byte[] rgb = FrameUtils.nv21ToRgbSquare(data, width, height, SignClassifier.INPUT_SIZE);
            SignClassifier.Result result = classifier.classify(rgb);
            mainHandler.post(() -> handleResult(result));
        });
    }

    private void handleResult(SignClassifier.Result result) {
        predictedLetterView.setText(result.confidence >= CONFIDENCE_THRESHOLD ? result.label : "");

        if (result.confidence < CONFIDENCE_THRESHOLD || result.label.equals("BLANK")) {
            stableCount = 0;
            lastLabel = "";
            return;
        }

        if (result.label.equals(lastLabel)) {
            stableCount++;
        } else {
            lastLabel = result.label;
            stableCount = 1;
        }

        if (stableCount == STABLE_FRAMES_REQUIRED) {
            commitLabel(result.label);
        }
    }

    private void commitLabel(String label) {
        if (label.equals("SPACE")) {
            speakBuffer();
        } else {
            wordBuffer.append(label);
            spokenBufferView.setText(wordBuffer.toString());
        }
    }

    private void speakBuffer() {
        if (wordBuffer.length() == 0) return;
        String word = wordBuffer.toString();
        tts.speak(word, TextToSpeech.QUEUE_ADD, null, "utt_" + System.currentTimeMillis());
        wordBuffer.setLength(0);
        spokenBufferView.setText("");
    }

    @SuppressWarnings("deprecation")
    private void releaseCamera() {
        if (camera != null) {
            camera.setPreviewCallback(null);
            camera.stopPreview();
            camera.release();
            camera = null;
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        releaseCamera();
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (surfaceView.getHolder().getSurface() != null && surfaceView.getHolder().getSurface().isValid()
                && ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCamera(surfaceView.getHolder());
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (classifier != null) classifier.close();
        if (tts != null) { tts.stop(); tts.shutdown(); }
        inferenceThread.quitSafely();
    }
}
