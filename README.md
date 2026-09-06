# ASL Speak

Android app: webcam -> on-device ML sign classifier -> TextToSpeech. Built to run on a
Samsung Galaxy Note 2 (GT-N7100), stretched to its practical firmware ceiling.

## Hardware reality check (read before filing "why is this slow")

The Note 2 is a 2012 phone: quad-core Exynos 4412 (Cortex-A9), Mali-400MP4 GPU
(OpenGL ES 2.0, no compute shaders), no NNAPI. That rules out GPU/NNAPI delegates
entirely -- everything here runs on CPU via TFLite + XNNPACK. "Low resource, ML-only,
Note-2-optimized" in practice means:

- minSdk 19 (KitKat) -- the Note 2's official OTA ceiling, not stock 4.1.2.
- A tiny int8-quantized CNN (~100-300KB), 96x96 input, no GPU delegate.
- Legacy `android.hardware.Camera` API (Camera2/CameraX need API 21+).
- Inference throttled to ~5/sec with a debounce, not per-frame.

This will also run fine, faster, on any modern Android device -- the constraints
above are the floor, not a ceiling.

## Project layout

- `app/` -- the Android app (Java).
  - `SignClassifier.java` -- TFLite interpreter wrapper, whole-frame classification
    over the fingerspelling alphabet + BLANK + SPACE.
  - `FrameUtils.java` -- NV21 -> RGB pixel conversion and center-crop/downscale to
    the model's fixed input size. This is data plumbing, not a detection algorithm --
    the CNN does all the actual interpretation.
  - `MainActivity.java` -- camera capture, inference scheduling/throttling, TTS.
- `training/train_asl_model.py` -- trains and quantizes the model. **You need to run
  this yourself** against a labeled ASL dataset; no trained weights are included.

## Getting a real model into the app

1. Get a dataset with one folder per label matching `SignClassifier.LABELS`
   (A-Z, SPACE, BLANK) -- e.g. Kaggle's "ASL Alphabet" dataset, plus your own
   "blank/no hand in frame" photos (important: without a BLANK class the app will
   hallucinate a letter in every frame since there's no separate hand detector).
2. `pip install tensorflow==2.13.*`
3. `python training/train_asl_model.py --data_dir <dataset> --out asl_alphabet.tflite`
4. `cp asl_alphabet.tflite app/src/main/assets/`
5. Build the APK (below). Without this file the app installs and runs but shows a
   toast saying the model is missing.

## Building the APK

```
./gradlew assembleDebug
```

Output: `app/build/outputs/apk/debug/app-debug.apk`. Install with
`adb install app/build/outputs/apk/debug/app-debug.apk`, or copy it to the phone
and enable "install from unknown sources".

I have not built or run this project in this session (no Android SDK/emulator
available here) -- treat it as an unverified scaffold. Before relying on it, build
it, install it on a real device, and confirm the Gradle/AGP/TFLite version
combination resolves for your local SDK; pinned versions especially in
`app/build.gradle` may need adjusting.

## Known gaps / next steps

- No trained model is checked in (see above) -- classification is a no-op until
  you supply `asl_alphabet.tflite`.
- Fingerspelling only (single static letter per hand pose). Real ASL includes
  motion-based signs (J, Z, whole words) that a single-frame classifier can't
  capture -- that would need a small temporal model (e.g. a few stacked frames)
  as a follow-up, not a v1 requirement.
- No hand-presence indicator beyond the BLANK class's confidence -- if the model's
  BLANK isn't well-trained, expect false letters when no hand is in frame.
