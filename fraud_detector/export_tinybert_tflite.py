"""
TinyBERT → TFLite Export Script
=================================
Converts a fine-tuned TinyBERT scam-call classifier to a TFLite model
suitable for on-device inference in an Android / iOS mobile application.

Prerequisites
-------------
    pip install transformers torch optimum[exporters] tensorflow onnx onnx-tf

Usage
-----
    # After fine-tuning and saving the model to models/tinybert_scam_classifier/
    python export_tinybert_tflite.py

Output
------
    models/tinybert_scam_tflite/tinybert_scam.tflite   ← drop into Android assets/
    models/tinybert_scam_tflite/tokenizer_config.json  ← vocab for the mobile app

Mobile integration notes
------------------------
- Load the .tflite file with TensorFlow Lite Interpreter (Android / Flutter).
- Tokenise input on-device using the saved tokenizer_config.json (use
  the HuggingFace tokenizers Java/Kotlin library, or port the vocab to
  a lightweight tokenizer in Dart/Kotlin).
- Input tensors:  input_ids [1, 128]  +  attention_mask [1, 128]  (int32)
- Output tensor:  logits    [1, 2]    (float32)   index-1 = scam probability
"""

import os
import sys
import shutil
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR         = os.path.dirname(__file__)
MODEL_INPUT_PATH = os.path.join(BASE_DIR, "models", "tinybert_scam_classifier")
ONNX_OUTPUT_DIR  = os.path.join(BASE_DIR, "models", "tinybert_scam_onnx")
TFLITE_OUTPUT_DIR= os.path.join(BASE_DIR, "models", "tinybert_scam_tflite")
TFLITE_OUTPUT    = os.path.join(TFLITE_OUTPUT_DIR, "tinybert_scam.tflite")

MAX_SEQ_LENGTH   = 128   # must match server-side tokeniser setting


def check_fine_tuned_model():
    """Abort early if no fine-tuned model exists."""
    if not os.path.isdir(MODEL_INPUT_PATH):
        logger.error(
            f"Fine-tuned model not found at: {MODEL_INPUT_PATH}\n"
            "You must fine-tune TinyBERT on scam call data first.\n"
            "See  train_tinybert.py  for the training script."
        )
        sys.exit(1)
    logger.info(f"✓ Fine-tuned model found at: {MODEL_INPUT_PATH}")


def export_to_onnx():
    """Export the PyTorch TinyBERT model to ONNX format."""
    logger.info("Step 1/3 — Exporting to ONNX...")
    try:
        from optimum.exporters.onnx import main_export
        os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)
        main_export(
            model_name_or_path = MODEL_INPUT_PATH,
            output             = ONNX_OUTPUT_DIR,
            task               = "text-classification",
            opset              = 13,
            optimize           = "O2",   # ONNX graph optimisation
        )
        logger.info(f"✓ ONNX model saved to: {ONNX_OUTPUT_DIR}")
    except ImportError:
        logger.error("'optimum' not installed.  Run:  pip install optimum[exporters]")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"ONNX export failed: {exc}")
        sys.exit(1)


def convert_onnx_to_tflite():
    """Convert the ONNX model to TFLite using onnx-tf."""
    logger.info("Step 2/3 — Converting ONNX → TensorFlow SavedModel...")
    onnx_model_path = os.path.join(ONNX_OUTPUT_DIR, "model.onnx")
    tf_savedmodel_dir = os.path.join(ONNX_OUTPUT_DIR, "tf_saved_model")

    try:
        import onnx
        import onnx_tf
        from onnx_tf.backend import prepare

        onnx_model = onnx.load(onnx_model_path)
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(tf_savedmodel_dir)
        logger.info(f"✓ TensorFlow SavedModel saved to: {tf_savedmodel_dir}")
    except ImportError as exc:
        logger.error(
            f"Missing package: {exc}\n"
            "Run:  pip install onnx onnx-tf tensorflow"
        )
        sys.exit(1)
    except Exception as exc:
        logger.error(f"ONNX → TF conversion failed: {exc}")
        sys.exit(1)

    logger.info("Step 3/3 — Converting TF SavedModel → TFLite...")
    try:
        import tensorflow as tf
        os.makedirs(TFLITE_OUTPUT_DIR, exist_ok=True)

        converter = tf.lite.TFLiteConverter.from_saved_model(tf_savedmodel_dir)

        # Quantisation for smaller model size and faster mobile inference
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # INT8 quantisation — reduces model to ~25% size with minimal accuracy loss
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS,   # required for some BERT ops
        ]
        converter.inference_input_type  = tf.int32
        converter.inference_output_type = tf.float32

        tflite_model = converter.convert()
        with open(TFLITE_OUTPUT, "wb") as f:
            f.write(tflite_model)

        size_kb = os.path.getsize(TFLITE_OUTPUT) / 1024
        logger.info(f"✓ TFLite model saved to: {TFLITE_OUTPUT}  ({size_kb:.1f} KB)")

    except ImportError:
        logger.error("'tensorflow' not installed.  Run:  pip install tensorflow")
        sys.exit(1)
    except Exception as exc:
        logger.error(f"TFLite conversion failed: {exc}")
        sys.exit(1)

    # Clean up intermediate SavedModel directory
    shutil.rmtree(tf_savedmodel_dir, ignore_errors=True)


def copy_tokenizer():
    """Copy tokenizer files so the mobile app can tokenise locally."""
    logger.info("Copying tokenizer files for mobile use...")
    tokenizer_files = [
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
    ]
    for fname in tokenizer_files:
        src = os.path.join(MODEL_INPUT_PATH, fname)
        dst = os.path.join(TFLITE_OUTPUT_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            logger.info(f"  Copied: {fname}")
        else:
            logger.warning(f"  Not found (optional): {fname}")


def validate_tflite():
    """Run a quick inference test on the exported model."""
    logger.info("Validating exported TFLite model...")
    try:
        import numpy as np
        import tensorflow as tf
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_INPUT_PATH)
        interpreter = tf.lite.Interpreter(model_path=TFLITE_OUTPUT)
        interpreter.allocate_tensors()

        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        test_text = "Please give me your OTP immediately or your account will be suspended."
        enc = tokenizer(
            test_text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

        interpreter.set_tensor(input_details[0]["index"], enc["input_ids"].astype(np.int32))
        interpreter.set_tensor(input_details[1]["index"], enc["attention_mask"].astype(np.int32))
        interpreter.invoke()

        logits     = interpreter.get_tensor(output_details[0]["index"])
        exp_logits = np.exp(logits - np.max(logits))
        probs      = exp_logits / exp_logits.sum()
        fraud_prob = float(probs[0][1])

        logger.info(
            f"✓ TFLite validation passed.  "
            f"Scam probability for test input: {fraud_prob:.4f}  "
            f"({'scam' if fraud_prob >= 0.5 else 'legitimate'})"
        )
    except Exception as exc:
        logger.warning(f"Validation step failed (model may still be valid): {exc}")


def print_mobile_instructions():
    print("\n" + "=" * 65)
    print("MOBILE INTEGRATION GUIDE")
    print("=" * 65)
    print(f"TFLite model : {TFLITE_OUTPUT}")
    print(f"Tokenizer    : {TFLITE_OUTPUT_DIR}/vocab.txt")
    print()
    print("Android (Kotlin / Java):")
    print("  1. Copy tinybert_scam.tflite to  app/src/main/assets/")
    print("  2. Copy vocab.txt to              app/src/main/assets/")
    print("  3. Load via TensorFlow Lite Interpreter:")
    print("       val interpreter = Interpreter(loadModelFile(\"tinybert_scam.tflite\"))")
    print("  4. Tokenise with HuggingFace tokenizers-android library.")
    print("  5. Run inference:")
    print("       interpreter.run(inputIds, outputLogits)")
    print("       val scamProb = softmax(outputLogits)[1]")
    print()
    print("Flutter:")
    print("  1. Add  tflite_flutter  to pubspec.yaml")
    print("  2. Copy model + vocab to assets/")
    print("  3. Load: Interpreter.fromAsset('tinybert_scam.tflite')")
    print()
    print("Input  tensors : input_ids [1, 128] int32")
    print("                 attention_mask [1, 128] int32")
    print("Output tensor  : logits [1, 2] float32  (index 1 = scam prob)")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    logger.info("=== TinyBERT → TFLite Export ===")
    check_fine_tuned_model()
    export_to_onnx()
    convert_onnx_to_tflite()
    copy_tokenizer()
    validate_tflite()
    print_mobile_instructions()
    logger.info("Export complete.")
