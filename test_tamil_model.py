import os
import sys
import time

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "sabaridsnfuji/Tamil_Offline_Handwritten_OCR"

ENCODER_NAME = "google/vit-base-patch16-224-in21k"

DECODER_NAME = "d42kw01f/Tamil-RoBERTa"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("=" * 60)
    print("Tamil Handwriting OCR Test")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")

    print("\nLoading Tamil OCR model...")
    print("First run will download the model.")
    print("Please wait...\n")

    start = time.perf_counter()

    # --------------------------------------------------------
    # IMAGE PROCESSOR
    # --------------------------------------------------------

    image_processor = AutoImageProcessor.from_pretrained(
        ENCODER_NAME
    )

    # --------------------------------------------------------
    # TAMIL TOKENIZER
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        DECODER_NAME
    )

    # --------------------------------------------------------
    # PROCESSOR
    # --------------------------------------------------------

    processor = TrOCRProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_NAME
    )

    model.to(DEVICE)

    model.eval()

    # --------------------------------------------------------
    # GENERATION SETTINGS
    # --------------------------------------------------------

    model.config.decoder_start_token_id = (
        processor.tokenizer.cls_token_id
    )

    model.config.pad_token_id = (
        processor.tokenizer.pad_token_id
    )

    model.config.eos_token_id = (
        processor.tokenizer.sep_token_id
    )

    model.config.vocab_size = (
        model.config.decoder.vocab_size
    )

    elapsed = time.perf_counter() - start

    print(
        f"\nModel loaded successfully in {elapsed:.1f} seconds."
    )

    return processor, model


# ============================================================
# OCR
# ============================================================

def recognize(image_path, processor, model):

    print("\n" + "=" * 60)

    print("Reading image:")

    print(image_path)

    print("=" * 60)

    # --------------------------------------------------------
    # CHECK FILE
    # --------------------------------------------------------

    if not os.path.exists(image_path):

        print("\nERROR: Image does not exist.")

        return


    # --------------------------------------------------------
    # OPEN IMAGE
    # --------------------------------------------------------

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as error:

        print(
            "\nCould not open image:"
        )

        print(error)

        return


    print(
        f"\nOriginal image size: {image.size}"
    )


    # --------------------------------------------------------
    # PROCESS IMAGE
    # --------------------------------------------------------

    pixel_values = processor(
        images=image,
        return_tensors="pt"
    ).pixel_values


    pixel_values = pixel_values.to(
        DEVICE
    )


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    print("\nRunning Tamil recognition...")

    start = time.perf_counter()


    with torch.inference_mode():

        generated_ids = model.generate(

            pixel_values,

            max_length=64,

            num_beams=4,

            early_stopping=True,

            no_repeat_ngram_size=3,

        )


    elapsed = time.perf_counter() - start


    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    text = processor.batch_decode(

        generated_ids,

        skip_special_tokens=True,

    )[0]


    text = text.strip()


    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    print("\n" + "=" * 60)

    print("RESULT")

    print("=" * 60)


    if text:

        print("\nRecognized Tamil:")

        print()

        print(text)

    else:

        print(
            "\nNo text was recognized."
        )


    print(
        f"\nRecognition time: {elapsed:.2f} seconds"
    )

    print("=" * 60)


# ============================================================
# MAIN
# ============================================================

def main():

    processor, model = load_model()


    print("\n")

    print(
        "Enter the full path of your Tamil handwriting image."
    )

    print()

    print(
        "Example:"
    )

    print(
        r"C:\Users\mouli\Downloads\amma.jpg"
    )

    print()


    image_path = input(
        "Image path: "
    ).strip()


    # Remove quotes automatically if the user copied
    # the path using Windows "Copy as path".

    image_path = image_path.strip(
        '"'
    )


    recognize(
        image_path,
        processor,
        model
    )


if __name__ == "__main__":

    main()