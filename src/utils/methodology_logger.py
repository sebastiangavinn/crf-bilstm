from src.utils.preprocess_logger import log_preprocessing
from src.utils.inference import predict_sentence

def log_methodology_examples(logger, vocab, model=None, device=None):
    logger.info("=== METHODOLOGY EXAMPLES START ===")

    # --------------------------------------------------
    # 4.2.1 Preprocessing (Case Folding & Tokenizing)
    # --------------------------------------------------
    example_text = "Penyakit hawar daun bakteri menyebabkan daun padi mengering"
    log_preprocessing(example_text, logger)

    # --------------------------------------------------
    # 4.2.2 BIO Labeling (Contoh manual)
    # --------------------------------------------------
    tokens = ["penyakit", "hawar", "daun", "bakteri", "menyebabkan", "daun", "padi", "mengering"]
    tags = ["O", "B-PENYAKIT", "I-PENYAKIT", "I-PENYAKIT", "O", "B-BAGIAN_TANAMAN", "I-BAGIAN_TANAMAN", "B-GEJALA"]

    logger.info("=== BIO Labeling Example ===")
    for t, tag in zip(tokens, tags):
        logger.info(f"{t:12s} -> {tag}")

    # --------------------------------------------------
    # 4.2.3 Encoding & Padding
    # --------------------------------------------------
    encoded = [vocab.word2idx.get(t, 1) for t in tokens]
    padded = encoded + [0] * (10 - len(encoded))

    logger.info("=== Encoding & Padding Example ===")
    logger.info(f"Tokens  : {tokens}")
    logger.info(f"Encoded : {encoded}")
    logger.info(f"Padded  : {padded}")

    # --------------------------------------------------
    # 4.3 Model Prediction (jika model tersedia)
    # --------------------------------------------------
    if model is not None and device is not None:
        logger.info("=== Model Prediction Example ===")
        result = predict_sentence(
            model,
            "Penyakit hawar daun bakteri menyebabkan daun padi mengering",
            vocab,
            device
        )
        for token, tag in result:
            logger.info(f"{token:12s} -> {tag}")

    logger.info("=== METHODOLOGY EXAMPLES END ===")
