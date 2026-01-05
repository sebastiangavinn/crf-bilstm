import re

def log_preprocessing(text: str, logger):
    lowered = text.lower()
    cleaned = re.sub(r'[^a-z0-9\s]', '', lowered)
    tokens = cleaned.split()

    logger.info("=== Preprocessing Example ===")
    logger.info(f"Original     : {text}")
    logger.info(f"Case folding : {lowered}")
    logger.info(f"Tokenized    : {tokens}")

def log_bio_example(tokens, tags, logger):
    logger.info("=== BIO Labeling Example ===")
    for t, tag in zip(tokens, tags):
        logger.info(f"{t:12s} -> {tag}")

def log_encoding(tokens, vocab, max_len, logger):
    encoded = [vocab.word2idx.get(t, 1) for t in tokens]
    padded = encoded + [0] * (max_len - len(encoded))

    logger.info("=== Encoding & Padding Example ===")
    logger.info(f"Tokens  : {tokens}")
    logger.info(f"Encoded : {encoded}")
    logger.info(f"Padded  : {padded[:max_len]}")

def log_model_prediction(model, sentence, vocab, device, logger):
    from src.utils.inference import predict_sentence

    result = predict_sentence(model, sentence, vocab, device)

    logger.info("=== Model Prediction Example ===")
    for token, tag in result:
        logger.info(f"{token:12s} -> {tag}")
