from transformers import pipeline

translator = pipeline(
    "translation",
    model="facebook/nllb-200-distilled-600M"
)

def translate_text(text: str, src: str, tgt: str) -> str:
    return translator(
        text,
        src_lang=src,
        tgt_lang=tgt
    )[0]["translation_text"]
