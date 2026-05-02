"""
Voice Cloning with Chatterbox TTS
===================================
resemble-ai/chatterbox 모델을 사용하여 WAV 레퍼런스 기반 zero-shot 음성 복제.

Reference files:
  - ElevenLabs_Adam_angry.wav
  - ElevenLabs_Samantha_calm.wav

Output files (cloned_outputs/):
  - chatterbox_adam_angry.wav
  - chatterbox_samantha_calm.wav

Install:
  pip install chatterbox-tts
"""

import sys
import perth
import torch
import torchaudio
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Windows: perth.PerthImplicitWatermarker == None -> DummyWatermarker fallback
# ──────────────────────────────────────────────────────────────────────────────
if perth.PerthImplicitWatermarker is None:
    print("[INFO] perth.PerthImplicitWatermarker not available - using DummyWatermarker.")
    perth.PerthImplicitWatermarker = perth.DummyWatermarker

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "cloned_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REFERENCES = {
    "adam_angry":    BASE_DIR / "ElevenLabs_Adam_angry.wav",
    "samantha_calm": BASE_DIR / "ElevenLabs_Samantha_calm.wav",
}

TEXTS = {
    "adam_angry":    "I hear you, and I want you to know that everything is going to be just fine.",
    "samantha_calm": "You had ONE job - how could you possibly mess this up AGAIN?!",
}

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"[INFO] Device: {DEVICE}")

    for key, path in REFERENCES.items():
        if not path.exists():
            print(f"[ERROR] Reference file not found: {path}")
            sys.exit(1)

    print("\n[Chatterbox] Loading model...")
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError:
        print("[ERROR] Run: pip install chatterbox-tts")
        sys.exit(1)

    model = ChatterboxTTS.from_pretrained(device=DEVICE)
    print(f"[Chatterbox] Model ready. Sample rate: {model.sr} Hz")

    for key, ref_path in REFERENCES.items():
        out_path = OUTPUT_DIR / f"chatterbox_{key}.wav"
        text = TEXTS[key]

        print(f"\n[Chatterbox] Generating -> {out_path.name}")
        print(f"  Reference : {ref_path.name}")
        print(f"  Text      : {text}")

        wav = model.generate(
            text,
            audio_prompt_path=str(ref_path),
        )
        torchaudio.save(str(out_path), wav, model.sr)
        print(f"  [OK] Saved: {out_path}")

    print("\n" + "=" * 55)
    print("  Output files in cloned_outputs/")
    print("=" * 55)
    for f in sorted(OUTPUT_DIR.glob("chatterbox_*.wav")):
        size_kb = f.stat().st_size / 1024
        print(f"  {f.name:<40} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
