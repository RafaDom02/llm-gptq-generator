import argparse
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cuantiza Qwen 3.5 a GPTQ 4-bit con una calibracion parecida a AutoRound."
    )
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--output-dir", default="./Qwen3.5-9B-GPTQ-Int4")
    parser.add_argument("--dataset", default="NeelNanda/pile-10k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--nsamples", type=int, default=64)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device-mode", choices=("single_gpu", "auto", "cpu"), default="single_gpu")
    parser.add_argument("--gpu-max-memory", default="12GiB")
    parser.add_argument("--cpu-max-memory", default="64GiB")
    parser.add_argument("--cache-block-outputs", action="store_true")
    parser.add_argument("--no-cache-block-outputs", dest="cache_block_outputs", action="store_false")
    parser.set_defaults(cache_block_outputs=False)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false")
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"\nERROR: {message}\n", file=sys.stderr)
    raise SystemExit(1)


def get_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def parse_major(raw_version: str | None) -> int | None:
    if not raw_version:
        return None

    try:
        return int(raw_version.split(".", 1)[0])
    except ValueError:
        return None


def ensure_compatible_stack() -> None:
    transformers_version = get_version("transformers")
    optimum_version = get_version("optimum")
    gptqmodel_version = get_version("gptqmodel")

    transformers_major = parse_major(transformers_version)
    optimum_major = parse_major(optimum_version)

    if transformers_major is None:
        fail("No encuentro `transformers` en el entorno activo.")

    if optimum_version is None:
        fail(
            "No encuentro `optimum` en el entorno activo. "
            "Instala: `pip install --upgrade optimum gptqmodel`"
        )

    if transformers_major >= 5 and (optimum_major is None or optimum_major < 2):
        fail(
            "Tu entorno mezcla `transformers` moderno con un `optimum` demasiado antiguo.\n\n"
            f"Detectado:\n"
            f"- transformers=={transformers_version}\n"
            f"- optimum=={optimum_version}\n"
            f"- gptqmodel=={gptqmodel_version or 'NO INSTALADO'}\n\n"
            "Ese combo provoca exactamente errores como:\n"
            "- `ImportError: cannot import name 'is_tf_available'`\n\n"
            "Arreglalo asi en `quant_clean`:\n"
            "1. `pip uninstall -y auto-gptq`\n"
            "2. `pip install --upgrade transformers optimum accelerate datasets gptqmodel`\n"
            "3. vuelve a ejecutar `python main.py ...`\n\n"
            "No uses `auto_gptq==0.7.1` con este stack."
        )


def load_dependencies():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        fail(
            "Falta `datasets`. Instala al menos: "
            "`pip install datasets transformers accelerate optimum gptqmodel`"
        )
        raise exc

    try:
        import torch
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoTokenizer,
            GPTQConfig,
        )
    except ImportError as exc:
        fail(
            "No se pudieron importar los componentes de `transformers` para GPTQ. "
            "Asegurate de tener una version reciente de `transformers`, `optimum` y `gptqmodel`."
        )
        raise exc

    return load_dataset, torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPTQConfig


def patch_gptq_hf_device_map(torch) -> None:
    from transformers.quantizers.quantizer_gptq import GptqHfQuantizer

    if getattr(GptqHfQuantizer, "_patched_hf_device_map", False):
        return

    original = GptqHfQuantizer._process_model_after_weight_loading

    def wrapped(self, model, **kwargs):
        if not hasattr(model, "hf_device_map"):
            if torch.cuda.is_available():
                model.hf_device_map = {"": 0}
            else:
                model.hf_device_map = {"": "cpu"}

        return original(self, model, **kwargs)

    GptqHfQuantizer._process_model_after_weight_loading = wrapped
    GptqHfQuantizer._patched_hf_device_map = True


def build_max_memory(torch, args: argparse.Namespace) -> dict | None:
    if args.device_mode == "cpu":
        return None

    if not torch.cuda.is_available():
        return None

    if args.device_mode in ("single_gpu", "auto"):
        return {
            0: args.gpu_max_memory,
            "cpu": args.cpu_max_memory,
        }

    return None


def build_device_map(torch, args: argparse.Namespace):
    if args.device_mode == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        # Para GPTQ en optimum/transformers se necesita `hf_device_map`
        # durante `pack_model`; con mapas manuales puede no existir.
        # Usamos `auto` y limitamos memoria con `max_memory`.
        return "auto"

    return "cpu"


def pick_text(row: dict, preferred_key: str) -> str | None:
    if preferred_key in row and isinstance(row[preferred_key], str):
        return row[preferred_key]

    for key in ("text", "content", "prompt", "document"):
        value = row.get(key)
        if isinstance(value, str):
            return value

    return None


def build_calibration_texts(load_dataset, tokenizer, args: argparse.Namespace) -> list[str]:
    dataset = load_dataset(args.dataset, split=args.split)
    texts: list[str] = []

    for row in dataset:
        raw_text = pick_text(row, args.text_column)
        if not raw_text or not raw_text.strip():
            continue

        tokenized = tokenizer(
            raw_text,
            truncation=True,
            max_length=args.seqlen,
            add_special_tokens=False,
        )
        input_ids = tokenized.get("input_ids") or []
        if not input_ids:
            continue

        clipped_text = tokenizer.decode(input_ids, skip_special_tokens=True).strip()
        if not clipped_text:
            continue

        texts.append(clipped_text)
        if len(texts) >= args.nsamples:
            break

    if len(texts) < args.nsamples:
        fail(
            f"Solo pude extraer {len(texts)} muestras validas de `{args.dataset}`. "
            f"Revisa `--text-column`, `--split` o reduce `--nsamples`."
        )

    return texts


def quantize(args: argparse.Namespace) -> None:
    ensure_compatible_stack()
    load_dataset, torch, AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPTQConfig = load_dependencies()
    patch_gptq_hf_device_map(torch)

    print("=== Configuracion ===")
    print(f"model_id     : {args.model_id}")
    print(f"output_dir   : {args.output_dir}")
    print(f"dataset      : {args.dataset}")
    print(f"split        : {args.split}")
    print(f"text_column  : {args.text_column}")
    print(f"bits         : {args.bits}")
    print(f"group_size   : {args.group_size}")
    print(f"nsamples     : {args.nsamples}")
    print(f"seqlen       : {args.seqlen}")
    print(f"batch_size   : {args.batch_size}")
    print(f"device_mode  : {args.device_mode}")
    print(f"gpu_max_mem  : {args.gpu_max_memory}")
    print(f"cpu_max_mem  : {args.cpu_max_memory}")
    print(f"cache_blocks : {args.cache_block_outputs}")
    print("=====================\n")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    config = AutoConfig.from_pretrained(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
    )
    if not hasattr(config, "use_cache"):
        text_config = getattr(config, "text_config", None)
        config.use_cache = getattr(text_config, "use_cache", True)

    calibration_texts = build_calibration_texts(load_dataset, tokenizer, args)
    print(f"Calibration samples listas: {len(calibration_texts)}")

    quantization_config = GPTQConfig(
        bits=args.bits,
        group_size=args.group_size,
        dataset=calibration_texts,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        model_seqlen=args.seqlen,
        cache_block_outputs=args.cache_block_outputs,
    )

    print("Cargando y cuantizando el modelo...")
    device_map = build_device_map(torch, args)
    max_memory = build_max_memory(torch, args)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            config=config,
            device_map=device_map,
            max_memory=max_memory,
            torch_dtype="auto",
            trust_remote_code=args.trust_remote_code,
            quantization_config=quantization_config,
        )
    except torch.OutOfMemoryError:
        fail(
            "CUDA OOM durante la cuantizacion.\n\n"
            "Prueba una configuracion mas conservadora, por ejemplo:\n"
            "--nsamples 32 --seqlen 512 --gpu-max-memory 10GiB\n\n"
            "Tambien puedes cerrar otros procesos que esten usando la GPU."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nCuantizacion completada.")
    print(f"Modelo guardado en: {output_dir.resolve()}")


def main() -> None:
    args = parse_args()
    quantize(args)


if __name__ == "__main__":
    main()
