## Cuantizacion GPTQ Int8 HighFidelity

Entorno validado:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quant_clean311
```

## Instalacion recomendada (sin usar /tmp del sistema)

Este flujo evita los errores vistos con `gptqmodel` en build aislado de `pip` y guarda temporales/cache en el proyecto.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n quant_clean311 python=3.11
conda activate quant_clean311

cd /bigdata/laboratorio/u028528/test-quant
mkdir -p .tmp .pip-cache

TMPDIR="$PWD/.tmp" PIP_CACHE_DIR="$PWD/.pip-cache" \
pip install -U pip setuptools wheel

TMPDIR="$PWD/.tmp" PIP_CACHE_DIR="$PWD/.pip-cache" \
pip install --no-build-isolation -r requirements.txt
```

Comprobacion rapida del entorno activo:

```bash
which python
which pip
python -V
pip -V
```

## Crear checkpoint cuantizado a 8 bits (Int8 HighFidelity)

Ejecuta este comando para generar el checkpoint GPTQ de 8 bits:

```bash
mkdir -p .tmp .pip-cache .hf-home .xdg-cache logs

XDG_CACHE_HOME="$PWD/.xdg-cache" \
HF_HOME="$PWD/.hf-home" \
HF_HUB_CACHE="$PWD/.hf-home/hub" \
TRANSFORMERS_CACHE="$PWD/.hf-home/transformers" \
TMPDIR="$PWD/.tmp" \
PIP_CACHE_DIR="$PWD/.pip-cache" \
CUDA_VISIBLE_DEVICES=0 \
/home/u028528/miniconda3/envs/quant_clean311/bin/python main.py \
  --model-id "Qwen/Qwen3.5-9B" \
  --output-dir "./Qwen3.5-9B-GPTQ-Int8-HighFidelity" \
  --bits 8 \
  --nsamples 128 \
  --seqlen 1024 \
  --gpu-max-memory 70GiB \
  2>&1 | tee logs/quant_int8_hifi_$(date +%Y%m%d_%H%M%S).log
```

Este flujo crea el checkpoint en:

- `./Qwen3.5-9B-GPTQ-Int8-HighFidelity`

Verificacion rapida (debe mostrar `8`):

```bash
python - <<'PY'
import json
with open('Qwen3.5-9B-GPTQ-Int8-HighFidelity/config.json') as f:
    cfg = json.load(f)
print(cfg['quantization_config']['bits'])
PY
```

Parametros usados para alta fidelidad:

- dataset: `NeelNanda/pile-10k`
- `nsamples=128`
- `seqlen=1024`
- `bits=8`
- `group_size=128`

## Benchmark MMLU

Para comparar modelo base y cuantizado sin tener que escribir todos los flags:

```bash
bash benchmark_mmlu.sh
```

Eso ejecuta:

- modelo base: `Qwen/Qwen3.5-9B`
- modelo cuantizado: variable `QUANT_MODEL` (por defecto del script)
- tarea: `mmlu`
- batch de evaluacion: `2`

Genera dos logs:

- `./eval_base_mmlu.log`
- `./eval_quant_mmlu.log`

Si solo quieres evaluar el cuantizado:

```bash
bash benchmark_mmlu_quant_only.sh
```

Tambien puedes sobreescribir variables:

```bash
BASE_MODEL="Qwen/Qwen3.5-9B" \
QUANT_MODEL="./Qwen3.5-9B-GPTQ-Int8-HighFidelity" \
TASKS="mmlu" \
EVAL_BS=2 \
bash benchmark_mmlu.sh
```

Si quieres evaluar mas tareas:

```bash
TASKS="mmlu,hellaswag,lambada_openai" bash benchmark_mmlu.sh
```

## Notas

- estos scripts no cuantizan; asumen que el checkpoint cuantizado ya existe
- la evaluacion se hace con `auto-round --eval`, no con el stack viejo de `auto_gptq`
- si el cuantizado se guardo en otra carpeta, cambia `QUANT_MODEL`

## Evaluacion limpia de Int8 HighFidelity (recomendada)

Para comparar de forma robusta con `lm_eval`:

```bash
mkdir -p .tmp .pip-cache .hf-home .xdg-cache logs

XDG_CACHE_HOME="$PWD/.xdg-cache" \
HF_HOME="$PWD/.hf-home" \
HF_HUB_CACHE="$PWD/.hf-home/hub" \
TRANSFORMERS_CACHE="$PWD/.hf-home/transformers" \
TMPDIR="$PWD/.tmp" \
PIP_CACHE_DIR="$PWD/.pip-cache" \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD:${PYTHONPATH:-}" \
/home/u028528/miniconda3/envs/quant_clean311/bin/lm_eval run \
  --model hf \
  --model_args pretrained=Qwen/Qwen3.5-9B,trust_remote_code=True,dtype=auto \
  --tasks mmlu \
  --batch_size 2 \
  2>&1 | tee logs/lm_eval_base_clean.log
```

```bash
XDG_CACHE_HOME="$PWD/.xdg-cache" \
HF_HOME="$PWD/.hf-home" \
HF_HUB_CACHE="$PWD/.hf-home/hub" \
TRANSFORMERS_CACHE="$PWD/.hf-home/transformers" \
TMPDIR="$PWD/.tmp" \
PIP_CACHE_DIR="$PWD/.pip-cache" \
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$PWD:${PYTHONPATH:-}" \
/home/u028528/miniconda3/envs/quant_clean311/bin/lm_eval run \
  --model hf \
  --model_args pretrained=./Qwen3.5-9B-GPTQ-Int8-HighFidelity,dtype=auto \
  --tasks mmlu \
  --batch_size 2 \
  2>&1 | tee logs/lm_eval_quant_int8_hifi_full.log
```

Para ver el resumen al final de cada log:

```bash
tail -n 80 logs/lm_eval_base_clean.log
tail -n 80 logs/lm_eval_quant_int8_hifi_full.log
```

## Diagnostico del error de instalacion

Error observado al instalar `requirements.txt`:

- `gptqmodel` fallaba en el entorno de build aislado de `pip` porque no detectaba `torch`
- al desactivar aislamiento aparecieron errores de metadata/build tools en combinaciones de versiones

Solucion aplicada y validada:

- usar Python `3.11` en entorno limpio (`quant_clean311`)
- instalar con `--no-build-isolation`
- usar `TMPDIR` y `PIP_CACHE_DIR` dentro del proyecto (`.tmp`, `.pip-cache`)

Con este flujo, la instalacion de `requirements.txt` finaliza correctamente.

## Comprobaciones utiles

Para ver que entorno tienes activo:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda env list
```

Para inspeccionar las versiones reales del entorno:

```bash
conda activate quant_clean311
python -m pip list | rg 'torch|transformers|optimum|auto|gptq|lm-eval|peft'
```

Para inspeccionar el stack actual:

```bash
conda activate quant_clean311
python -m pip list | rg 'torch|transformers|optimum|accelerate|datasets|gptqmodel|auto-round|lm-eval'
```

## Conclusion

El problema no era solo el script. El entorno y el proceso de build de dependencias tambien importan.

Antes de intentar evaluar o inferir con este export, hay que decidir una de estas rutas:

1. reconstruir un entorno compatible con `auto_gptq`
2. dejar de usar `auto_gptq` y mover la carga del modelo a un stack moderno compatible con Qwen 3.5
3. evaluar el modelo cuantizado solo con la via soportada por `auto-round`, pero con un entorno limpio y consistente
