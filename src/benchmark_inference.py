import json
import time
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

MODEL_NAME = "distilgpt2"
OUTPUT_PATH = "results/benchmark_results.json"


def get_gpu_memory_mb():

    if not torch.cuda.is_available():
        return 0

    return torch.cuda.max_memory_allocated() / 1024 / 1024


def reset_gpu_memory():

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build_fake_medical_context():

    paragraph = """
    O paciente apresenta histórico clínico complexo,
    sintomas respiratórios persistentes,
    alterações laboratoriais e necessidade
    de acompanhamento multidisciplinar.
    """

    # reduzido para CPU
    return paragraph * 3


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return model, tokenizer


def run_generation(model, tokenizer, context, use_cache):

    model.config.use_cache = use_cache

    prompt = f"""
    Gere um resumo médico:

    {context}

    Resumo:
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    start_time = time.time()

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            use_cache=use_cache,
            pad_token_id=tokenizer.eos_token_id,
        )

    end_time = time.time()

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return {
        "use_cache": use_cache,
        "tempo_execucao_segundos": round(
            end_time - start_time,
            2
        ),
        "pico_vram_mb": round(
            get_gpu_memory_mb(),
            2
        ),
        "tokens_entrada": int(
            inputs["input_ids"].shape[1]
        ),
        "tokens_gerados": 1,
        "amostra_saida": generated_text[-200:],
    }


def main():

    results = {
        "modelo": MODEL_NAME,
        "cuda_disponivel": torch.cuda.is_available(),
        "flashattention_2_ativo": False,
        "benchmarks": [],
    }

    context = build_fake_medical_context()

    model, tokenizer = load_model()

    print("\nExecutando benchmark SEM KV Cache...\n")

    benchmark_without_cache = run_generation(
        model=model,
        tokenizer=tokenizer,
        context=context,
        use_cache=False,
    )

    results["benchmarks"].append(
        benchmark_without_cache
    )

    print("\nExecutando benchmark COM KV Cache...\n")

    benchmark_with_cache = run_generation(
        model=model,
        tokenizer=tokenizer,
        context=context,
        use_cache=True,
    )

    results["benchmarks"].append(
        benchmark_with_cache
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("\nBenchmark finalizado.\n")

    print(
        json.dumps(
            results,
            indent=4,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()