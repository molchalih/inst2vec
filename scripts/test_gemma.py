import torch
from transformers import pipeline


pipe = pipeline(
    "image-text-to-text",
    model="google/translategemma-4b-it",
    device="cuda",
    dtype=torch.bfloat16,
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "source_lang_code": "ru",
                "target_lang_code": "en",
                "text": """Вечор, ты помнишь, вьюга злилась,
На мутном небе мгла носилась;
Луна, как бледное пятно,
Сквозь тучи мрачные желтела,
И ты печальная сидела —
А нынче… погляди в окно:""",
            }
        ],
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])
