"""RAG pipeline: retrieve documentation chunks, then generate an answer with a local LLM."""
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("QA_MODEL", "Qwen/Qwen2.5-3B-Instruct")   # use Qwen/Qwen2.5-1.5B-Instruct on weak machines

SYSTEM = ("You answer questions about the FastAPI documentation. "
          "Use ONLY the provided context. Only include code that appears in the context; "
          "never write your own code examples. "
          "If the answer is not in the context, say you don't know. "
          "End your answer with a line like 'Sources: [1], [3]' listing the context numbers you used.")


class RAG:
    def __init__(self, retriever, model_name=MODEL):
        self.retriever = retriever
        cuda = torch.cuda.is_available()
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if cuda else torch.float32,
            device_map="auto" if cuda else None,
        )
        print(f"Loaded {model_name} on {self.model.device}")

    def answer(self, question, k=4, max_new_tokens=300, retriever=None):
        """Return (answer_text, retrieved_chunks)."""
        hits = (retriever or self.retriever).search(question, k=k)
        context = "\n\n".join(
            f"[{i + 1}] ({h['source']} :: {h['heading']})\n{h['text']}" for i, h in enumerate(hits))
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip(), hits
