from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import ModelArgs, Transformer

class LLaMA:

    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
            assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
            ckpt_path = checkpoints[0]
            print(f'Loading checkpoint "{ckpt_path}"')
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            print(f"Loaded checkpoint in {time.time() - prev_time:.2f}s")
            prev_time = time.time()
        with open(Path(checkpoints_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()
        
        if device == "cuda":
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.HalfTensor)
        
        model = Transformer(model_args).to(device)

        if load_model:
            # The only unmatched key in the checkpoint is rope.freqs. Remove it
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f"Loaded state dict in {time.time() - prev_time:.2f}s")
        
        return LLaMA(model, tokenizer, model_args)

    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None, top_k: int = 50 ,sampling_strategy: str = 'top_p', beam_size: int = 3):
        print("Using sampling strategy:", sampling_strategy)
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, f"batch size must be less than or equal to {self.args.max_batch_size}"
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len, f"prompt length must be less than or equal to {self.args.max_seq_len}"
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Default: other sampling strategies
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id
        cur_iterator = tqdm(range(1, total_len), desc="Generating tokens")
        for cur_pos in cur_iterator:
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos], cur_pos)
            if temperature != 1.0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            if sampling_strategy == 'top_p':
                next_token = self._sample_top_p(probs, top_p)
            elif sampling_strategy == 'top_k':
                next_token = self._sample_top_k(probs, top_k)
            elif sampling_strategy == 'random_sampling':
                next_token = self.random_sampling(probs)
            elif sampling_strategy == 'greedy':
                next_token = self.greedy(probs)
            else:
                raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")
            next_token = next_token.reshape(-1)
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        return (out_tokens, out_text)

    def _sample_top_k(self, probs, top_k):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, top_k)
        probs_sort = probs_sort[:, :top_k]
        # (B, top_k)
        probs_idx = probs_idx[:, :top_k]
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top k distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token) 
        return next_token
    
    def random_sampling(self, probs):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token) 
        return next_token

    def greedy(self, probs):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.argmax(probs_sort, dim=-1)
        next_token = torch.gather(probs_idx, -1, next_token.unsqueeze(-1)).squeeze(-1)
        return next_token
    
    def _sample_top_p(self, probs, p):
        # (B, vocab_size)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # (B, vocab_size)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        # (B, vocab_size)
        # (Substracting "probs_sort" shifts the cumulative sum by 1 position to the right before masking)
        mask = probs_sum - probs_sort > p 
        # Zero out all the probabilities of tokens that are not selected by the Top P
        probs_sort[mask] = 0.0 
        # Redistribute the probabilities so that they sum up to 1.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        # Sample a token (its index) from the top p distribution
        next_token = torch.multinomial(probs_sort, num_samples=1)
        # Get the token position in the vocabulary corresponding to the sampled index
        next_token = torch.gather(probs_idx, -1, next_token) 
        return next_token

    def beam_search(self, prompt: str, max_gen_len: int = 20, beam_size: int = 3):
        """
        Sequence-level beam search for text generation with per-beam KV cache management.
        prompt: input prompt string
        max_gen_len: maximum number of tokens to generate
        beam_size: number of beams to keep
        Returns: list of (sequence, score) tuples
        """
        import copy
        device = self.args.device
        input_ids = self.tokenizer.encode(prompt, out_type=int, add_bos=True, add_eos=False)
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        # Helper to deep copy all layer caches
        def get_model_cache():
            return [
                (layer.attention.cache_k.clone(), layer.attention.cache_v.clone())
                for layer in self.model.layers
            ]

        def set_model_cache(cache):
            for layer, (k, v) in zip(self.model.layers, cache):
                layer.attention.cache_k.copy_(k)
                layer.attention.cache_v.copy_(v)

        # Fill the KV cache with the full prompt, as in text_completion
        for i in range(len(input_ids)):
            with torch.no_grad():
                _ = self.model.forward(input_tensor[:, i:i+1], i)
        base_cache = get_model_cache()

        # Each beam: (tokens, logprob, cache)
        beams = [(input_tensor.clone(), 0.0, base_cache)]
        completed = []
        ngram_size = 3  # No-repeat n-gram size
        for step in range(max_gen_len):
            candidates = []
            for tokens, score, cache in beams:
                # Processing of each beam
                set_model_cache(cache)
                with torch.no_grad():
                    logits = self.model.forward(tokens[:, -1:], tokens.shape[1]-1)
                    log_probs = torch.log_softmax(logits[:, -1], dim=-1)  # (1, vocab_size)
                # No-repeat n-gram constraint
                token_list = tokens.squeeze(0).tolist()
                if len(token_list) >= ngram_size - 1:
                    ngram_prefix = tuple(token_list[-(ngram_size-1):])
                    ngram_set = set()
                    for i in range(len(token_list) - ngram_size + 1):
                        ngram = tuple(token_list[i:i+ngram_size])
                        ngram_set.add(ngram)
                    for idx in range(log_probs.shape[-1]):
                        potential_ngram = ngram_prefix + (idx,)
                        if potential_ngram in ngram_set:
                            log_probs[0, idx] = float('-inf')
                top_log_probs, top_indices = torch.topk(log_probs, beam_size, dim=-1)
                for i in range(beam_size):
                    next_token = top_indices[0, i].unsqueeze(0).unsqueeze(0)  # shape (1,1)
                    new_tokens = torch.cat([tokens, next_token], dim=1)
                    new_score = score + top_log_probs[0, i].item()
                    # Save cache after this step
                    new_cache = get_model_cache()
                    # If EOS, add to completed
                    if next_token.item() == self.tokenizer.eos_id():
                        completed.append((new_tokens, new_score))
                    else:
                        candidates.append((new_tokens, new_score, new_cache))
            if not candidates:
                break
            # Keep top beams
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]
            beams = candidates
        # Add remaining beams
        completed.extend([(tokens, score) for tokens, score, _ in beams])
        completed = sorted(completed, key=lambda x: x[1], reverse=True)
        results = []
        base_prompt_len = len(input_ids)
        bos_id = self.tokenizer.bos_id() if hasattr(self.tokenizer, 'bos_id') else None
        for tokens, score in completed:
            token_list = tokens.squeeze(0).tolist()
            prompt_len_adjusted = base_prompt_len
            # Remove BOS token if present at the start (for display only)
            if bos_id is not None and len(token_list) > 0 and token_list[0] == bos_id:
                token_list = token_list[1:]
                prompt_len_adjusted = base_prompt_len - 1  # Adjust for removed BOS
            # Cut at EOS if present
            if self.tokenizer.eos_id() in token_list:
                eos_idx = token_list.index(self.tokenizer.eos_id())
                token_list = token_list[:eos_idx]
            # Always slice off the original prompt length
            gen_tokens = token_list[prompt_len_adjusted:] if len(token_list) > prompt_len_adjusted else []
            gen_text = self.tokenizer.decode(gen_tokens)
            # Output is prompt + generated text
            results.append((prompt + gen_text, score))
        return results

if __name__ == '__main__':
    torch.manual_seed(0)

    allow_cuda = True
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'
    print(f'Using device: {device}')

    prompts  = ["India is a beautiful country but "]

    model = LLaMA.build(
        checkpoints_dir='llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=512,
        max_batch_size=1,
        device=device
    )

    sampling_strategy = 'random_sampling'  # Change to 'beam_search', 'top_p', 'top_k', 'random_sampling', or 'greedy' as needed
    print(f"Using sampling strategy: {sampling_strategy}")

    if sampling_strategy == 'beam_search':
        print("Running beam search for prompt:")
        for prompt in prompts:
            print(prompt)
            results = model.beam_search(prompt, max_gen_len=64, beam_size=2)
            print("\nBeam search results (top 2):")
            for i, (text, score) in enumerate(results[:2]):
                print(f"Beam {i+1}: Score={score:.2f}\n{text}\n{'-'*50}")
    else:
        for prompt in prompts:
            out_tokens, out_texts = model.text_completion([prompt], max_gen_len=128,sampling_strategy=sampling_strategy)
            print(f'{out_texts[0]}')
            print('-' * 50)