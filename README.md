This is LLaMA 2 implemented from scratch in PyTorch inspired from Umar Jamil with some improvements

**Major Update**
1. Added different sampling technique like beam search, random sampling and top k apart from greedy and top p which were already present in the repo
2. Added N-gram repetition constrain to avoid the repetitive generation

**Minor Update**
1. Decoupled batch dimension to make the use of batching more efficient and add more freedom when dealing with batching 
