import re

import numpy as np

from transformers import AutoTokenizer, AutoModel, pipeline


class EmbedProt:
    def __init__(self):
        super(EmbedProt, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Rostlab/prot_bert_bfd", do_lower_case=False
        )
        self.model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")

    def __call__(self, proteins, device=0):
        fe = pipeline(
            "feature-extraction",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )
        seqs = [" ".join(list(x)) for x in proteins]
        seqs = [re.sub(r"[UZOB]", "X", sequence) for sequence in seqs]
        embs = []
        for s in seqs:
            emb = np.array(fe([s])[0])  # (n, 1024)
            cls = emb[0]
            rest = emb[1:].mean(0)
            embs.append(np.concatenate([cls, rest]))
        return embs
