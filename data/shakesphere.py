"""
data/datasets/shakespeare.py

Dataset utility for the tiny-Shakespeare pretraining corpus.

Responsibilities:
    - Read raw text from disk
    - Encode the full corpus into a flat 1D tensor of token IDs
      using a pre-trained tokenizer passed in from outside
    - Split into train / val token tensors
    - Expose get_batch() to sample random (input_ids, target_ids)
      windows on demand during training

Usage (from train.py):
    tokenizer = BPETokenizer.load(...)
    dataset = ShakespeareDataset(path="shakespeare.txt", tokenizer=tokenizer)
    x, y = dataset.get_batch(split="train", batch_size=32, seq_len=128)
"""
import torch

class ShakespeareDataset:
    def __init__(self, path, tokenizer, test_split=0.1, val_split=0.1, device="cpu"):
        """
        Args:
            path       : path to the raw .txt corpus file
            tokenizer  : an already-trained tokenizer with an .encode() method
            val_split  : fraction of tokens to hold out for validation (default 10%)
            device     : torch device to put the token tensors on
        """
        self.tokenizer = tokenizer
        self.device = device

        # TODO: Step 1 -- read raw text from `path`
        # plain open() + read() is sufficient, no special parsing needed
        # store result as a local variable, you won't need to keep the raw
        # text around after encoding
        raw_txt = open(path, "r").read()

        # TODO: Step 2 -- encode the full corpus text into a flat list/tensor
        # of integer token IDs using self.tokenizer.encode(raw_text)
        # convert to a 1D torch.LongTensor (dtype=torch.long, required by
        # nn.Embedding which expects integer indices)
        # store as a local variable `token_ids` for the split step below
        token_ids = torch.tensor(tokenizer.encode(raw_txt), dtype=torch.long)

        # TODO: Step 3 -- split token_ids into train and val tensors
        # compute split index: n = int(len(token_ids) * (1 - val_split))
        # everything up to n -> self.train_ids
        # everything from n onward -> self.val_ids
        # both should be 1D LongTensors on self.device
        # e.g. 80% train, 10% test, 10% val
        i_test_end = int(len(token_ids) * (1 - val_split))
        i_train_end  = int(len(token_ids) * (1 - val_split - test_split))

        self.train_ids = token_ids[:i_train_end]
        self.test_ids  = token_ids[i_train_end:i_test_end]
        self.val_ids   = token_ids[i_test_end:]

        # TODO: after the split, print a small summary so you can verify
        # things look right when you first run this:
        #   - total tokens in corpus
        #   - tokens in train split
        #   - tokens in val split
        #   - a sanity-check: decode the first ~50 tokens and print them,
        #     so you can confirm the tokenizer round-tripped correctly on
        #     the real corpus (not just your toy test strings)
        print(f'Total dataset length :: ${len(token_ids)}')
        print(f'Train split :: ${len(self.train_ids)}')
        print(f'Test split :: ${len(self.test_ids)}')
        print(f'Validation split :: ${len(self.val_ids)}')
        print(f'First 50 tokens decoded: {self.tokenizer.decode(self.train_ids[:50].tolist())}')

        return


    def get_batch(self, split, batch_size, seq_len):
        """
        Sample a random batch of (input_ids, target_ids) windows.

        Args:
            split      : "train" or "val"
            batch_size : number of independent sequences per batch (B)
            seq_len    : context window length (T)

        Returns:
            input_ids  : LongTensor of shape (B, T)
            target_ids : LongTensor of shape (B, T)
                         same as input_ids but shifted one position forward --
                         target_ids[b, t] is the token the model should predict
                         after seeing input_ids[b, 0:t+1]
        """
        if split not in ["train", "test", "val"]:
            raise ValueError('Split should be one of ["train", "test", "val"]')

        data = None
        if split == "train":
            data = self.train_ids
        elif split == "test":
            data = self.test_ids
        else: 
            data = self.val_ids

        # TODO: sample batch_size random starting positions
        # each starting position `i` must satisfy: i + seq_len + 1 <= len(data)
        # (the +1 is because target needs one extra token beyond the input window)
        # use torch.randint to sample all B starting positions at once
        # store as a 1D tensor `start_positions` of shape (B,)
        idxs = torch.randint(0, len(data) - seq_len, (batch_size,))

        # TODO: slice out input and target windows for each starting position
        # input_ids[b]  = data[start_positions[b] : start_positions[b] + seq_len]
        # target_ids[b] = data[start_positions[b] + 1 : start_positions[b] + seq_len + 1]
        # stack B slices into tensors of shape (B, T) using torch.stack
        input_slice = torch.stack([data[i:i+seq_len] for i in idxs])
        target_slice = torch.stack([data[i+1:i+seq_len+1] for i in idxs])

        # TODO: move both tensors to self.device and return them
        return input_slice.to(self.device),target_slice.to(self.device)