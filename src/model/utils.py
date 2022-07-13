import torch


class LivesafeDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, idx):
        return self.examples[idx]

    def __len__(self):
        return len(self.examples)



def offset2lineNum(filePath, offStart):
    with open(filePath, 'r', newline='', encoding='utf8') as f:  
#         print(f.read()[:off_start])
#         print(f.read()[:off_start].count('\n'))
        return f.read()[:offStart].count('\n')