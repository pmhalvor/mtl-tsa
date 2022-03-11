# Author: Fabio Rodrigues Pereira
# E-mail: fabior@uio.no

# Author: Per Morten Halvorsen
# E-mail: pmhalvor@uio.no

# Author: Eivind Grønlie Guren
# E-mail: eivindgg@ifi.uio.no


from utils.preprocessing import OurDataset, pad_b
from utils.models import Transformer, TransformerMTL

from torch.utils.data import DataLoader
import torch


# NORBERT = 'ltgoslo/norbert'
NORBERT = 'data/216'

train_file = 'data/train.conll'
dev_file = 'data/dev.conll'
test_file = 'data/test.conll'


train_dataset = OurDataset(
    data_file=train_file,
    specify_y='BIO',
    NORBERT_path=NORBERT,
    tokenizer=None
)

# x_ds, y_ds, att_ds = next(iter(train_dataset))
# sentence_tk_ds = train_dataset.tokenizer.convert_ids_to_tokens(x_ds)
# sentence_ds = train_dataset.tokenizer.decode(x_ds)

dev_dataset = OurDataset(
    data_file=dev_file,
    specify_y='BIO',
    NORBERT_path=None,
    tokenizer=train_dataset.tokenizer
)

test_dataset = OurDataset(
    data_file=test_file,
    specify_y='BIO',
    NORBERT_path=None,
    tokenizer=train_dataset.tokenizer
)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda batch: pad_b(batch=batch,
                                   IGNORE_ID=train_dataset.IGNORE_ID)
)

# x, y, att = next(iter(train_loader))

dev_loader = DataLoader(
    dataset=dev_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda batch: pad_b(batch=batch,
                                   IGNORE_ID=train_dataset.IGNORE_ID)
)

# x1, y1, att1 = next(iter(dev_loader))

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=lambda batch: pad_b(batch=batch,
                                   IGNORE_ID=train_dataset.IGNORE_ID)
)

# x2, y2, att2 = next(iter(test_loader))

model_bio = Transformer(
    NORBERT=NORBERT,
    tokenizer=train_dataset.tokenizer,
    num_labels=3,
    IGNORE_ID=train_dataset.IGNORE_ID,
    device="cuda" if torch.cuda.is_available() else "cpu",
    epochs=10,  # best is 2
    lr_scheduler=False,
    factor=0.1,
    lrs_patience=2,
    loss_funct='cross-entropy',
    random_state=1,
    verbose=True,
    lr=0.0001,
    momentum=0.9,
    epoch_patience=1,
    label_indexer=None,
    optmizer='AdamW'
)

model_bio.fit(
    train_loader=train_loader,
    verbose=True,
    dev_loader=dev_loader
)

binary_f1, propor_f1 = model_bio.evaluate(test_loader)
torch.save(model_bio, "data/transformer_bio.pt")
