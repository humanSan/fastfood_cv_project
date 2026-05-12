(venv) PS C:\Users\axd210123\Downloads\project> python train_segformer.py
Loading FoodSeg103 dataset from HuggingFace …
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
[transformers] You passed `num_labels=104` which is incompatible to the `id2label` map of length `150`.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 208/208 [00:00<00:00, 5912.68it/s] 
[transformers] SegformerForSemanticSegmentation LOAD REPORT from: nvidia/segformer-b0-finetuned-ade-512-512
Key                           | Status   |
------------------------------+----------+-------------------------------------------------------------------------------------------------------
decode_head.classifier.bias   | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([150]) vs model:torch.Size([104])
decode_head.classifier.weight | MISMATCH | Reinit due to size mismatch - ckpt: torch.Size([150, 256, 1, 1]) vs model:torch.Size([104, 256, 1, 1])

Notes:
- MISMATCH:     ckpt weights were loaded, but they did not match the original empty weight shapes.
  Train samples: 4983
  Val samples:   2135  (split: 'validation')
  Num classes:   104

Starting training on cuda for 10 epochs …
  Optimizer: AdamW (lr=6e-05, weight_decay=0.01)
  Scheduler: CosineAnnealingLR → 12460 steps
Epoch 1/10 [Train]:  13%|█████████▏                                                           | 167/1246 [01:53<11:31,  1.56it/s, loss=3.2113, lr=6.00e-05] 