Starting training on cuda for 12 epochs …
  Optimizer: AdamW (lr=6e-05, weight_decay=0.01)
  Scheduler: CosineAnnealingLR → 7476 steps
Epoch 1/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:20<00:00,  1.72s/it, loss=2.3367, lr=5.90e-05]Exception ignored in: <function ResourceTracker.__del__ at 0x00000180C086F380>
Exception ignored in: <function ResourceTracker.__del__ at 0x000002025D0DFC40>
Traceback (most recent call last):
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Epoch 1/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:21<00:00,  1.58s/it, loss=2.3367, lr=5.90e-05]
  Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 267/267 [03:42<00:00,  1.49it/s]Exception ignored in: <function ResourceTracker.__del__ at 0x000001E38304FBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Exception ignored in: <function ResourceTracker.__del__ at 0x000002945E9FFBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
  Epoch  1 │ Train Loss: 2.5174 │ Val Loss: 1.8761 │ mIoU: 2.98% │ Pixel Acc: 58.34% │ Time: 1205s
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 48.50it/s]
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 44.78it/s] 
  ★ New best mIoU: 2.98% → saved to weights\segformer_foodseg103_best
Epoch 2/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:21<00:00,  1.51s/it, loss=1.6703, lr=5.60e-05]Exception ignored in: <function ResourceTracker.__del__ at 0x00000248CCB7FBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Exception ignored in: <function ResourceTracker.__del__ at 0x00000254D92BF380>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Epoch 2/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:22<00:00,  1.58s/it, loss=1.6703, lr=5.60e-05]
  Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 267/267 [03:28<00:00,  1.51it/s]Exception ignored in: <function ResourceTracker.__del__ at 0x000001F3408BFBA0>
Traceback (most recent call last):
Exception ignored in: <function ResourceTracker.__del__ at 0x000001F6C089FBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
  Epoch  2 │ Train Loss: 1.7003 │ Val Loss: 1.5176 │ mIoU: 6.09% │ Pixel Acc: 64.06% │ Time: 1193s
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 50.98it/s]
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 46.08it/s] 
  ★ New best mIoU: 6.09% → saved to weights\segformer_foodseg103_best
Epoch 3/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:47<00:00,  1.50s/it, loss=1.1118, lr=5.14e-05]Exception ignored in: <function ResourceTracker.__del__ at 0x000001F5A8C1F380>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Exception ignored in: <function ResourceTracker.__del__ at 0x000002180304FBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Epoch 3/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:48<00:00,  1.62s/it, loss=1.1118, lr=5.14e-05]
  Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 267/267 [03:47<00:00,  1.50it/s]Exception ignored in: <function ResourceTracker.__del__ at 0x0000013BB676FBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Exception ignored in: <function ResourceTracker.__del__ at 0x00000144CB40F380>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
  Epoch  3 │ Train Loss: 1.4229 │ Val Loss: 1.3282 │ mIoU: 8.52% │ Pixel Acc: 67.73% │ Time: 1237s
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 42.42it/s] 
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 42.08it/s] 
  ★ New best mIoU: 8.52% → saved to weights\segformer_foodseg103_best
  ★ New best mIoU: 8.52% → saved to weights\segformer_foodseg103_best
Epoch 4/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:22<00:00,  1.50s/it, loss=0.7671, lr=4.53e-05]EEpoch 4/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:22<00:00,  1.50s/it, loss=0.7671, lr=4.53e-05]Exception ignored in: <function ResourceTracker.__del__ at 0x00000211A646F920>
Exception ignored in: <function ResourceTracker.__del__ at 0x0000027117E7FBA0>
Exception ignored in: <function ResourceTracker.__del__ at 0x0000027117E7FBA0>
Traceback (most recent call last):
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Epoch 4/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:23<00:00,  1.58s/it, loss=0.7671, lr=4.53e-05] 
  Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 267/267 [03:37<00:00,  1.49it/s]Exception ignored in: <function ResourceTracker.__del__ at 0x00000265BCB0F380>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Exception ignored in: <function ResourceTracker.__del__ at 0x000002101F9EFC40>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
  Epoch  4 │ Train Loss: 1.2534 │ Val Loss: 1.2159 │ mIoU: 10.35% │ Pixel Acc: 69.39% │ Time: 1202s
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 52.30it/s] 
Writing model shards: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 142.47it/s] 
  ★ New best mIoU: 10.35% → saved to weights\segformer_foodseg103_best
Epoch 5/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:23<00:00,  1.50s/it, loss=1.3693, lr=3.81e-05]Exception ignored in: <function ResourceTracker.__del__ at 0x000002A369ACFBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Exception ignored in: <function ResourceTracker.__del__ at 0x000001AC5DFCFBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Epoch 5/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:24<00:00,  1.58s/it, loss=1.3693, lr=3.81e-05]
  Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 267/267 [03:37<00:00,  1.49it/s]Exception ignored in: <function ResourceTracker.__del__ at 0x0000018CE94AF920>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
Exception ignored in: <function ResourceTracker.__del__ at 0x000001ABC003F380>
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
  Epoch  5 │ Train Loss: 1.1301 │ Val Loss: 1.1480 │ mIoU: 12.06% │ Pixel Acc: 70.41% │ Time: 1202s
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 91.94it/s] 
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 46.55it/s]
  ★ New best mIoU: 12.06% → saved to weights\segformer_foodseg103_best
Epoch 6/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:49<00:00,  1.51s/it, loss=1.3424, lr=3.05e-05]Exception ignored in: <function ResourceTracker.__del__ at 0x0000027A5074FBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Exception ignored in: <function ResourceTracker.__del__ at 0x000002438C87F380>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Epoch 6/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:49<00:00,  1.62s/it, loss=1.3424, lr=3.05e-05]
  Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 267/267 [03:12<00:00,  1.50it/s]Exception ignored in: <function ResourceTracker.__del__ at 0x000001B5ABA8FBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Exception ignored in: <function ResourceTracker.__del__ at 0x00000218DAC8F380>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
  Epoch  6 │ Train Loss: 1.0273 │ Val Loss: 1.1234 │ mIoU: 13.48% │ Pixel Acc: 70.85% │ Time: 1203s
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 46.05it/s] 
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 52.94it/s] 
  ★ New best mIoU: 13.48% → saved to weights\segformer_foodseg103_best
Epoch 7/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:22<00:00,  1.51s/it, loss=0.9908, lr=2.29e-05]Exception ignored in: <function ResourceTracker.__del__ at 0x000002999D8EFBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
Exception ignored in: <function ResourceTracker.__del__ at 0x0000014FC0A0FBA0>
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Epoch 7/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:23<00:00,  1.58s/it, loss=0.9908, lr=2.29e-05]
  Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 267/267 [03:41<00:00,  1.48it/s]Exception ignored in: <function ResourceTracker.__del__ at 0x000001EAC7B2FBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
Exception ignored in: <function ResourceTracker.__del__ at 0x00000143C29AFC40>
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
  Epoch  7 │ Train Loss: 0.9571 │ Val Loss: 1.0835 │ mIoU: 14.56% │ Pixel Acc: 71.86% │ Time: 1205s
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 46.46it/s] 
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 78.22it/s] 
  ★ New best mIoU: 14.56% → saved to weights\segformer_foodseg103_best
Epoch 8/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:24<00:00,  1.50s/it, loss=0.8134, lr=1.57e-05]Exception ignored in: <function ResourceTracker.__del__ at 0x00000259C08BFBA0>
Exception ignored in: <function ResourceTracker.__del__ at 0x00000190C088FBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Epoch 8/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:25<00:00,  1.58s/it, loss=0.8134, lr=1.57e-05]
  Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 267/267 [03:43<00:00,  1.49it/s]Exception ignored in: <function ResourceTracker.__del__ at 0x0000023CC34CFBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Exception ignored in: <function ResourceTracker.__del__ at 0x0000021FDD69F920>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
  Epoch  8 │ Train Loss: 0.9062 │ Val Loss: 1.0695 │ mIoU: 15.20% │ Pixel Acc: 72.33% │ Time: 1210s
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 44.96it/s] 
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 61.99it/s]
  ★ New best mIoU: 15.20% → saved to weights\segformer_foodseg103_best
Epoch 9/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:22<00:00,  1.50s/it, loss=0.7500, lr=9.64e-06]Exception ignored in: <function ResourceTracker.__del__ at 0x00000179408DFBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Exception ignored in: <function ResourceTracker.__del__ at 0x00000229B737F380>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Epoch 9/12 [Train]: 100%|██████████████████████████████████████████████████████████████████████| 623/623 [16:23<00:00,  1.58s/it, loss=0.7500, lr=9.64e-06]
  Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 267/267 [03:51<00:00,  1.52it/s]Exception ignored in: <function ResourceTracker.__del__ at 0x000002BA10A1FBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Exception ignored in: <function ResourceTracker.__del__ at 0x000001FBB91DFBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
  Epoch  9 │ Train Loss: 0.8612 │ Val Loss: 1.0643 │ mIoU: 15.43% │ Pixel Acc: 72.33% │ Time: 1215s
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 48.49it/s]
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 48.61it/s]
  ★ New best mIoU: 15.43% → saved to weights\segformer_foodseg103_best
Epoch 10/12 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 623/623 [16:19<00:00,  1.49s/it, loss=0.6278, lr=4.95e-06]Exception ignored in: <function ResourceTracker.__del__ at 0x0000026731CCF380>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Exception ignored in: <function ResourceTracker.__del__ at 0x000001831886FBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Epoch 10/12 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 623/623 [16:20<00:00,  1.57s/it, loss=0.6278, lr=4.95e-06]
  Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 267/267 [03:42<00:00,  1.52it/s]Exception ignored in: <function ResourceTracker.__del__ at 0x000001C0CCB6F380>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Exception ignored in: <function ResourceTracker.__del__ at 0x0000020F6624FBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
  Epoch 10 │ Train Loss: 0.8321 │ Val Loss: 1.0597 │ mIoU: 15.66% │ Pixel Acc: 72.44% │ Time: 1203s
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 56.74it/s]
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 48.60it/s] 
  ★ New best mIoU: 15.66% → saved to weights\segformer_foodseg103_best
Epoch 11/12 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 623/623 [16:20<00:00,  1.50s/it, loss=0.5551, lr=2.01e-06]Exception ignored in: <function ResourceTracker.__del__ at 0x0000027F4512FBA0>
Traceback (most recent call last):
Exception ignored in: <function ResourceTracker.__del__ at 0x0000027DC08FFBA0>
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Epoch 11/12 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 623/623 [16:21<00:00,  1.58s/it, loss=0.5551, lr=2.01e-06]
  Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 267/267 [03:10<00:00,  1.49it/s]Exception ignored in: <function ResourceTracker.__del__ at 0x000002542F567A60>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Exception ignored in: <function ResourceTracker.__del__ at 0x000001A3E0C6FA60>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
  Epoch 11 │ Train Loss: 0.8177 │ Val Loss: 1.0578 │ mIoU: 15.72% │ Pixel Acc: 72.53% │ Time: 1173s
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 43.36it/s] 
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 34.73it/s] 
  ★ New best mIoU: 15.72% → saved to weights\segformer_foodseg103_best
Epoch 12/12 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 623/623 [16:20<00:00,  1.49s/it, loss=0.5971, lr=1.00e-06]Exception ignored in: <function ResourceTracker.__del__ at 0x00000254315CFBA0>
Traceback (most recent call last):
Exception ignored in: <function ResourceTracker.__del__ at 0x00000267BB59FBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
Epoch 12/12 [Train]: 100%|█████████████████████████████████████████████████████████████████████| 623/623 [16:20<00:00,  1.57s/it, loss=0.5971, lr=1.00e-06]
  Validating: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 267/267 [03:42<00:00,  1.52it/s]Exception ignored in: <function ResourceTracker.__del__ at 0x000001C4522FFBA0>
Exception ignored in: <function ResourceTracker.__del__ at 0x00000191A2D1FBA0>
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
Traceback (most recent call last):
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 80, in __del__
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 89, in _stop
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
  File "C:\Users\axd210123\Downloads\project\venv\Lib\site-packages\multiprocess\resource_tracker.py", line 102, in _stop_locked
AttributeError: '_thread.RLock' object has no attribute '_recursion_count'
  Epoch 12 │ Train Loss: 0.8090 │ Val Loss: 1.0551 │ mIoU: 15.85% │ Pixel Acc: 72.63% │ Time: 1204s
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 44.26it/s]
Writing model shards: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 46.00it/s]
  ★ New best mIoU: 15.85% → saved to weights\segformer_foodseg103_best

Training complete in 240.9 minutes.