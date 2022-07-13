# dialogue_qa
Using QA PLM for event argument extraction


```
python .\src\model\finetune_bert.py

python .\src\model\evaluate-v2.0.py '.\results\gt\eval.json' '.\results\pred\0epoch\predictions.json' --out-file '.\results\pred\0epoch\results.json'
```