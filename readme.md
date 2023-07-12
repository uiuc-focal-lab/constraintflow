Dependencies:
python
z3

Run Experiments:
```
python run_paperexp.py 
```

Run One Certifier:
```
python paper_parse.py deeppoly_affine tempout/deeppoly_affine
python experiments.py test_cases_correct/deeppoly_relu "[nprev] [nsym] 1" > tempout/deeppoly_relu
```

