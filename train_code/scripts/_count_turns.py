import json
for fname in ['data/tool_calling_train.jsonl', 'data/tool_calling_val.jsonl']:
    lens = []
    for line in open(fname):
        ex = json.loads(line)
        turns = len([m for m in ex['messages'] if m['role'] in ('user','assistant')])
        lens.append(turns)
    short = sum(1 for l in lens if l <= 4)
    long_ = sum(1 for l in lens if l > 4)
    print(f'{fname}: {len(lens)} examples, short(<=4)={short}, long(>4)={long_}, max={max(lens)}')
