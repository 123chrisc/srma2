# SRMA 2

## Setup

REMEMBER TO UPDATE THE .env file

1. 
```bash
python3 -m venv .venv
```

2. 
```bash
source .venv/bin/activate
```

3. 
```bash
pip install -r requirements.txt
```

4. 
```bash
fastapi dev src/main.py
```

5. Head to http://localhost:8000/docs to make the requests and see the responses


## Debugging
To see the ids of the batches/ensembles. Use 

1. 
```
sqlite3 srma.db
``` 
2. 
```
.tables
```
3. 
```
select * from batch_ids;
```