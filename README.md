1. Instantiate a .venv virtual environment

2. 
```shell
pip install -r requirements.txt
```

3. fastapi dev src/main.py

4. Head to http://localhost:8000/docs to make the requests



To see the ids of the batches/ensembles. Use 
1. `sqlite3 srma.db` 
2. `.tables`
3. `select * from batch_ids;`