```

################## Step 1 ##################
python preprocessor.py

# or
python preprocessor.py --folder path/to/proto




################## Step 2 ##################
python embeddings.py

# or 
python embeddings.py --input-folder path/to/json --persist-dir path/to/vectorstore



################## Step 3 ##################
python query.py

# Or specify a custom query
python query.py --query "Find all services in proto_v2.json"

# Specify number of results
python query.py --num-results 5

# Use different vector store location
python query.py --vectorstore-dir path/to/vectorstore
```