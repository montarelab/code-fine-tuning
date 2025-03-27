# Plan

## Step 1. Find required repo

https://github.com/search?q=size%3A10000+language%3APython&type=Repositories&ref=advsearch&l=Python&l=

# File 1 structure

1. Imports and logger setup

   - Standard imports (atexit, numpy, queue, torch, multiprocessing)
   - Custom imports from slowfast package

2. AsycnActionPredictor class

   - Inner \_Predictor class (implements mp.Process)

     - **init** method - initializes predictor with config, queues and GPU ID
     - run method - creates model and processes tasks from queue

   - Main class methods:
     - **init** - Creates worker processes
     - put - Adds tasks to queue
     - get - Retrieves results in the correct order
     - **call** - Convenience method to put and get
     - shutdown - Terminates worker processes
     - Properties: result_available, default_buffer_size

3. AsyncVis class

   - Inner \_VisWorker class (implements mp.Process)

     - Handles visualization tasks asynchronously

   - Main class methods:
     - **init** - Sets up visualization workers
     - put/get methods - For task queue management
     - **call**, shutdown and properties

4. \_StopToken class (simple signal class)

5. AsyncDemo class

   - Combines prediction and visualization pipeline
   - Methods for putting tasks and getting results

6. draw_predictions function
   - Handles the actual visualization of predictions
   - Contains debugging code in a "hotfix" section

## Step 2. Preprocess it

Examples of benchmarks
![alt text](image.png)

## Step 3. Find tune the model

## Step 4. Estimate by benchmarks
