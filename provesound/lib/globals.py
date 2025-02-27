import time

# The file contains global variables that are used to keep track of the time taken by the
# ProveSound algorithm.

start_time = None 
verification_time = 0
generation_time = 0

def update_start_time():
    global start_time
    start_time = time.time()

def update_verification_time():
    global verification_time
    verification_time += time.time() - start_time
    update_start_time()

def update_generation_time():
    global generation_time
    generation_time += time.time() - start_time
    update_start_time()

def reset_time():
    global verification_time
    global generation_time
    verification_time = 0
    generation_time = 0

def get_verification_time():
    global verification_time
    return verification_time    

def get_generation_time():
    global generation_time
    return generation_time