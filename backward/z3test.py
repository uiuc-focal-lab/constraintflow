from z3 import *
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Timeout reached.")

timeout_duration = 3
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(timeout_duration)


s = Solver()
#s.set("timeout", 3000)
x = Int('x')
y = Int('y')
z = Int('z')
p = (x * x * x)+(y*y*y) == (z*z*z)
s.add(p)
s.add(x > 5)
s.add(y > 5)
s.add(z > 5)
try:
    import time
    i=1
    while(i<2):
        i = i+1
        print('l')
        time.sleep(1)
    c = s.check()
    #m = s.model()
    #signal.alarm(0)
except TimeoutError:
    print("timeout reached")
