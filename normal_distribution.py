from math import *

mean=2
var=0.04
y=0.3

for i in xrange(1,10):
    y=i*0.1
    
    ln_in = sqrt(2*var*pi)*y
    sqrt_in = -log(ln_in)*2*var
    x=sqrt(sqrt_in)+mean

    print y,x