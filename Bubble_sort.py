# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:55:29 2016

@author: darren
"""

import random
import time
from operator import itemgetter

number=5
save_list=[]

#create random list
for i in range(0,number):    
    random_number1= random.random()
    random_number2= random.random()
    
    save_list.append((random_number1,random_number2))
    
print "The lists before sorting:"
for i in range(0,number):
    print save_list[i]
    
before=time.time()
#sort
save_list.sort(key=itemgetter(1))
another_list=sorted(save_list,key=itemgetter(1))

after=time.time()

print "\n"+"After:"
for i in range(0,number):
    print another_list[i]
    
print "\n"+"time_use:",after-before

#for i in range(0,number):
#    for j in range(0,number-1-i):
#        if(save_list[j]>save_list[j+1]):
#            temp=save_list[j]
#            save_list[j+1]=save_list[j]
#            save_list[j]=temp