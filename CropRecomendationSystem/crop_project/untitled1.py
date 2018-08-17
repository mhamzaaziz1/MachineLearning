##import pandas as pd
#import numpy as np
#import pandas as pd
#import sys 
#from tkinter import * 
#
#dates = pd.date_range('20160101', periods=6)
#df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=list('ABCD'))
#
#root = Tk() 
#
#t1 = Text(root) 
#t1.pack() 
#
#class PrintToT1(object): 
# def write(self, s): 
#     t1.insert(END, s) 
#
#sys.stdout = PrintToT1() 
#print (df)

#mainloop() 




import pandas as pd
df=pd.read_csv("C:\Users\Anderson\Music\UNSW_NB15-master/NUSW-NB15_features.csv")





#from tkinter import *
#
#master = Tk()
#
#def callback():
#    print ("click!")
#
#b = Button(master, text="OK", command=callback)
#b.pack()
#
#mainloop()