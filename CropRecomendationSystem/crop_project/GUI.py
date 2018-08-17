# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#
#! /usr/bin/env python
#
# GUI module generated by PAGE version 4.12
# In conjunction with Tcl version 8.6
#    Apr 27, 2018 11:39:25 AM

import sys

try:
    from Tkinter import *
except ImportError:
    from tkinter import *

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

import GUI_support
from working_file import UNSWNB15 as aa


def vp_start_gui():
    '''Starting point when module is the main routine.'''
    global val, w, root
    root = Tk()
    top = New_Toplevel (root)
    GUI_support.init(root, top)
    root.mainloop()

w = None
def create_New_Toplevel(root, *args, **kwargs):
    '''Starting point when module is imported by another program.'''
    global w, w_win, rt
    rt = root
    w = Toplevel (root)
    top = New_Toplevel (w)
    GUI_support.init(w, top, *args, **kwargs)
    return (w, top)

def destroy_New_Toplevel():
    global w
    w.destroy()
    w = None




def callback():
    print ("click!")

class New_Toplevel():
    def __init__(self, top=None):
        '''This class configures and populates the toplevel window.
           top is the toplevel containing window.'''
        _bgcolor = '#d9d9d9'  # X11 color: 'gray85'
        _fgcolor = '#000000'  # X11 color: 'black'
        _compcolor = '#d9d9d9' # X11 color: 'gray85'
        _ana1color = '#d9d9d9' # X11 color: 'gray85' 
        _ana2color = '#d9d9d9' # X11 color: 'gray85' 
        font11 = "-family {Segoe UI Semibold} -size 14 -weight bold "  \
            "-slant italic  -overstrike 0"
        font9 = "-family {Segoe UI Black} -size 14 -weight bold -slant"  \
            " roman -underline 0 -overstrike 0"

        top.geometry("800x450+438+108")
        top.title("Intrusion Detection System")
        top.configure(background="#d9d9d9")

        self.Proceed = Button(top, command=aa.ShowDescription)
        self.Proceed.place(relx=0.3, rely=0.03, height=24, width=287)
        self.Proceed.configure(activebackground="#d9d9d9")
        self.Proceed.configure(activeforeground="#000000")
        self.Proceed.configure(background="#d9d9d9")
        self.Proceed.configure(disabledforeground="#a3a3a3")
        self.Proceed.configure(foreground="#000000")
        self.Proceed.configure(highlightbackground="#d9d9d9")
        self.Proceed.configure(highlightcolor="black")
        self.Proceed.configure(pady="0")
        self.Proceed.configure(text='''Data First Look''')
        self.Proceed.configure(width=187)
        
        self.Steps = Label(root,text="Problem Steps")
        self.Steps.place(relx=0.02, rely=0.13, height=61, width=154)
        self.Steps.configure(background="#d9d9d9")
        self.Steps.configure(disabledforeground="#a3a3a3")
        self.Steps.configure(font=font11)
        self.Steps.configure(foreground="#ff00ff")
        self.Steps.configure(text="Problem Steps")
        self.Steps.configure(width=154)

        self.PreProcessing = Button(top, command=aa.PreProcessing)
        self.PreProcessing.place(relx=0.02, rely=0.25, height=24, width=187)
        self.PreProcessing.configure(activebackground="#d9d9d9")
        self.PreProcessing.configure(activeforeground="#000000")
        self.PreProcessing.configure(background="#d9d9d9")
        self.PreProcessing.configure(disabledforeground="#a3a3a3")
        self.PreProcessing.configure(foreground="#000000")
        self.PreProcessing.configure(highlightbackground="#d9d9d9")
        self.PreProcessing.configure(highlightcolor="black")
        self.PreProcessing.configure(pady="0")
        self.PreProcessing.configure(text='''Pre-Processing''')
        self.PreProcessing.configure(width=187)
        

        self.FeatureSelection = Button(top, command=aa.RecursiveFeatureElimination)
        self.FeatureSelection.place(relx=0.02, rely=0.35, height=24, width=187)
        self.FeatureSelection.configure(activebackground="#d9d9d9")
        self.FeatureSelection.configure(activeforeground="#000000")
        self.FeatureSelection.configure(background="#d9d9d9")
        self.FeatureSelection.configure(disabledforeground="#a3a3a3")
        self.FeatureSelection.configure(foreground="#000000")
        self.FeatureSelection.configure(highlightbackground="#d9d9d9")
        self.FeatureSelection.configure(highlightcolor="black")
        self.FeatureSelection.configure(pady="0")
        self.FeatureSelection.configure(text='''Feature Selection''')
        self.FeatureSelection.configure(width=187)

        self.ModelBuild = Button(top)
        self.ModelBuild.place(relx=0.02, rely=0.45, height=24, width=187)
        self.ModelBuild.configure(activebackground="#d9d9d9")
        self.ModelBuild.configure(activeforeground="#000000")
        self.ModelBuild.configure(background="#d9d9d9")
        self.ModelBuild.configure(disabledforeground="#a3a3a3")
        self.ModelBuild.configure(foreground="#000000")
        self.ModelBuild.configure(highlightbackground="#d9d9d9")
        self.ModelBuild.configure(highlightcolor="black")
        self.ModelBuild.configure(pady="0")
        self.ModelBuild.configure(text='''Model Building''')
        self.ModelBuild.configure(width=187)

        self.Evaluation = Button(top)
        self.Evaluation.place(relx=0.02, rely=0.55, height=24, width=187)
        self.Evaluation.configure(activebackground="#d9d9d9")
        self.Evaluation.configure(activeforeground="#000000")
        self.Evaluation.configure(background="#d9d9d9")
        self.Evaluation.configure(disabledforeground="#a3a3a3")
        self.Evaluation.configure(foreground="#000000")
        self.Evaluation.configure(highlightbackground="#d9d9d9")
        self.Evaluation.configure(highlightcolor="black")
        self.Evaluation.configure(pady="0")
        self.Evaluation.configure(text='''Model Evaluation''')
        self.Evaluation.configure(width=187)

        self.CrossValidation = Button(top, command=callback)
        self.CrossValidation.place(relx=0.02, rely=0.65, height=24, width=187)
        self.CrossValidation.configure(activebackground="#d9d9d9")
        self.CrossValidation.configure(activeforeground="#000000")
        self.CrossValidation.configure(background="#d9d9d9")
        self.CrossValidation.configure(disabledforeground="#a3a3a3")
        self.CrossValidation.configure(foreground="#000000")
        self.CrossValidation.configure(highlightbackground="#d9d9d9")
        self.CrossValidation.configure(highlightcolor="black")
        self.CrossValidation.configure(pady="0")
        self.CrossValidation.configure(text='''Model Cross Validation''')
        self.CrossValidation.configure(width=187)

        self.StepGuide = Label(top)
        self.StepGuide.place(relx=0.42, rely=0.20, height=21, width=84)
        self.StepGuide.configure(activebackground="#f9f9f9")
        self.StepGuide.configure(activeforeground="black")
        self.StepGuide.configure(background="#d9d9d9")
        self.StepGuide.configure(disabledforeground="#a3a3a3")
        self.StepGuide.configure(foreground="#000000")
        self.StepGuide.configure(highlightbackground="#d9d9d9")
        self.StepGuide.configure(highlightcolor="black")
        self.StepGuide.configure(text='''Step Guide''')
        self.StepGuide.configure(width=84)


        self.Step1 = Label(top)
        self.Step1.place(relx=0.45, rely=0.25, height=21, width=384)
        self.Step1.configure(activebackground="#f9f9f9")
        self.Step1.configure(activeforeground="black")
        self.Step1.configure(background="#d9d9d9")
        self.Step1.configure(disabledforeground="#a3a3a3")
        self.Step1.configure(foreground="#000000")
        self.Step1.configure(highlightbackground="#d9d9d9")
        self.Step1.configure(highlightcolor="black")
        self.Step1.configure(text='''Step 1: Pre Processing involve removal of extra data and fill missing data''')
        self.Step1.configure(width=84)

        self.Step2 = Label()
        self.Step2.place(relx=0.45, rely=0.30, height=21, width=384)
        self.Step2.configure(activebackground="#f9f9f9")
        self.Step2.configure(activeforeground="black")
        self.Step2.configure(background="#d9d9d9")
        self.Step2.configure(disabledforeground="#a3a3a3")
        self.Step2.configure(foreground="#000000")
        self.Step2.configure(highlightbackground="#d9d9d9")
        self.Step2.configure(highlightcolor="black")
        self.Step2.configure(text='''Step 2: Select relevant feature to solve problem optimally''')
        self.Step2.configure(width=84)

        self.Step3 = Label(top)
        self.Step3.place(relx=0.52, rely=0.35, height=21, width=384)
        self.Step3.configure(activebackground="#f9f9f9")
        self.Step3.configure(activeforeground="black")
        self.Step3.configure(background="#d9d9d9")
        self.Step3.configure(disabledforeground="#a3a3a3")
        self.Step3.configure(foreground="#000000")
        self.Step3.configure(highlightbackground="#d9d9d9")
        self.Step3.configure(highlightcolor="black")
        self.Step3.configure(text='''Step 3: Build model using Random Forest Tree''')
        self.Step3.configure(width=84)

        self.Step4 = Label(top)
        self.Step4.place(relx=0.52, rely=0.40, height=21, width=384)
        self.Step4.configure(activebackground="#f9f9f9")
        self.Step4.configure(activeforeground="black")
        self.Step4.configure(background="#d9d9d9")
        self.Step4.configure(disabledforeground="#a3a3a3")
        self.Step4.configure(foreground="#000000")
        self.Step4.configure(highlightbackground="#d9d9d9")
        self.Step4.configure(highlightcolor="black")
        self.Step4.configure(text='''Step 4: Evaluation based on Accuracy''')
        self.Step4.configure(width=84)

        self.Step5 = Label(top)
        self.Step5.place(relx=0.52, rely=0.45, height=21, width=384)
        self.Step5.configure(activebackground="#f9f9f9")
        self.Step5.configure(activeforeground="black")
        self.Step5.configure(background="#d9d9d9")
        self.Step5.configure(disabledforeground="#a3a3a3")
        self.Step5.configure(foreground="#000000")
        self.Step5.configure(highlightbackground="#d9d9d9")
        self.Step5.configure(highlightcolor="black")
        self.Step5.configure(text='''Step 5: Cross Validate dataset''')
        self.Step5.configure(width=84)


        
        #self.Load = Button(top)
        #self.Load.place(relx=0.67, rely=0.1, height=24, width=187)
        #self.Load.configure(activebackground="#d9d9d9")
        #self.Load.configure(activeforeground="#000000")
        #self.Load.configure(background="#d9d9d9")
        #self.Load.configure(disabledforeground="#a3a3a3")
        #self.Load.configure(foreground="#000000")
        #self.Load.configure(highlightbackground="#d9d9d9")
        #self.Load.configure(highlightcolor="black")
        #self.Load.configure(pady="0")
        #self.Load.configure(text='''Load''')
        #self.Load.configure(width=187)

        
        
        self.recomendedcrop = Label(root,text="hello")
        self.recomendedcrop.place(relx=0.05, rely=0.83, height=61, width=254)
        self.recomendedcrop.configure(background="#d9d9d9")
        self.recomendedcrop.configure(disabledforeground="#a3a3a3")
        self.recomendedcrop.configure(font=font11)
        self.recomendedcrop.configure(foreground="#ff0000")
        self.recomendedcrop.configure(text="Note: All Steps are nested ")
        self.recomendedcrop.configure(width=154)




if __name__ == '__main__':
    vp_start_gui()