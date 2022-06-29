from tkinter import *
from tkinter import ttk
window=Tk()
window.geometry("300x300")
window.title("Road accident prediction model")
label=Label(window,text="Road accident prediction").pack()
label1=Label(window,text="Carriage Hazards").place(x=40,y=80)
values = {'Present':'1','None':'2'}
s = 80
for i,j in values.items():
    Radiobutton(window,text=i,value = j).place(x=40,y = s+20)
    s+=20
label2=Label(window,text="Light Conditions").place(x=40,y=160)
values1 = {'Light Present':'1','Darkness':'2'}
s = 160
for i,j in values1.items():
    Radiobutton(window,text=i,value = j).place(x=40,y = s+20)
    s+=20

n = StringVar()
label3 = Label(window,text='Day of the week').place(x=40,y=240)
monthchoosen = ttk.Combobox(window, width=27, textvariable=n)

monthchoosen['values'] = (' Sunday',
                          ' Monday',
                          ' Tuesday',
                          ' Wednsday',
                          ' Thursday',
                          ' Friday',
                          ' Saturday')
monthchoosen.place(x=40, y=260)
monthchoosen.current(1)

n = StringVar()
label4 = Label(window,text='Special conditions at site').place(x=40,y=300)
conditionchoosen = ttk.Combobox(window, width=27, textvariable=n)

conditionchoosen['values'] =(' None',
                          ' Roadworks',
                          ' Oil or diesel',
                          ' Mud',
                          ' Road surface defective',
                          ' Auto traffic signal - out',
                          ' Road sign or marking defective or obscured',
                            "Auto signal part defective",
                             "Data missing or out of range"
                             )
conditionchoosen.place(x=40, y=320)
conditionchoosen.current(0)


window.mainloop()


