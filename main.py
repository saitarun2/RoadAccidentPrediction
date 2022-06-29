from tkinter import *
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

window.mainloop()


