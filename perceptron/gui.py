import tkinter
import numpy as np


class GUI:
    black_color = "black"
    white_color = "white"

    def __init__(self, predict_method):
        w = self.generate_buttons()
        self.vector = np.array([0] * 7 * 5)
        self.predict_method = predict_method
        tkinter.mainloop()
        self.buttons = []

    def clear(self):
        self.vector = np.array([0] * 7 * 5)
        for button in self.buttons:
            button["bg"] = self.white_color

    def generate_buttons(self, c=5, r=7):
        self.buttons = []
        window = tkinter.Tk()
        last_clicked = [None]
        predict_button = tkinter.Button(window, bg=self.white_color, activebackground=self.white_color, height=1,
                                        width=25,
                                        text='predict')
        predict_button["command"] = self.predict
        predict_button.grid(column=7, row=0)

        clear_button = tkinter.Button(window, bg=self.white_color, activebackground=self.white_color, height=1,
                                      width=25,
                                      text='clear')
        clear_button["command"] = self.clear
        clear_button.grid(column=21, row=0)
        for x in range(c):
            for y in range(r):
                self.buttons.append(
                    tkinter.Button(window, bg=self.white_color, activebackground=self.white_color, height=4,
                                   width=4))
                self.buttons[-1].grid(column=x, row=y + 1)
                self.buttons[-1].i_x = x
                self.buttons[-1].i_y = y
                self.buttons[-1]["command"] = lambda b=self.buttons[-1]: self.click(b, last_clicked, n_columns=c)
        return window

    def predict(self):   
        print(self.predict_method(self.vector))

    def click(self, button, last_clicked, n_columns):
        if last_clicked[0]:
            last_clicked[0]["bg"] = self.white_color
            last_clicked[0]["activebackground"] = self.white_color
        if button["bg"] == self.white_color:
            button["bg"] = self.black_color
        else:
            button["bg"] = self.white_color
        button["activebackground"] = self.black_color
        old_value = self.vector[button.i_y * n_columns + button.i_x]
        self.vector[button.i_y * n_columns + button.i_x] = abs(old_value - 1)
  