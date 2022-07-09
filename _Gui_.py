import Tkinter as tkinter
from mpi4py import MPI
import numpy as np

class SimpleGui:
	NLines = 100
	Nrows = 100
	Buf_0 = np.empty(3, dtype='i')
	def __init__(self, master, _MAINCAM_RANK_, _PLOT_RANK_, Comm):
		self._MAINCAM_RANK_ = _MAINCAM_RANK_
		self._PLOT_RANK_ = _PLOT_RANK_
		self.Comm = Comm

		self.master = master
		self.master.title("mvp1")

		self.Master_label = tkinter.Label(master, text="master label")
		self.Master_label.grid(row = 0, column = 50, columnspan = 10)

		self.Recogn_button = tkinter.Button(master, text = "Reconhecimento", command = self.FaceRecog)
		self.Recogn_button.grid(row = 20, column = 50, columnspan = 10)

		self.Time_box = tkinter.Text(master, height=2, width=20)
		self.Time_box.grid(row = 50, column = 45, columnspan = 10, sticky = tkinter.E)

		self.timer_str = tkinter.StringVar()
		self.timer_str.set("start timer")
		self.timer_button = tkinter.Button(master, textvariable = self.timer_str, command = self.StartTimer)
		self.timer_button.grid(row = 50, column = 55, columnspan = 10, sticky = tkinter.W)
		self.master.bind("<Return>", self.EventStartTimer)
		self.master.bind("<s>", self.EventStopTimer)

		self.Close_button = tkinter.Button(master, text = "quit", command = self.Quit)
		self.Close_button.grid(row = 80, column = 50, columnspan = 10)
		self.master.bind("<q>", self.EventQuit)

	def FaceRecog(self):
		self.Buf_0[0] = -2
		self.Comm.Send([self.Buf_0, self.Buf_0.size, MPI.INT], dest = self._MAINCAM_RANK_, tag = 0)

	def StartTimer(self):
		if "start timer" == self.timer_str.get():
			temp = self.Time_box.get("1.0", tkinter.END)
			self.Time_box.delete("1.0", tkinter.END)
			try:
				self.Buf_0[0] = int(float(temp)*1000)
			except:
				print "not a number"
				return
			self.timer_str.set("stop timer (t = "+str(self.Buf_0[0]/1000.)+" sec)")
		else:
			self.Buf_0[0] = 100000000
			self.timer_str.set("start timer")
		self.Comm.Send([self.Buf_0, self.Buf_0.size, MPI.INT], dest = self._MAINCAM_RANK_, tag = 0)

	def EventStartTimer(self, event):
		if "start timer" == self.timer_str.get():
			self.StartTimer()

	def EventStopTimer(self, event):
		if "start timer" != self.timer_str.get():
			self.Buf_0[0] = 100000000
			self.timer_str.set("start timer")
			self.Comm.Send([self.Buf_0, self.Buf_0.size, MPI.INT], dest = self._MAINCAM_RANK_, tag = 0)

	def Quit(self):
		self.Buf_0[0] = -1
		self.Comm.Send([self.Buf_0, self.Buf_0.size, MPI.INT], dest = self._MAINCAM_RANK_, tag = 0)
		self.Comm.Send([self.Buf_0, self.Buf_0.size, MPI.INT], dest = self._PLOT_RANK_, tag = 0)
		self.master.quit()

	def EventQuit(self, event):
		self.Quit()
