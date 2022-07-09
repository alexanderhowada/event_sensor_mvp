import sqlite3

class _SQLite_:

	def __init__(self, DbName):
		self.Connection = sqlite3.connect(DbName)
		self.cursor = self.Connection.cursor()

	def __del__(self):
		self.Connection.close()

	def Exec(self, Command):
		temp = self.cursor.execute(Command)
		Values = []
		for line in temp:
			Values.append(line)
		return Values

	def Commit(self):
		self.Connection.commit()
