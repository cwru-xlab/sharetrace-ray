import functools
from multiprocessing import Process


def actor(func):
	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		p = Process(target=func, args=args, kwargs=kwargs)
		p.start()
		p.join()

	return wrapper


def g(q):
	pass


def f(q):
	pass


def t(m):
	print(type(m))


if __name__ == '__main__':
	class Message:
		def __init__(self):
			self.a = 1
			self.b = 2


	t(Message)
