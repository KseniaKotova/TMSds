import sys

if len(sys.argv) == 3:
	try:
		num1 = int(sys.argv[1])
		num2 = int(sys.argv[2])
		print(f"Sum: {num1 + num2}")
	except ValueError:
		print("Entry numbers")
else:
	print("Usage: python sum.py <num1> <num2>")
