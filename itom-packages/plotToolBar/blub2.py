class A():
    def __init__(self):
        import __main__
        print(__main__.__dict__["x"])
        # print(x)
        pass
    
    def p():
        print("Base class A")