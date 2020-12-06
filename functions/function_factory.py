from .sphere_function import SphereFunction
from .rastrigin_function import RastriginFunction
from .rosenbrock_function import RosenbrockFunction
from .ackley_function import AckleyFunction
from .eggholder_function import EggHolderFunction
from .easom_function import EasomFunction

class FunctionFactory:
    
    instance = None
    
    @staticmethod
    def getInstance():
        if FunctionFactory.instance is None:
            FunctionFactory.instance = FunctionFactory()
        return FunctionFactory.instance
    
    FUNCTIONS = {
        'sphere': SphereFunction,
        'rastrigin': RastriginFunction,
        'rosenbrock': RosenbrockFunction,
        'ackley': AckleyFunction,
        'eggholder': EggHolderFunction,
        'easom': EasomFunction,
    }
    
    def getFunction(self, func_name, num_var):
        func_name = func_name.lower()
        assert func_name in FunctionFactory.FUNCTIONS, "Function \"{}\" is not supported.".format(func_name)
        return FunctionFactory.FUNCTIONS[func_name](num_var)