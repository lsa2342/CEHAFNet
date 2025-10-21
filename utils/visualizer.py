from bytecode import Bytecode, Instr


class get_local(object):
    cache = {}  # 创建一个缓存字典用于存储变量值，以及一个标记 is_activate 用于激活功能
    is_activate = False

    def __init__(self, *varnames):
        """varname: tuple"""
        self.varnames = varnames  # 存储需要跟踪的局部变量名

    def __call__(self, func):
        if not type(self).is_activate:  # 如果功能未激活，直接返回原始函数
            return func

        # 获取模块、类和函数的名称，用于构建唯一标识符
        module_name = func.__module__  # 获取模块名
        class_name = func.__qualname__.split('.')[0]  # 获取类名
        func_name = func.__qualname__.split('.')[-1]  # 获取函数名

        c = Bytecode.from_code(func.__code__)  # 从函数代码创建字节码对象

        extra_code = [Instr('STORE_FAST', '_res')]  # store return variable

        # store local variables
        for var_name in self.varnames:
            full_name = f"{module_name}.{class_name}.{func_name}.{var_name}"  # 使用完整的名称（模块.类.函数.变量名）作为缓存键
            type(self).cache[full_name] = []  # create cache

            # 依次加载和存储局部变量到新的变量名 var_name + '_value'
            extra_code.extend([Instr('LOAD_FAST', var_name),
                               Instr('STORE_FAST', var_name + '_value')])

        # push to TOS
        extra_code.extend([Instr('LOAD_FAST', '_res')])  # 将结果 _res 压入栈顶

        for var_name in self.varnames:  # 将局部变量值也压入栈
            extra_code.extend([Instr('LOAD_FAST', var_name + '_value')])

        # 构建包含返回值和局部变量的元组并存储在 _result_tuple 中
        extra_code.extend([
            Instr('BUILD_TUPLE', 1 + len(self.varnames)),
            Instr('STORE_FAST', '_result_tuple'),
            Instr('LOAD_FAST', '_result_tuple')
        ])

        # 将新的字节码插入原始函数字节码中
        c[-1:-1] = extra_code
        func.__code__ = c.to_code()

        # # 定义一个包装器函数，用于在执行时将局部变量的值存入缓存
        def wrapper(*args, **kwargs):
            res, *values = func(*args, **kwargs)  # 调用原始函数并返回结果和变量值
            for var_idx in range(len(self.varnames)):
                value = values[var_idx].detach().cpu().numpy()  # 获取并转换变量值（先 detach 再转换为 numpy 格式）
                full_name = f"{module_name}.{class_name}.{func_name}.{self.varnames[var_idx]}"  # 根据唯一标识符将值存入缓存
                type(self).cache[full_name].append(value)
            return res

        return wrapper

    @classmethod
    def clear(cls):
        for key in cls.cache.keys():
            cls.cache[key] = []

    @classmethod
    def activate(cls):
        cls.is_activate = True


class vis(object):
    is_activate = False
    cache = {}  # 创建一个缓存字典用于存储变量值，以及一个标记 is_activate 用于激活功能

    def __init__(self, *vars: str):
        """varname: tuple"""
        self.vars = vars  # 存储需要跟踪的局部变量名

    def __call__(self, func):
        if not type(self).is_activate:  # 如果功能未激活，直接返回原始函数
            return func

        # 获取模块、类和函数的名称，用于构建唯一标识符
        module_name = func.__module__  # 获取模块名
        class_name = func.__qualname__.split('.')[0]  # 获取类名
        func_name = func.__qualname__.split('.')[-1]  # 获取函数名

        c = Bytecode.from_code(func.__code__)  # 从函数代码创建字节码对象

        extra_code = [Instr('STORE_FAST', '_res')]  # store return variable

        # store local variables
        for var_name in self.vars:
            full_name = f"{module_name}.{class_name}.{func_name}.{var_name}"  # 使用完整的名称（模块.类.函数.变量名）作为缓存键
            type(self).cache[full_name] = []  # create cache

            # 依次加载和存储局部变量到新的变量名 var_name + '_value'
            extra_code.extend([Instr('LOAD_FAST', var_name),
                               Instr('STORE_FAST', var_name + '_value')])

        # push to TOS
        extra_code.extend([Instr('LOAD_FAST', '_res')])  # 将结果 _res 压入栈顶

        for var_name in self.vars:  # 将局部变量值也压入栈
            extra_code.extend([Instr('LOAD_FAST', var_name + '_value')])

        # 构建包含返回值和局部变量的元组并存储在 _result_tuple 中
        extra_code.extend([
            Instr('BUILD_TUPLE', 1 + len(self.varnames)),
            Instr('STORE_FAST', '_result_tuple'),
            Instr('LOAD_FAST', '_result_tuple')
        ])

        # 将新的字节码插入原始函数字节码中
        c[-1:-1] = extra_code
        func.__code__ = c.to_code()

        # # 定义一个包装器函数，用于在执行时将局部变量的值存入缓存
        def wrapper(*args, **kwargs):
            res, *values = func(*args, **kwargs)  # 调用原始函数并返回结果和变量值
            for var_idx in range(len(self.vars)):
                value = values[var_idx].detach().cpu().numpy()  # 获取并转换变量值（先 detach 再转换为 numpy 格式）
                full_name = f"{module_name}.{class_name}.{func_name}.{self.vars[var_idx]}"  # 根据唯一标识符将值存入缓存
                type(self).cache[full_name].append(value)
            return res

        return wrapper

    @classmethod
    def clear(cls):
        for key in cls.cache.keys():
            cls.cache[key] = []

    @classmethod
    def activate(cls):
        cls.is_activate = True