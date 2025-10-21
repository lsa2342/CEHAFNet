from bytecode import Bytecode, Instr


class get_local(object):
    cache = {}  # ����һ�������ֵ����ڴ洢����ֵ���Լ�һ����� is_activate ���ڼ����
    is_activate = False

    def __init__(self, *varnames):
        """varname: tuple"""
        self.varnames = varnames  # �洢��Ҫ���ٵľֲ�������

    def __call__(self, func):
        if not type(self).is_activate:  # �������δ���ֱ�ӷ���ԭʼ����
            return func

        # ��ȡģ�顢��ͺ��������ƣ����ڹ���Ψһ��ʶ��
        module_name = func.__module__  # ��ȡģ����
        class_name = func.__qualname__.split('.')[0]  # ��ȡ����
        func_name = func.__qualname__.split('.')[-1]  # ��ȡ������

        c = Bytecode.from_code(func.__code__)  # �Ӻ������봴���ֽ������

        extra_code = [Instr('STORE_FAST', '_res')]  # store return variable

        # store local variables
        for var_name in self.varnames:
            full_name = f"{module_name}.{class_name}.{func_name}.{var_name}"  # ʹ�����������ƣ�ģ��.��.����.����������Ϊ�����
            type(self).cache[full_name] = []  # create cache

            # ���μ��غʹ洢�ֲ��������µı����� var_name + '_value'
            extra_code.extend([Instr('LOAD_FAST', var_name),
                               Instr('STORE_FAST', var_name + '_value')])

        # push to TOS
        extra_code.extend([Instr('LOAD_FAST', '_res')])  # ����� _res ѹ��ջ��

        for var_name in self.varnames:  # ���ֲ�����ֵҲѹ��ջ
            extra_code.extend([Instr('LOAD_FAST', var_name + '_value')])

        # ������������ֵ�;ֲ�������Ԫ�鲢�洢�� _result_tuple ��
        extra_code.extend([
            Instr('BUILD_TUPLE', 1 + len(self.varnames)),
            Instr('STORE_FAST', '_result_tuple'),
            Instr('LOAD_FAST', '_result_tuple')
        ])

        # ���µ��ֽ������ԭʼ�����ֽ�����
        c[-1:-1] = extra_code
        func.__code__ = c.to_code()

        # # ����һ����װ��������������ִ��ʱ���ֲ�������ֵ���뻺��
        def wrapper(*args, **kwargs):
            res, *values = func(*args, **kwargs)  # ����ԭʼ���������ؽ���ͱ���ֵ
            for var_idx in range(len(self.varnames)):
                value = values[var_idx].detach().cpu().numpy()  # ��ȡ��ת������ֵ���� detach ��ת��Ϊ numpy ��ʽ��
                full_name = f"{module_name}.{class_name}.{func_name}.{self.varnames[var_idx]}"  # ����Ψһ��ʶ����ֵ���뻺��
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
    cache = {}  # ����һ�������ֵ����ڴ洢����ֵ���Լ�һ����� is_activate ���ڼ����

    def __init__(self, *vars: str):
        """varname: tuple"""
        self.vars = vars  # �洢��Ҫ���ٵľֲ�������

    def __call__(self, func):
        if not type(self).is_activate:  # �������δ���ֱ�ӷ���ԭʼ����
            return func

        # ��ȡģ�顢��ͺ��������ƣ����ڹ���Ψһ��ʶ��
        module_name = func.__module__  # ��ȡģ����
        class_name = func.__qualname__.split('.')[0]  # ��ȡ����
        func_name = func.__qualname__.split('.')[-1]  # ��ȡ������

        c = Bytecode.from_code(func.__code__)  # �Ӻ������봴���ֽ������

        extra_code = [Instr('STORE_FAST', '_res')]  # store return variable

        # store local variables
        for var_name in self.vars:
            full_name = f"{module_name}.{class_name}.{func_name}.{var_name}"  # ʹ�����������ƣ�ģ��.��.����.����������Ϊ�����
            type(self).cache[full_name] = []  # create cache

            # ���μ��غʹ洢�ֲ��������µı����� var_name + '_value'
            extra_code.extend([Instr('LOAD_FAST', var_name),
                               Instr('STORE_FAST', var_name + '_value')])

        # push to TOS
        extra_code.extend([Instr('LOAD_FAST', '_res')])  # ����� _res ѹ��ջ��

        for var_name in self.vars:  # ���ֲ�����ֵҲѹ��ջ
            extra_code.extend([Instr('LOAD_FAST', var_name + '_value')])

        # ������������ֵ�;ֲ�������Ԫ�鲢�洢�� _result_tuple ��
        extra_code.extend([
            Instr('BUILD_TUPLE', 1 + len(self.varnames)),
            Instr('STORE_FAST', '_result_tuple'),
            Instr('LOAD_FAST', '_result_tuple')
        ])

        # ���µ��ֽ������ԭʼ�����ֽ�����
        c[-1:-1] = extra_code
        func.__code__ = c.to_code()

        # # ����һ����װ��������������ִ��ʱ���ֲ�������ֵ���뻺��
        def wrapper(*args, **kwargs):
            res, *values = func(*args, **kwargs)  # ����ԭʼ���������ؽ���ͱ���ֵ
            for var_idx in range(len(self.vars)):
                value = values[var_idx].detach().cpu().numpy()  # ��ȡ��ת������ֵ���� detach ��ת��Ϊ numpy ��ʽ��
                full_name = f"{module_name}.{class_name}.{func_name}.{self.vars[var_idx]}"  # ����Ψһ��ʶ����ֵ���뻺��
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